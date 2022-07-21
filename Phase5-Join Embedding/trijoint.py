import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.models as models
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()


# # =============================================================================

class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()

    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y


def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class cross_modal_discriminator(nn.Module):
    def __init__(self):
        super(cross_modal_discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opts.embDim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, fea):
        output = self.model(fea)

        return output
# Skip-thoughts LSTM
class stRNN(nn.Module):
    def __init__(self):
        super(stRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=opts.stDim, hidden_size=opts.srnnDim, bidirectional=False, batch_first=True)

    def forward(self, x, sq_lengths):
        # here we use a previous LSTM to get the representation of each instruction
        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx \
            .view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the lstm
        out, hidden = self.lstm(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        unsorted_idx = original_idx.view(-1, 1, 1).expand_as(unpacked)
        # we get the last index of each sequence in the batch
        idx = (sq_lengths - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        # we sort and get the last element of each sequence
        output = unpacked.gather(0, unsorted_idx.long()).gather(1, idx.long())
        output = output.view(output.size(0), output.size(1) * output.size(2))

        return output

# Im2recipe model
class im2recipe(nn.Module):
    def __init__(self):
        super(im2recipe, self).__init__()
        if opts.preModel == 'resNet50':

            resnet = models.resnext101_32x8d(pretrained=True)
            modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
            self.visionMLP = nn.Sequential(*modules)

            self.visual_embedding = nn.Sequential(
                nn.Linear(opts.imfeatDim + 300, opts.embDim),
                nn.BatchNorm1d(opts.embDim),
                nn.LeakyReLU(0.02, inplace=True),
            )

            self.tfidf_embedding = nn.Sequential(
                nn.Linear(300, 1024),
                nn.LeakyReLU(0.02, inplace=True),
                )

            self.recipe_embedding = nn.Sequential(
                nn.Linear(1024 + 1024, opts.embDim),
                nn.BatchNorm1d(opts.embDim),
                nn.LeakyReLU(0.02, inplace=True),
            )

        else:
            raise Exception('Only resNet50 model is implemented.')

        self.stRNN_ = stRNN()
        self.table = TableModule()

        if opts.semantic_reg:
            self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

    def forward(self, x, x2, y, z1, z2):  # we need to check how the input is going to be provided to the model
        # recipe embedding
        tfidf_emb = self.tfidf_embedding(y)  # joining on the last dim
        recipe_emb = self.table([self.stRNN_(z1, z2), tfidf_emb], 1)
        recipe_emb = self.recipe_embedding(recipe_emb)
        recipe_emb = norm(recipe_emb)

        # visual embedding
        x1 = self.visionMLP(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = self.table([x1, x2], 1)
        visual_emb = self.visual_embedding(x)
        visual_emb = norm(visual_emb)

        if opts.semantic_reg:
            visual_sem = self.semantic_branch(visual_emb)
            recipe_sem = self.semantic_branch(recipe_emb)
            # final output
            output = [visual_emb, recipe_emb, visual_sem, recipe_sem]
        else:
            # final output
            output = [visual_emb, recipe_emb]

        return output