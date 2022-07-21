import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import ImagerLoader
from args import get_parser
from trijoint import *
import datetime
from test import do_test
from triplet_loss import *
from test import do_test

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

if not (torch.cuda.device_count()):
    device = torch.device(*('cpu', 0))
else:
    torch.cuda.manual_seed(opts.seed)
    device = torch.device(*('cuda', 0))

result_dir = opts.result_dir
print(result_dir)
print('batch_size:', opts.batch_size)
print('workers:', opts.workers)
print('gama:', opts.gama)
print('margin:', opts.margin)

triplet_loss = Soft_margin_Loss(device, gama=opts.gama, margin=opts.margin)
cm_discriminator = torch.nn.DataParallel(cross_modal_discriminator().cuda())
optimizer_cmD = torch.optim.Adam(cm_discriminator.parameters(), lr=opts.lr, betas=(0.5, 0.999))


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = torch.log(D(interpolates))
    fake = torch.autograd.Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,  # fack samples
        inputs=interpolates,  # real samples
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def main():
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    model.to(device)
    # define loss function (criterion) and optimizer
    if opts.semantic_reg:
        weights_class = torch.Tensor(opts.numClasses).fill_(1)
        # CrossEntropyLoss combines LogSoftMax and NLLLoss in one single class
        class_crit = nn.CrossEntropyLoss(weight=weights_class).to(device)
        # we will use two different criteria
        criterion = [triplet_loss, class_crit]
    else:
        criterion = triplet_loss

    # # creating different parameter groups
    vision_params = list(map(id, model.visionMLP.parameters()))
    base_params = filter(lambda p: id(p) not in vision_params, model.parameters())

    # optimizer - with lr initialized accordingly

    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.visionMLP.parameters(), 'lr': opts.lr}
    ], lr=opts.lr)

    if opts.resume:
        if os.path.isfile(opts.resume):
            print("=> loading checkpoint '{}'".format(opts.resume))
            checkpoint = torch.load(opts.resume)
            opts.start_epoch = checkpoint['epoch']
            best_val_i2t = checkpoint['best_val_i2t']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_epoch = checkpoint['epoch']
            best_sum_recall = checkpoint['best_sum_recall']
            best_epoch_medr = checkpoint['best_epoch_medr']

            print('best_sum_recall:', best_sum_recall)
            print("(epoch {}) - best_val_i2t: "
                  .format(checkpoint['epoch']), best_val_i2t)
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))
            best_epoch = 0
            best_val_i2t = {1: 0.0, 5: 0.0, 10: 0.0}
            best_sum_recall = 0
            best_epoch_medr = 0.0

    else:
        best_epoch = 0
        best_val_i2t = {1: 0.0, 5: 0.0, 10: 0.0}
        best_sum_recall = 0
        best_epoch_medr = 0.0

        # models are save only when their loss obtains the best value in the validation
    print('There are %d parameter groups' % len(optimizer.param_groups))
    print('Initial base params lr: %f' % optimizer.param_groups[0]['lr'])
    print('Initial vision params lr: %f' % optimizer.param_groups[1]['lr'])

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    cudnn.benchmark = True

    # preparing the training laoder
    train_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
                     transforms.Compose([
                         transforms.Scale(256),  # rescale the image keeping the original aspect ratio
                         transforms.CenterCrop(256),  # we get only the center of that rescaled
                         transforms.RandomCrop(224),  # random crop within the center crop
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         normalize,
                     ]), data_path=opts.data_path, partition='train', sem_reg=opts.semantic_reg),
        batch_size=opts.batch_size, shuffle=True,
        num_workers=opts.workers, pin_memory=True)
    print('Training loader prepared.')

    # preparing validation loader
    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
                     transforms.Compose([
                         transforms.Scale(256),  # rescale the image keeping the original aspect ratio
                         transforms.CenterCrop(224),  # we get only the center of that rescaled
                         transforms.ToTensor(),
                         normalize,
                     ]), data_path=opts.data_path, sem_reg=opts.semantic_reg, partition='val'),
        batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.workers, pin_memory=True)
    print('Validation loader prepared.')

    # run epochs
    stage = 1
    valtrack = 0
    train_start_time = datetime.datetime.now()
    print(train_start_time)
    for epoch in range(opts.start_epoch, opts.epochs):
        if valtrack == 10 and stage == 2:
            print('training complete!')
            print(datetime.datetime.now())
            print('it takes {} for the training'.format(datetime.datetime.now() - train_start_time))
            break

        if valtrack == 3 and stage == 1:
            stage = 2
            valtrack = 0
            optimizer.param_groups[0]['lr'] = 0.00001
            optimizer.param_groups[1]['lr'] = 0.00001
            optimizer_cmD.param_groups[0]['lr'] = 0.00001
            print(' base params lr: %f' % optimizer.param_groups[0]['lr'])
            print(' vision params lr: %f' % optimizer.param_groups[1]['lr'])
            print(' cmD parameters lr: %f' % optimizer_cmD.param_groups[0]['lr'])

        start_time = datetime.datetime.now()
        train(train_loader, model, criterion, optimizer, epoch)
        print('it takes {} for the training epoch'.format(datetime.datetime.now() - start_time))

        start_time = datetime.datetime.now()
        medR, recall_i2t = validate(val_loader, model, criterion)
        print('it takes {} for the validation'.format(datetime.datetime.now() - start_time))
        # sum_recall = 0.0
        sum_recall = recall_i2t[1] + 0.8 * recall_i2t[5] + 0.6 * recall_i2t[10]

        print('current sum recall:', sum_recall)
        if sum_recall > best_sum_recall:
            best_sum_recall = sum_recall
            best_val_i2t = recall_i2t
            best_epoch_medr = medR
            best_epoch = epoch
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_i2t': best_val_i2t,
                'best_epoch_medr': best_epoch_medr,
                'best_sum_recall': best_sum_recall,
                'optimizer': optimizer.state_dict(),
            }, True)

            ma_filename = opts.snapshots + '{}_cm.pth.tar'.format(result_dir)
            torch.save(cm_discriminator.state_dict(), ma_filename)

            valtrack = 0
        else:
            valtrack += 1

        print('** best_epoch: %d - valtrack: %d - best_sum_recall_i2t: %f' % (best_epoch, valtrack, best_sum_recall))
        print('** best_epoch_medr: %f - best_val_i2t:' % best_epoch_medr, best_val_i2t)
        print('_____________________________________________________________________________')
        print(datetime.datetime.now())
    do_test()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    tri_losses = AverageMeter()
    losses = AverageMeter()
    cm_losses = AverageMeter()

    if opts.semantic_reg:
        img_losses = AverageMeter()
        rec_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = list()
        for j in range(len(input)):
            # if j>1:
            input_var.append(input[j].to(device))
            # else:
            # input_var.append(input[j].to(device))

        target_var = list()
        for j in range(len(target)):
            target_var.append(target[j].to(device))

        # compute output
        output = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4])
        img_id_fea = output[0]
        rec_id_fea = output[1]

        real_validity = cm_discriminator(img_id_fea.detach())
        fake_validity = cm_discriminator(rec_id_fea.detach())
        gradient_penalty = compute_gradient_penalty(cm_discriminator, img_id_fea.detach(), rec_id_fea.detach())
        loss_cmD = torch.log(1 - torch.mean(real_validity)) + torch.log(
            torch.mean(fake_validity)) + 10 * gradient_penalty
        optimizer_cmD.zero_grad()
        loss_cmD.backward()
        optimizer_cmD.step()

        g_fake_validity = cm_discriminator(rec_id_fea)
        loss_cmG = torch.log(1 - torch.mean(g_fake_validity))
        # compute loss
        if opts.semantic_reg:
            tri_loss = global_loss(triplet_loss, torch.cat((img_id_fea, rec_id_fea)))[0]
            img_loss = criterion[1](output[2], target_var[1])
            rec_loss = criterion[1](output[3], target_var[2])
            # combined loss
            loss = tri_loss + \
                   0.005 * img_loss + \
                   0.005 * rec_loss + 0.005 * loss_cmG

            # measure performance and record losses
            tri_losses.update(tri_loss.data, input[0].size(0))
            img_losses.update(img_loss.data, input[0].size(0))
            rec_losses.update(rec_loss.data, input[0].size(0))
            losses.update(loss.data, input[0].size(0))
            cm_losses.update(loss_cmG.data, input[0].size(0))
        else:
            loss = criterion(output[0], output[1], target_var[0].float())
            # measure performance and record loss
            tri_losses.update(loss.data, input[0].size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if opts.semantic_reg:
        print('Epoch: {0}\t'
              'triplet loss {triplet_loss.val:.4f} ({triplet_loss.avg:.4f})\t'
              'img Loss {img_loss.val:.4f} ({img_loss.avg:.4f})\t'
              'rec loss {rec_loss.val:.4f} ({rec_loss.avg:.4f})\t'
              'vision ({visionLR}) - recipe ({recipeLR})\t'.format(
            epoch, triplet_loss=tri_losses, img_loss=img_losses,
            rec_loss=rec_losses, visionLR=optimizer.param_groups[1]['lr'],
            recipeLR=optimizer.param_groups[0]['lr']))
        print('cmG loss {cm_loss.val:.4f} ({cm_loss.avg:.4f})\t'
              'loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            cm_loss=cm_losses, loss=losses))
    else:
        print('Epoch: {0}\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'vision ({visionLR}) - recipe ({recipeLR})\t'.format(
            epoch, loss=tri_losses, visionLR=optimizer.param_groups[1]['lr'],
            recipeLR=optimizer.param_groups[0]['lr']))


def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input_var = list()
            for j in range(len(input)):
                # input_var.append(torch.autograd.Variable(input[j], volatile=True).cuda())
                input_var.append(input[j].to(device))
            target_var = list()
            for j in range(len(target) - 2):  # we do not consider the last two objects of the list
                target[j] = target[j].to(device)
                target_var.append(target[j].to(device))

            # compute output
            output = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4])
            img_emd_modal = output[0]
            recipe_emb_modal = output[1]

            if i == 0:
                # data0 = output[0].data.cpu().numpy()
                # data1 = output[1].data.cpu().numpy()
                data0 = img_emd_modal.data.cpu().numpy()
                data1 = recipe_emb_modal.data.cpu().numpy()
                data2 = target[-2]
                data3 = target[-1]
            else:
                # data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
                # data1 = np.concatenate((data1, output[1].data.cpu().numpy()), axis=0)
                data0 = np.concatenate((data0, img_emd_modal.data.cpu().numpy()), axis=0)
                data1 = np.concatenate((data1, recipe_emb_modal.data.cpu().numpy()), axis=0)
                data2 = np.concatenate((data2, target[-2]), axis=0)
                data3 = np.concatenate((data3, target[-1]), axis=0)

    medR, recall = rank(opts, data0, data1, data2)
    print('* Val medR {medR:.4f}\t'
          'Recall {recall}'.format(medR=medR, recall=recall))

    return medR, recall


def rank(opts, img_embeds, rec_embeds, rec_ids):
    random.seed(opts.seed)
    type_embedding = opts.embtype
    im_vecs = img_embeds
    instr_vecs = rec_embeds
    names = rec_ids

    # Sort based on names to always pick same samples for medr
    idxs = np.argsort(names)
    names = names[idxs]

    # Ranker
    N = opts.medr
    idxs = range(N)

    glob_rank = []
    glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    for i in range(10):

        ids = random.sample(range(0, len(names)), N)
        im_sub = im_vecs[ids, :]
        instr_sub = instr_vecs[ids, :]
        ids_sub = names[ids]

        med_rank = []
        recall = {1: 0.0, 5: 0.0, 10: 0.0}

        for ii in idxs:
            distance = {}
            for j in range(N):
                distance[j] = np.linalg.norm(im_sub[ii] - instr_sub[j])
            distance_sorted = sorted(distance.items(), key=lambda x: x[1])
            pos = np.where(np.array(distance_sorted) == distance[ii])[0][0]

            if (pos + 1) == 1:
                recall[1] += 1
            if (pos + 1) <= 5:
                recall[5] += 1
            if (pos + 1) <= 10:
                recall[10] += 1

            # store the position
            med_rank.append(pos + 1)

        for i in recall.keys():
            recall[i] = recall[i] / N

        med = np.median(med_rank)
        # print "median", med

        for i in recall.keys():
            glob_recall[i] += recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = round(glob_recall[i] / 10, 3)

    return np.average(glob_rank), glob_recall


def save_checkpoint(state, is_best):
    filename = opts.snapshots + '{}.pth.tar'.format(result_dir)
    if is_best:
        print('model_e%03d.pth.tar' % (state['epoch']))
        torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()