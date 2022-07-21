import time
import torch.nn.parallel
import torch.optim
import torch.utils.data
from data_loader import ImagerLoader  # our data_loader
import numpy as np
from trijoint import *
import pickle
from args import get_parser

import os
import random
# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.manual_seed(opts.seed)

np.random.seed(opts.seed)

if not (torch.cuda.device_count()):
    device = torch.device(*('cpu', 0))
else:
    torch.cuda.manual_seed(opts.seed)
    device = torch.device(*('cuda', 0))


def improved_rank_emb(result_path, test_size):
    random.seed(1234)
    with open(result_path + 'img_embeds.pkl', 'rb') as f:
        im_vecs = pickle.load(f)
    with open(result_path + 'rec_embeds.pkl', 'rb') as f:
        instr_vecs = pickle.load(f)
    with open(result_path + 'rec_ids.pkl', 'rb') as f:
        names = pickle.load(f)
    # Sort based on names to always pick same samples for medr
    idxs = np.argsort(names)
    names = names[idxs]

    # Ranker
    N = test_size
    idxs = range(N)

    results = ''
    glob_med = []
    glob_r2i_med = []

    glob_recall = {1: 0.0, 5: 0.0, 10: 0.0}
    glob_r2i_recall = {1: 0.0, 5: 0.0, 10: 0.0}

    for i in range(10):
        results += 'the {}th test \n'.format(i)
        ids = random.sample(range(0, len(names)), N)
        im_sub = im_vecs[ids, :]
        instr_sub = instr_vecs[ids, :]

        i2r_dist = np.zeros((N, N))
        recall = {1: 0.0, 5: 0.0, 10: 0.0}
        r2i_recall = {1: 0.0, 5: 0.0, 10: 0.0}
        rank_list = []
        r2i_rank_list = []

        for ii in idxs:
            distance = {}
            for j in range(N):
                distance[j] = np.linalg.norm(im_sub[ii] - instr_sub[j])  # for im2recipe
            i2r_dist[ii] = list(distance.values())
            distance_sorted = sorted(distance.items(), key=lambda x: x[1])
            pos = np.where(np.array(distance_sorted) == distance[ii])[0][0]

            if (pos + 1) == 1:
                recall[1] += 1
            if (pos + 1) <= 5:
                recall[5] += 1
            if (pos + 1) <= 10:
                recall[10] += 1

            # store the position
            rank_list.append(pos + 1)

        for i in recall.keys():
            recall[i] = (recall[i] + 0.0) / N

        med = np.median(rank_list)
        # print("median", med)
        results += 'i2r median: {} \t'.format(med)
        results += 'i2r recall: {} \n'.format(recall)

        for i in recall.keys():
            glob_recall[i] += recall[i]
        glob_med.append(med)

        r2i_dist = i2r_dist.T
        for ii in idxs:
            distance = {}
            for j in range(N):
                distance[j] = r2i_dist[ii][j]
            distance_sorted = sorted(distance.items(), key=lambda x: x[1])
            pos = np.where(np.array(distance_sorted) == distance[ii])[0][0]

            if (pos + 1) == 1:
                r2i_recall[1] += 1
            if (pos + 1) <= 5:
                r2i_recall[5] += 1
            if (pos + 1) <= 10:
                r2i_recall[10] += 1

            # store the position
            r2i_rank_list.append(pos + 1)

        for i in r2i_recall.keys():
            r2i_recall[i] = (r2i_recall[i] + 0.0) / N

        r2i_med = np.median(r2i_rank_list)
        # print("median", med)
        results += 'r2i median: {} \t'.format(r2i_med)
        results += 'r2i recall: {} \n'.format(r2i_recall)

        for i in r2i_recall.keys():
            glob_r2i_recall[i] += r2i_recall[i]
        glob_r2i_med.append(r2i_med)

    for i in glob_recall.keys():
        glob_recall[i] = round(glob_recall[i] / 10, 3)
    final_med = np.average(glob_med)
    for i in glob_r2i_recall.keys():
        glob_r2i_recall[i] = round(glob_r2i_recall[i] / 10, 3)
    final_r2i_med = np.average(glob_r2i_med)

    print("Mean i2r median", final_med)
    print("Mean r2i median", final_r2i_med)

    print("i2r Recall", glob_recall)
    print("r2i Recall", glob_r2i_recall)

    final_result = "i2r median: {} \n".format(final_med) + "i2r Recall: {} \n".format(glob_recall) + results
    final_result = "r2i median: {} \n".format(final_r2i_med) + "r2i Recall: {} \n".format(
        glob_r2i_recall) + final_result
    with open(result_path + '{}.txt'.format(N), 'w') as f:
        f.write(final_result)


def do_test():
    result_dir = opts.result_dir
   
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP)
    model.to(device) 

    model_path = opts.snapshots + '{}.pth.tar'.format(result_dir)
    print("=> loading checkpoint '{}'".format(model_path))
    if device.type == 'cpu':
        checkpoint = torch.load(model_path, encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(model_path, encoding='latin1')
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, checkpoint['epoch']))

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # preparing test loader
    test_loader = torch.utils.data.DataLoader(
        ImagerLoader(opts.img_path,
                     transforms.Compose([
                         transforms.Scale(256),  # rescale the image keeping the original aspect ratio
                         transforms.CenterCrop(224),  # we get only the center of that rescaled
                         transforms.ToTensor(),
                         normalize,
                     ]), data_path=opts.data_path, sem_reg=opts.semantic_reg, partition='test'),
        batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.workers, pin_memory=True)
    print('Test loader prepared.')

    # run test
    test(test_loader, model, result_dir)
    
    result_path = '../../results/{}/'.format(result_dir)
    print('1k test')
    improved_rank_emb(result_path, test_size=1000)

    print('10k test')
    improved_rank_emb(result_path, test_size=10000)


def test(test_loader, model, result_dir):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        with torch.no_grad():
            input_var = list()
            for j in range(len(input)):
                input_var.append(input[j].to(device))
            target_var = list()
            for j in range(len(target) - 2):  # we do not consider the last two objects of the list
                target_var.append(target[j].to(device))

            # compute output
            output = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4])
            img_id_fea = output[0]
            rec_id_fea = output[1]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i == 0:
                data0 = img_id_fea.data.cpu().numpy()
                data1 = rec_id_fea.data.cpu().numpy()
                data2 = target[-2]
                data3 = target[-1]
            else:
                data0 = np.concatenate((data0, img_id_fea.data.cpu().numpy()), axis=0)
                data1 = np.concatenate((data1, rec_id_fea.data.cpu().numpy()), axis=0)
                data2 = np.concatenate((data2, target[-2]), axis=0)
                data3 = np.concatenate((data3, target[-1]), axis=0)

    if not os.path.exists(opts.path_results + result_dir):
        os.mkdir(opts.path_results + result_dir)
    print(result_dir)
    with open(opts.path_results + result_dir + '/img_embeds.pkl', 'wb') as f:
        pickle.dump(data0, f)
    with open(opts.path_results + result_dir + '/rec_embeds.pkl', 'wb') as f:
        pickle.dump(data1, f)
    with open(opts.path_results + result_dir + '/img_ids.pkl', 'wb') as f:
        pickle.dump(data2, f)
    with open(opts.path_results + result_dir + '/rec_ids.pkl', 'wb') as f:
        pickle.dump(data3, f)


if __name__ == '__main__':
    result_dir = opts.result_dir
    print(result_dir)
    do_test()

    result_path = '../../results/{}/'.format(result_dir)

    print('1k test')
    improved_rank_emb(result_path, test_size=1000)

    print('10k test')
    improved_rank_emb(result_path, test_size=10000)
