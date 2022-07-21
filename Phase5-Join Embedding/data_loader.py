from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import sys
import pickle
import numpy as np
import lmdb
import torch


def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        print('img missing', file=sys.stderr)
        print(path)
        return Image.new('RGB', (224, 224), 'white')


class ImagerLoader(data.Dataset):
    def __init__(self, img_path, transform=None, target_transform=None,
                 loader=default_loader, square=False, data_path=None, partition=None, sem_reg=None):

        if data_path == None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition

        self.env = lmdb.open(os.path.join(data_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)

        with open(os.path.join(data_path, partition + '_keys.pkl'), 'rb') as f:
            self.ids = pickle.load(f)

        self.square = square
        self.imgPath = img_path
        self.mismtch = 0.8
        self.maxInst = 20

        with open('../../data/id_tfidf_tri_vec_all_v4.pkl', 'rb') as f:
            self.id_vec_dict = pickle.load(f)

        with open('../../data/id2img_101_cls_vec.pkl', 'rb') as f:
            self.id_imgembs = pickle.load(f)

        with open('../../data/id2class_1005.pkl', 'rb') as f:
            self.id2class_id = pickle.load(f)

        if sem_reg is not None:
            self.semantic_reg = sem_reg
        else:
            self.semantic_reg = False

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        recipId = self.ids[index]
        # we force 80 percent of them to be a mismatch
        target = 1

        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode('latin1'))
        sample = pickle.loads(serialized_sample, encoding='latin1')
        imgs = sample['imgs']

        # image
        if self.partition == 'train':
                # We do only use the first five images per recipe during training
            imgIdx = np.random.choice(range(min(5, len(imgs))))
        else:
            imgIdx = 0

        loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
        loader_path = os.path.join(*loader_path)
        path = os.path.join(self.imgPath, self.partition, loader_path, imgs[imgIdx]['id'])
        # path = os.path.join(self.imgPath, loader_path, imgs[imgIdx]['id'])


        # recipe
        rec_vec = self.id_vec_dict[recipId]
        rec_vec = torch.FloatTensor(rec_vec)

        instrs = sample['intrs']
        itr_ln = len(instrs)
        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)

        # load image
        img = self.loader(path)
        img2 = self.id_imgembs[recipId][imgIdx]
        img2 = torch.FloatTensor(img2)
        
        if self.square:
            img = img.resize(self.square)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # rec_class = sample['classes'] - 1
        rec_id = self.ids[index]
        rec_class = self.id2class_id[rec_id]

        # img_class = sample['classes'] - 1
        img_id = self.ids[index]    
        img_class = self.id2class_id[img_id]
            

        # output
        if self.partition == 'train':
            if self.semantic_reg:
                return [img, img2, rec_vec, instrs, itr_ln], [target, img_class, rec_class]
            else:
                return [img, img2, rec_vec, instrs, itr_ln], [target]
        else:
            if self.semantic_reg:
                return [img, img2, rec_vec, instrs, itr_ln], [target, img_class, rec_class, img_id, rec_id]
            else:
                return [img, img2, rec_vec, instrs, itr_ln], [target, img_id, rec_id]

    def __len__(self):
        return len(self.ids)