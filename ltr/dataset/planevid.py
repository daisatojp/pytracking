import os
import os.path
from os.path import join
import numpy as np
import torch
import cv2
import csv
import pandas
import random
from glob import glob
from collections import OrderedDict
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
from ltr.admin.environment import env_settings


class PlaneVid(BaseDataset):
    """ PlainVid dataset
    """

    def __init__(self, root=None, image_loader=default_image_loader):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
        """
        super().__init__(root, image_loader)

        # all folders inside the root
        self.sequence_list = glob(join(root, '*'))
        self.sequence_frame_list = [glob(join(seq_path, '*.jpg')) for seq_path in self.sequence_list]

    def get_name(self):
        return 'plainvid'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible,  visible_ratio

    def get_sequence_info(self, seq_id):
        seq_path = self.sequence_list[seq_id]
        bbox = torch.from_numpy(np.loadtxt(join(seq_path, 'bb.txt'), dtype=np.float32))
        valid = torch.from_numpy(np.ones(shape=(bbox.shape[0],), dtype=np.bool))
        visible = torch.from_numpy(np.ones(shape=(bbox.shape[0],), dtype=np.uint8))
        visible_ratio = torch.from_numpy(np.ones(shape=(bbox.shape[0],), dtype=np.float32))
        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_id, frame_id):
        return self.sequence_frame_list[seq_id][frame_id]

    def _get_frame(self, seq_id, frame_id):
        return self.image_loader(self._get_frame_path(seq_id, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = [self._get_frame(seq_id, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        # debug
        # for i in range(3):
        #     frame = frame_list[i].copy()
        #     bbox = anno_frames['bbox'][i].numpy()
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 0, 0), 2)
        #     cv2.imwrite('planevid{0}.jpg'.format(i), frame)

        return frame_list, anno_frames, None
