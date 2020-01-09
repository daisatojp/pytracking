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


class TNPlaneVid(BaseDataset):
    """ TNPlainVid dataset
    """

    def __init__(self, root=None, image_loader=default_image_loader):
        """
        """
        super().__init__(root, image_loader)

        # all folders inside the root
        self.sequence_list = glob(join(root, '*'))
        self.sequence_frame_list = [glob(join(seq_path, '*.jpg')) for seq_path in self.sequence_list]

    def get_name(self):
        return 'tnplainvid'

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
        #     cv2.imwrite('tnplanevid{0}.jpg'.format(i), frame)

        return frame_list, anno_frames, None
