import os
from os.path import join, splitext, basename

import sys
import argparse
import cv2
from glob import glob
from pycvlib.utility.pascalvoc import read_pascalvoc

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker
from ltr.admin.loading import torch_load_legacy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--srcdir', type=str)
    parser.add_argument('--dstdir', type=str)
    args = parser.parse_args()

    weights_path = args.weights
    src_dir = args.srcdir
    dst_dir = args.dstdir

    out_dir = join(dst_dir, basename(src_dir))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    tracker = Tracker('dimp', 'dimp50')

    params = tracker.get_parameters()
    params.tracker_name = 'dimp'
    params.param_name = 'dimp50'
    params.net.net_path = weights_path

    tracker_ = tracker.tracker_class(params)
    tracker_.initialize_features()

    img_paths = glob(join(src_dir, '*.jpg'))
    img_path0 = img_paths[0]
    xml_path0 = splitext(img_path0)[0] + '.xml'

    annot = read_pascalvoc(xml_path0)
    xmin = annot['object'][0]['bndbox']['xmin']
    ymin = annot['object'][0]['bndbox']['ymin']
    xmax = annot['object'][0]['bndbox']['xmax']
    ymax = annot['object'][0]['bndbox']['ymax']
    init_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

    img = cv2.imread(img_path0)
    tracker_.initialize(img, {'init_bbox': init_bbox})
    for i in range(1, len(img_paths)):
        img_path = img_paths[i]
        frame = cv2.imread(img_path)
        frame_disp = frame.copy()
        out = tracker_.track(frame)
        state = [int(s) for s in out['target_bbox']]
        cv2.rectangle(
            frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
            (0, 255, 0), 5)
        cv2.imwrite(join(out_dir, 'frame{0:04d}.jpg'.format(i)), frame_disp)


if __name__ == '__main__':
    main()
