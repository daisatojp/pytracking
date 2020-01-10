import os
from os.path import join, splitext, basename
import argparse
import cv2
from glob import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pytracking.evaluation import Tracker
from pycvlib.utility.pascalvoc import read_pascalvoc


def test(tracker_, src_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

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
    imgw = img[1]
    imgh = img[0]
    tracker_.initialize(img, {'init_bbox': init_bbox})
    frame_disp = img.copy()
    cv2.rectangle(
        frame_disp, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 5)
    cv2.imwrite(join(out_dir, 'frame{0:04d}.jpg'.format(0)), frame_disp)

    for i in range(1, len(img_paths)):
        img_path = img_paths[i]
        frame = cv2.imread(img_path)
        frame_disp = frame.copy()
        out = tracker_.track(frame)
        coord = [int(s) for s in out['samplecoord']]
        cv2.rectangle(
            frame_disp, (coord[1], coord[0]), (coord[3], coord[2]), (255, 0, 0), 5)
        state = [int(s) for s in out['target_bbox']]
        cv2.rectangle(
            frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
            (0, 255, 0), 5)
        for distractor in out['distractors']:
            state = [int(s) for s in distractor]
            cv2.rectangle(
                frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                (0, 0, 255), 5)
        cv2.imwrite(join(out_dir, 'frame{0:04d}.jpg'.format(i)), frame_disp)

        scoremap = out['scoremap']
        fig, ax = plt.subplots()
        im = ax.imshow(scoremap)
        for si in range(scoremap.shape[0]):
            for sj in range(scoremap.shape[1]):
                text = ax.text(sj, si, '{0:d}'.format(int(max(0.0, scoremap[si, sj]) * 10.0) % 10), ha="center", va="center", color="w")
        plt.savefig(join(out_dir, 'score{0:04d}.jpg'.format(i)))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--srcdir', type=str)
    parser.add_argument('--dstdir', type=str)
    parser.add_argument('--dirs', action='store_true')
    args = parser.parse_args()

    weights_path = args.weights
    src_dir = args.srcdir
    dst_dir = args.dstdir
    dirs = args.dirs

    tracker = Tracker('dimp', 'dimp50')
    params = tracker.get_parameters()
    params.tracker_name = 'dimp'
    params.param_name = 'dimp50'
    params.net.net_path = weights_path
    tracker_ = tracker.tracker_class(params)
    tracker_.initialize_features()

    if not dirs:
        out_dir = join(dst_dir, basename(src_dir))
        test(tracker_, src_dir, out_dir)
    else:
        src_dirs = glob(join(src_dir, '*'))
        for src_dir in src_dirs:
            if os.path.isdir(src_dir):
                out_dir = join(dst_dir, basename(src_dir))
                test(tracker_, src_dir, out_dir)


if __name__ == '__main__':
    main()
