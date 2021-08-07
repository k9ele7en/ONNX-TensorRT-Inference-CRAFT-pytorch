#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________

This file contains main inference pipeline to Triton
_____________________________________________________________________________
"""
from icecream import ic
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from collections import OrderedDict

from trt_layer import RTLayer

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='TensorRT inference pipeline for CRAFT Text Detection')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=1100, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='images/', type=str, help='folder path to input images')

args = parser.parse_args()

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(args, image, text_threshold, link_threshold, low_text, poly):
    layer = RTLayer()
    boxes, polys, ret_score_text = layer(args, image, text_threshold, link_threshold, low_text, poly)
    return boxes, polys, ret_score_text


if __name__ == '__main__':
    t = time.time()
    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(args, image, args.text_threshold, args.link_threshold, args.low_text, args.poly)

        # save score text
        # filename, file_ext = os.path.splitext(os.path.basename(image_path))
        # mask_file = result_folder + "/res_" + filename + '_mask_triton.jpg'
        # cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder, method='rt')

    print("elapsed time : {}s".format(time.time() - t))

# Example cmd:
# python infer_trt.py --test_folder='./images'