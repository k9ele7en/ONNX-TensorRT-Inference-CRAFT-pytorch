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
from pathlib import Path

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
import onnxruntime

from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='Triton inference pipeline for CRAFT Text Detection')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
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

def test_net(args, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # load onnx file
    data_path = Path('./weights')
    model_path = str(sorted(data_path.glob('*.onnx'))[0])
    ic(str(model_path))
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    img_resized = imgproc.normalizeMeanVariance(img_resized)
    img_resized = torch.from_numpy(img_resized).permute(2, 0, 1)    # [h, w, c] to [c, h, w]    

    img_resized = (img_resized.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    img_resized = img_resized.to(device)
    ort_session = onnxruntime.InferenceSession(model_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_resized)}
    ort_outs = ort_session.run(None, ort_inputs)
    y= ort_outs[0]
   
    # make score and link map
    score_text = y[0,:,:,0]
    score_link = y[0,:,:,1]

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':
    t = time.time()
    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(args, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)

        # save score text
        # filename, file_ext = os.path.splitext(os.path.basename(image_path))
        # mask_file = result_folder + "/res_" + filename + '_mask_triton.jpg'
        # cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder, method='onnx')

    print("elapsed time : {}s".format(time.time() - t))

# Example cmd:
# python infer_onnx.py

