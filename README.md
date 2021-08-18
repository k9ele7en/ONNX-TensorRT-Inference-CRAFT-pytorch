# Convert Pytorch pretrain -> TensoRT engine directly for CRAFT (Character-Region Awareness For Text detection)
- Convert CRAFT Text detection pretrain Pytorch model into TensorRT engine directly, without ONNX step between<br>
- CRAFT: (forked from https://github.com/clovaai/CRAFT-pytorch)
Official Pytorch implementation of CRAFT text detector | [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)
- Using torch2trt_dynamic from https://github.com/grimoire/torch2trt_dynamic (branch of https://github.com/NVIDIA-AI-IOT/torch2trt with dynamic shapes support)

### Overview
Implementation of inference pipeline using Tensor RT for CRAFT text detector.
Two modules included:
- Convert pretrain Pytorch -> ONNX -> TensorRT
- Inference using Tensor RT

Note: This repo is about converting steps to finally get Tensor RT engine, and inference on the engine. More related repo about Tensor RT inference, check out:
- Advance inference pipeline using NVIDIA Triton Server (https://github.com/k9ele7en/triton-tensorrt-CRAFT-pytorch)
- Convenient converter from Pytorch to Tensor RT directly, without ONNX bridge step (https://github.com/k9ele7en/torch2tensorRT-dynamic-CRAFT-pytorch).

### Author
k9ele7en. Give 1 star if you find some value in this repo. <br>
Thank you.

### License
[MIT License] A short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source.

## Updates
**7 Aug, 2021**: Init repo, converter run success. Run infer by ONNX success. Run infer by RT engine return wrong output.


## Getting started
### 1. Install dependencies
#### Requirements
```
$ pip install -r requirements.txt
```
#### Install ONNX, TensorRT
Check details at [./README_Env.md](./README_Env.md)

### 2. Download the trained models
 
 *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
 | :--- | :--- | :--- | :--- | :--- |
General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)
LinkRefiner | CTW1500 | - | Used with the General Model | [Click](https://drive.google.com/open?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO)

### 3. Start converting Pytorch->TensorRT
#### Use single .sh script to run converter, ready to infer after complete successfully
```
sh prepare.sh
```

#### Seperate single converters
```
$ cd converters
$ python pth2onnx.py
$ python onnx2trt.py
```

### 4. Start infer on Tensor RT engine
```
$ python infer_trt.py
```

### 5. Infer on ONNX format
```
$ python infer_onnx.py
```
