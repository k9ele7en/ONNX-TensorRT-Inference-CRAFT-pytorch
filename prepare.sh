# Make sure you downloaded one of pretrain model and locate at /weights folder, link: https://github.com/clovaai/CRAFT-pytorch#test-instruction-using-pretrained-model
export failed=0 # To check if cmd run completed
export CUDA_VISIBLE_DEVICES=0 # Set specific GPU for TensorRT engine optimized for, start from 0

# I. Convert pth model into (Torch JIT, ONNX, TensorRT)
cd converters/
python pth2onnx.py || export failed=1
python onnx2trt.py || export failed=1
cd ..
# Convert ONNX to TensorRT can be done by trtexec command from TensorRT:
# /usr/src/tensorrt/bin/trtexec --onnx=./model_repository/detec_onnx/1/detec_onnx.onnx --explicitBatch --workspace=5000 --minShapes=input:1x3x256x256 --optShapes=input:1x3x700x700 --maxShapes=input:1x3x1200x1200 --buildOnly --saveEngine=./model_repository/detec_trt/1/detec_trt.plan


if [ ${failed} -ne 0 ]; then
        echo "Prepare Model Repo failed, check error on the terminal history above..."
      else
        echo "Convert source model into target formats and place in ./weights successfully."
        echo "Ready to infer."
      fi
