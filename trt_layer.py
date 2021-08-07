import os
from pathlib import Path

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

class RTLayer():
    """
    
    """

    def __init__(self, config=None, model_path=None, data_path='./weights',
                 engine_path=None, cuda_ctx=None, input_shape=None):
        super().__init__()
        data_path = Path(data_path)
        model_path = sorted(data_path.glob('*.engine'))

        self.engine_path=model_path[0]

        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self._load_plugins()
        self.engine = self._load_engine()
        self.input_shape = input_shape

    def __call__(self, image):

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()

        # feed to engine and process output
        height, width = img_resized.shape[2:4]
        self.input_shape = (height,width)
        img_resized = img_resized.cpu().detach().numpy()
        
        print(6,img_resized.shape)
        
        segment_inputs, segment_outputs, segment_bindings = self._allocate_buffers()
        
        stream = cuda.Stream()    

        with self.engine.create_execution_context() as context:
            context.active_optimization_profile = 0
            origin_inputshape=context.get_binding_shape(0)
            
            if (origin_inputshape[-1]==-1):
                origin_inputshape[-2],origin_inputshape[-1]=(self.input_shape)
                context.set_binding_shape(0,(origin_inputshape))
            
            input_img_array = np.array([img_resized] * self.engine.max_batch_size)
            img = torch.from_numpy(input_img_array).float().numpy()
            segment_inputs[0].host = img
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in segment_inputs]#Copy from the Python buffer src to the device pointer dest (an int or a DeviceAllocation) asynchronously,
            stream.synchronize()#Wait for all activity on this stream to cease, then return.
        
            context.execute_async(bindings=segment_bindings, stream_handle=stream.handle)#Asynchronously execute inference on a batch. 
            stream.synchronize()
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in segment_outputs]#Copy from the device pointer src (an int or a DeviceAllocation) to the Python buffer dest asynchronously
            stream.synchronize()
            # print(1,segment_outputs[0].host.shape)
            y_out = segment_outputs[0].host
        
        y1 =  y_out[0:np.array(context.get_binding_shape(2)).prod()].reshape(context.get_binding_shape(2))
        print('head: ',y_out[0:np.array(context.get_binding_shape(2)).prod()])
        print('tail: ',y_out[np.array(context.get_binding_shape(2)).prod()-2:])
        # y1 =  y_out[0:np.array(context.get_binding_shape(2)).prod()].reshape(context.get_binding_shape(2))
        
        y = torch.from_numpy(y1)
        print(5,'y2: ',y.shape)
        print('value: ',y)
        return y
    def _load_plugins(self):
        if trt.__version__[0] < '7':
            ctypes.CDLL("./libflattenconcat.so")
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        assert os.path.exists(self.engine_path)
        print("Reading engine from file {}".format(self.engine_path))
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        class HostDeviceMem(object):
            def __init__(self, host_mem, device_mem):
                self.host = host_mem
                self.device = device_mem

            def __str__(self):
                return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

            def __repr__(self):
                return self.__str__()
        for binding in self.engine:
            
            dims = self.engine.get_binding_shape(binding)
            # print(dims)
            if dims[-1] == -1:
                assert(self.input_shape is not None)
                dims[-2],dims[-1] = self.input_shape
            size = trt.volume(dims) * self.engine.max_batch_size#The maximum batch size which can be used for inference.
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):#Determine whether a binding is an input binding.
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings

    # def __del__(self):
    #     """Free CUDA memories and context."""
    #     del self.cuda_outputs
    #     del self.cuda_inputs
    #     del self.stream
