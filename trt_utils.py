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
