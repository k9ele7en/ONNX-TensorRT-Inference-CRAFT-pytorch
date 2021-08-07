# About
This is an introduction and basic guideline to setup environment for ONNX, TensorRT

## Open Neural Network Exchange (ONNX)
Open Neural Network Exchange (ONNX) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. ONNX provides an open source format for AI models, both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types. Currently we focus on the capabilities needed for inferencing (scoring).

ONNX is widely supported and can be found in many frameworks, tools, and hardware. Enabling interoperability between different frameworks and streamlining the path from research to production helps increase the speed of innovation in the AI community. We invite the community to join us and further evolve ONNX.

## TensorRT
TensorRT is an SDK for optimizing trained deep learning models to enable high-performance inference. TensorRT contains a deep learning inference optimizer for trained deep learning models, and a runtime for execution.
After you have trained your deep learning model in a framework of your choice, TensorRT enables you to run it with higher throughput and lower latency.

## PyCUDA
PyCUDA lets you access Nvidia's CUDA parallel computation API from Python.
For more information, check at: https://documen.tician.de/pycuda/

## I. Setup environment and tools
First, update your PIP:
```
$ python3 -m pip install --upgrade setuptools pip 
```
1. ONNX: install ONNX packages by pip and conda packages
```
$ python3 -m pip install nvidia-pyindex
$ conda install -c conda-forge onnx
$ pip install onnx_graphsurgeon 
```
Note: If convert pth to onnx get error (libstdc++.so.6: version `GLIBCXX_3.4.22' not found), fix by run below commands:
```
$ sudo add-apt-repository ppa:ubuntu-toolchain-r/test 
$ sudo apt-get update 
$ sudo apt-get install gcc-4.9 
$ sudo apt-get install --only-upgrade libstdc++6 
```
2. TensorRT (for detail install instruction, check at: https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#install)
- Login and download local repo file that matches the Ubuntu version and CPU architecture that you are using. from https://developer.nvidia.com/tensorrt
- Install downloaded deb package as guide (https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian): <br>
    - Install TensorRT from the Debian local repo package. Replace ubuntuxx04, cudax.x, trt8.x.x.x-ea and yyyymmdd with your specific OS version, CUDA version, TensorRT version and package date.
    ```
    $ os="ubuntuxx04"
    $ tag="cudax.x-trt8.x.x.x-ea-yyyymmdd"
    $ sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
    $ sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub

    $ sudo apt-get update
    $ sudo apt-get install tensorrt
    ```
    - Note this error when running above cmd, for ex. install TensorRT v7.2.3.4:
    ```
    $ sudo apt-get install tensorrt
    The following packages have unmet dependencies:
    tensorrt : Depends: libnvinfer-dev (= 7.2.3-1+cuda11.1) but 8.0.0-1+cuda11.3 is to be installed
                Depends: libnvinfer-plugin-dev (= 7.2.3-1+cuda11.1) but 8.0.0-1+cuda11.3 is to be installed
                Depends: libnvparsers-dev (= 7.2.3-1+cuda11.1) but 8.0.0-1+cuda11.3 is to be installed
                Depends: libnvonnxparsers-dev (= 7.2.3-1+cuda11.1) but 8.0.0-1+cuda11.3 is to be installed
                Depends: libnvinfer-samples (= 7.2.3-1+cuda11.1) but it is not going to be installed
    E: Unable to correct problems, you have held broken packages.
    ```
    - *Reason: APT-GET choose wrong version of dependencies to install as required by target TensorRT version. Run the followings cmd to solve: sudo apt-get -y install <dependency_name>=<target_version>...*
    ```
    $ sudo apt-get -y install libnvinfer-dev=7.2.3-1+cuda11.1
    $ sudo apt-get -y install libnvinfer-plugin-dev=7.2.3-1+cuda11.1
    $ sudo apt-get -y install libnvparsers-dev=7.2.3-1+cuda11.1
    $ sudo apt-get -y install libnvonnxparsers-dev=7.2.3-1+cuda11.1
    $ sudo apt-get -y install libnvinfer-samples=7.2.3-1+cuda11.1
    $ sudo apt-get -y install libnvinfer-plugin-dev=7.2.3-1+cuda11.1
    ...
    Now try again:
    $ sudo apt-get -y install tensorrt
    ```
    - If using Python 3.x:
    ```
    $ sudo apt-get install python3-libnvinfer-dev
    ```
    The following additional packages will be installed:
    python3-libnvinfer
    If you would like to run the samples that require ONNX graphsurgeon or use the Python module for your own project, run:
    ```
    $ sudo apt-get install onnx-graphsurgeon
    ```
    Verify the installation.
    ```
    $ dpkg -l | grep TensorRT
    ```
- Install pip packages (https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pip)
    - If your pip and setuptools Python modules are not up-to-date, then use the following command to upgrade these Python modules. If these Python modules are out-of-date then the commands which follow later in this section may fail.
    ```
    $ python3 -m pip install --upgrade setuptools pip
    ```
    You should now be able to install the nvidia-pyindex module.
    ```
    $ python3 -m pip install nvidia-pyindex
    ```
    - Install the TensorRT Python wheel.
    ```
    $ python3 -m pip install --upgrade nvidia-tensorrt
    ```
    The above pip command will pull in all the required CUDA libraries and cuDNN in Python wheel format because they are dependencies of the TensorRT Python wheel. Also, it will upgrade nvidia-tensorrt to the latest version if you had a previous version installed.

    - To verify that your installation is working, use the following Python commands to:
    ```
    python3
    >>> import tensorrt
    >>> print(tensorrt.__version__)
    >>> assert tensorrt.Builder(tensorrt.Logger())
    ```

3. PyCUDA (for details, check at: https://wiki.tiker.net/PyCuda/Installation/Linux/#step-1-download-and-unpack-pycuda)
-  Step 1: Download source of pip package tar.gz and unpack PyCUDA from https://pypi.org/project/pycuda/#files
```
$ tar xfz pycuda-VERSION.tar.gz
```
- Step 2: Install Numpy
PyCUDA is designed to work in conjunction with numpy, Python's array package. Here's an easy way to install it, if you do not have it already:
```
$ cd pycuda-VERSION
$ su -c "python distribute_setup.py" # this will install distribute
$ su -c "easy_install numpy" # this will install numpy using distribute
```
- Step 3: Build PyCUDA
Next, just type:
```
Install make if needed:
$ sudo apt-get install -y make

Start building:
$ cd pycuda-VERSION # if you're not there already
$ python configure.py --cuda-root=/where/ever/you/installed/cuda
$ su -c "make install"
```
- Step 4: Test PyCUDA
If you'd like to be extra-careful, you can run PyCUDA's unit tests:
```
$ cd pycuda-VERSION/test
$ python test_driver.py
```
