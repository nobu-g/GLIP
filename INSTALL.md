# Installation

1. Go to a host where the CUDA Toolkit 11.x (11.7) is installed.

```shell
ssh agni
```

1. Install Python dependencies.

```shell
poetry install
```

1. Build `maskrcnn_benchmark` from source for CUDA 11.7.

```shell
export CUDA_HOME=/usr/local/cuda-11.7
export CUDA_PATH=/usr/local/cuda-11.7
export PATH=...:$CUDA_HOME/bin:...
python setup.py build develop
```
