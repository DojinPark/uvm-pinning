# UVM-Pinning

> A driver modification to mitigate Unified Memory thrashing.

Unlike traditional memory copy from host memory using [cudaMemcpy()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8) which requires developers to manage GPU memory by manual, [cudaMallocManaged()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b) in *Unified Memory* enables demand-paging.
However, [*Unified Memory*](https://www.nextplatform.com/2019/01/24/unified-memory-the-final-piece-of-the-gpu-programming-puzzle/) undergoes severe slow down, because LRU page replacement policy [defined within Nvidia Driver](https://www.nvidia.com/en-us/geforce/forums/discover/272966/source-code-for-unified-memory-driver-nvidia-uvm-ko-/) causes repeated page fault on every access, taking exponential runtime compared to ideal situation.
It is found on this study, that this completely inefficient page replacement occurs on non-streaming type applications and defined it as *cyclic thrashing*.
*UVM-Pinning* detects cyclic thrashing under *Unified Memory*, and pins pages on GPU memory for later reuse, reducing runtime by 55% in best case experiment.

Refer to [Page Reuse in Cyclic Thrashing of GPU Under Oversubscription: Work-in-Progress](//uvm-pinningPark_CASES2020.pdf) for more details.

---


## Prerequisites
- Python 3
- CUDA Toolkit & NVIDIA Driver (must be paired versions)
```sh
# go to https://developer.nvidia.com/cuda-downloads
# download linux runfile installer of a version of your preference.
wget https://developer.download.nvidia.com/compute/cuda/{xx.xx.xx}/local_installers/cuda_{xx.xx.xx}_linux.run
sudo sh cuda_{xx.xx.xx}_linux.run --extract=toolkit
sudo sh toolkit/
```

## Installation

Linux:

```sh
./
```

## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

_For more examples and usage, please refer to the [Wiki][wiki]._


## Meta

DOjin Park – [@github](https://github.com/DojinPark) – djpark@arcs.skku.edu

Distributed under the GNU GPLv2 license. See ``LICENSE`` for more information.
