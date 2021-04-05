# UVM-Pinning 🌼

> A driver modification to mitigate Unified Memory thrashing.

Unlike traditional memory copy from host memory using [cudaMemcpy()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8) which requires developers to manage GPU memory by manual, [cudaMallocManaged()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b) in *Unified Memory* enables demand-paging.

However, [*Unified Memory*](https://www.nextplatform.com/2019/01/24/unified-memory-the-final-piece-of-the-gpu-programming-puzzle/) undergoes severe slow down, because LRU page replacement policy [defined within Nvidia Driver](https://www.nvidia.com/en-us/geforce/forums/discover/272966/source-code-for-unified-memory-driver-nvidia-uvm-ko-/) causes repeated page fault on every access, taking exponential runtime compared to ideal situation.

It is found on this study, that this completely inefficient page replacement occurs on non-streaming type applications and defined it as *cyclic thrashing*.

*UVM-Pinning* detects cyclic thrashing under *Unified Memory*, pins pages on GPU memory for later reuse and reduces runtime by 55% in best case experiment.

Refer to [Page Reuse in Cyclic Thrashing of GPU Under Oversubscription: Work-in-Progress](DojinPark_CASES2020.pdf) for more details.

---


## Prerequisites
- Linux OS
- Python 3
- CUDA Toolkit & NVIDIA Driver (must be paired versions)

Download linux runfile installer of a version of your preference

at https://developer.nvidia.com/cuda-downloads

Than extract driver source by following commands below

```sh
cd ~
wget https://developer.download.nvidia.com/compute/cuda/{XX.XX.XX}/local_installers/cuda_{XX.XX.XX}_linux.run
sudo sh cuda_{XX.XX.XX}_linux.run --extract=~/toolkit
sudo ~/toolkit/NVIDIA-Linux-{XX.XX.XX.XX}.run -x # A folder named ~/NVIDIA-Linux-{XX.XX.XX.XX} should be created.
```

## Driver Installation
Initial Setup
```sh
git clone https://github.com/DojinPark/uvm-pinning
sudo sh uvm-pinning/init.sh
sh uvm-pinning/copy-source.sh ~/NVIDIA-Linux-{XX.XX.XX.XX}/
```

Now, under virtual terminal (<kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>F2</kbd> for ubuntu)

```sh

```
## Usage

```sh
```


## Meta

Dojin Park – [@github](https://github.com/DojinPark) – djpark@arcs.skku.edu

Distributed under the GNU GPLv2 license. See ``LICENSE`` for more information.
