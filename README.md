# UVM-Pinning 🌼

> A driver modification to mitigate Unified Memory thrashing.

Unlike traditional memory copy from host memory using [cudaMemcpy()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8) which requires developers to manage GPU memory by manual, [cudaMallocManaged()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b) in *Unified Memory* enables demand-paging.

However, [*Unified Memory*](https://www.nextplatform.com/2019/01/24/unified-memory-the-final-piece-of-the-gpu-programming-puzzle/) undergoes severe slow down, because LRU page replacement policy [defined within Nvidia Driver](https://www.nvidia.com/en-us/geforce/forums/discover/272966/source-code-for-unified-memory-driver-nvidia-uvm-ko-/) causes repeated page fault on every access, taking exponential runtime compared to ideal situation.

It is found on this study, that this completely inefficient page replacement occurs on non-streaming type applications and defined it as *cyclic thrashing*.

*UVM-Pinning* detects *cyclic thrashing* under *Unified Memory*, pins pages on GPU memory for later reuse and reduces runtime by 55% in best-case experiment.

Refer to [Page Reuse in Cyclic Thrashing of GPU Under Oversubscription: Work-in-Progress](DojinPark_CASES2020.pdf) for more details.

<p float=left align=center>
  <figure>
    <img src="/logs/paper/complete/ra1500-pinning25-zoom-phys.png" width=40% />
    <figcaption>Vertical violet line indicates cyclic thrashing detection.</figcaption>
    <img src="/logs/paper/complete/fdtd120-phys.png" width=40% />
    <figcaption>Cavity implies pinnned pages in GPU memory.</figcaption>
  </figure>
</p>



---


## Prerequisites
- Linux OS
- Python 3
- CUDA Toolkit & NVIDIA Driver (must be paired versions)

Download linux runfile installer at https://developer.nvidia.com/cuda-downloads

Than extract driver source by following commands below.

(Edit filenames for your desired version of CUDA Toolkit.)

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
sh cuda_11.2.2_460.32.03_linux.run --extract=~/toolkit
~/toolkit/NVIDIA-Linux-x86_64-460.32.03.run -x    # A directory ~/NVIDIA-Linux-x86_64-460.32.03/ should be created.
```

## Driver Installation

```bash
git clone https://github.com/DojinPark/uvm-pinning
```

Initial Setup. You MUST edit *init.sh* to your preferred workspace path.

It is recommended to add this to your terminal startup script (i.e. ~/.bashrc)
```bash
source uvm-pinning/init.sh
```

Than inject uvm-pinning source to the extracted driver source.
```bash
bash uvm-pinning/inject-source.sh $DRIVER_PATH
```

Finally, under virtual terminal (<kbd>Ctrl</kbd> + <kbd>Alt</kbd> + <kbd>F2</kbd> for ubuntu) install the modified driver.
```bash
sudo bash $UVM_PINNING_PATH/install.sh
```


## Usage

Build benchmarks before running.
```bash
make -f $UVM_PINNING_PATH/benchmarks/Managed/makefile
```

- To run all benchmarks:
```bash
bash $UVM_PINNING_PATH/Managed/bin/run-all.sh
```

- To run all benchmarks with logs:
```bash
bash $UVM_PINNING_PATH/Managed/bin/run-all-logs.sh
```

- To try individual benchmark, for example, to run *addvector* with 1.2GB data size under 1GB GPU memory capacity,
```bash
./addvector 1228.8 1024
```

- Than to save obtain log text and plot page faults from individual benchmark:
```bash
sudo uvm-pinning-plot my_log
```

## Author

Dojin Park – [@github](https://github.com/DojinPark) – djpark@arcs.skku.edu

Distributed under the GNU GPLv2 license.
