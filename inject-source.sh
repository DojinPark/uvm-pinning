#!/bin/bash
export UVM_SRC_PATH=
sed -i '/#if defined(CLOCK_MONOTONIC_RAW)/e cat ./src/uvm_linux.before.h' $DRIVER_PATH/kernel/nvidia-uvm/uvm_linux.h

sed -i '/} uvm_pmm_gpu_t;/e cat ./src/uvm8_pmm_gpu.before.h' $DRIVER_PATH/kernel/nvidia-uvm/uvm8_gpu.h

sed -i '/static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)/e cat ./src/uvm8_pmm_gpu.before.h' $DRIVER_PATH/kernel/nvidia-uvm/uvm8_gpu.h
sed -i '/static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)/r ./src/uvm8_pmm_gpu.after.h' $DRIVER_PATH/kernel/nvidia-uvm/uvm8_gpu.h

STRING=$(<./src/uvm8_va_block.after.h)
sed -z -i "s/struct uvm_va_block_struct\n{/&\n$STRING/" $DRIVER_PATH/kernel/nvidia-uvm/uvm8_va_block.h
