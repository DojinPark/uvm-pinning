# Edit DRIVER_PATH to your extracted driver path
export DRIVER_PATH="./NVIDIA-Linux-x86_64-450.51.06"
export CUDA_PATH="/usr/local/cuda"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
export CPATH="$CUDA_PATH/include:$CPATH"
export CPP_INCLUDE_PATH="$CPP_INCLUDE_PATH:/usr/local/include/eigen3" # Eigen

# Logging utils
export UVM_PINNING_PREFIX='uvm-pinning'
function uvm-pinning-log {
	echo > /dev/null | sudo tee /var/log/kern.log # Cleanse other kernel logs
	cp /var/log/kern.log $1
	sed -i "/.*nvidia-uvm:.*/d" $1
	sed -i "s/^.*\($UVM_PINNING_PREFIX:\)//g" $1
	plot $1
}


