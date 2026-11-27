# Source this to run the shell's later CUDA processes under CUDA MPS, so their
# kernels overlap on the GPU instead of taking turns. A fatal GPU error can then
# stop all of them. The daemon stops when the shell exits.

# One daemon, in directories no other job shares. Apptainer passes these
# variables through and binds /tmp.
mps_dir=$(mktemp -d /tmp/mps-XXXXXX)
export CUDA_MPS_PIPE_DIRECTORY="$mps_dir/pipe" CUDA_MPS_LOG_DIRECTORY="$mps_dir/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control; rm -rf "$mps_dir"' EXIT
