# Installation of nsight: 
# wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_4/nsight-systems-2024.4.1_2024.4.1.61-1_amd64.deb
# apt install /workspace/nsight-systems-2024.4.1_2024.4.1.61-1_amd64.deb
UNIQUE_MACHINE_ID=15329  # i notice huge speed difference between machines on vast.ai even in the same configuration
GPU_NAME=4090
# nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x1_latent_1 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x1_latent_1.sh
# nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x1_latent_2 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x1_latent_2.sh
# nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x1_latent_3 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x1_latent_3.sh

# nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x2_latent_1 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x2_latent_1.sh
# nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x2_latent_2 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x2_latent_2.sh
# nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x2_latent_3 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x2_latent_3.sh

nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x3_latent_1 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x3_latent_1.sh
nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x3_latent_2 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x3_latent_2.sh
nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x3_latent_3 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x3_latent_3.sh

nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x6_latent_1 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x6_latent_1.sh
nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x6_latent_2 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x6_latent_2.sh
nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_dit_${GPU_NAME}x6_latent_3 bash /workspace/matrix/profiling_dit_settings/parallel_inference_${GPU_NAME}x6_latent_3.sh

# nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_vae_${GPU_NAME}x1 torchrun --nnodes 1 --nproc-per-node 1 debug_vae_infer_parallel.py
# nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_vae_${GPU_NAME}x2 torchrun --nnodes 1 --nproc-per-node 2 debug_vae_infer_parallel.py
# nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_vae_${GPU_NAME}x4 torchrun --nnodes 1 --nproc-per-node 4 debug_vae_infer_parallel.py
# nsys profile --trace=cuda,nvtx,osrt,openmp,mpi  --trace-fork-before-exec=true -o /workspace/matrix/nsight_result/${UNIQUE_MACHINE_ID}_vae_${GPU_NAME}x6 torchrun --nnodes 1 --nproc-per-node 6 debug_vae_infer_parallel.py