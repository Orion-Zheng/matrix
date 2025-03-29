import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def get_vae_parallel_group():
    # All ranks are in the same group here, for simplicity
    return dist.group.WORLD


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_test(rank, world_size):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    first_vae_worker_rank = 1  # Rank 1 is the source
    vae_group = get_vae_parallel_group()

    if rank == first_vae_worker_rank:
        # Simulate latents from ViT->VAE
        latents = torch.randn(2, 3, 5, 16, 16, dtype=dtype, device=device)  # dummy data

        # Send shape
        shape_len = torch.tensor([len(latents.shape)], dtype=torch.int, device=device)
        shape_tensor = torch.tensor(latents.shape, dtype=torch.int, device=device)

        print(f"[Rank {rank}] shape_len: {shape_len}, shape_tensor: {shape_tensor}, latents.shape: {latents.shape}")

        dist.broadcast(shape_len, src=first_vae_worker_rank, group=vae_group)
        dist.broadcast(shape_tensor, src=first_vae_worker_rank, group=vae_group)
        dist.broadcast(latents, src=first_vae_worker_rank, group=vae_group)

    else:
        time.sleep(5)
        print(f"[Rank {rank}] Waiting for latents...")
        # Receive shape
        shape_len = torch.zeros(1, dtype=torch.int, device=device)
        dist.broadcast(shape_len, src=first_vae_worker_rank, group=vae_group)

        shape_tensor = torch.zeros(shape_len[0], dtype=torch.int, device=device)
        dist.broadcast(shape_tensor, src=first_vae_worker_rank, group=vae_group)

        latents = torch.zeros(tuple(shape_tensor.tolist()), dtype=dtype, device=device)
        dist.broadcast(latents, src=first_vae_worker_rank, group=vae_group)

        print(f"[Rank {rank}] Received latents with shape: {latents.shape}")

    cleanup()


if __name__ == "__main__":
    # Usage: CUDA_VISIBLE_DEVICES=0,1 python test_mutli_process_sync.py
    world_size = 2  # Number of GPUs
    mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)