# Distributed Data Parallel (DDP) tutorial
# from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

import os
import sys
import tempfile
import datetime
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# logger from FACEBOOK_SWAV...
# from facebook_swav.src.logger import create_logger

parser = argparse.ArgumentParser(description="Evaluate models: Fine-tuning with 1% or 10% labels on ImageNet")

parser.add_argument("--dist_url", default="env://", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# For TcpStore, same way as on Linux.

def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # # initialize the process group (Linux)
    # dist.init_process_group(
    #     "gloo",
    #     rank=rank,
    #     world_size=world_size)

    # initialize the process group (Windows)
    print("init process group (start) - rank {} method {}".format(rank, args.dist_url))
    dist.init_process_group(
        "gloo",
        rank=rank,
        init_method=args.dist_url,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    print("init process group (end)".format(rank))

def cleanup():
    dist.destroy_process_group()

# Combining DDP with Model Parallelism

# DDP also works with multi-GPU models. DDP wrapping multi-GPU models is especially helpful when training large models with a huge amount of data.

class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)

# When passing a multi-GPU model to DDP, device_ids and output_device must NOT be set.
# Input and output data will be placed in proper devices by either the application or 
# the model forward() method.

def demo_model_parallel(rank, world_size, args):

    setup(rank, world_size, args)

    # create a logger
    # logger = create_logger(os.path.join(args.dump_path, "log"), rank=rank)

    # logger.info("============ Initialized logger ============")
    # logger.info(
    #     "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    # )

    # setup mp_model and devices for this process
    dev0 = (rank * 2) % world_size
    dev1 = (rank * 2 + 1) % world_size
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

def run_demo(args, demo_fn):

    # when using torch distributed (ddp) initialization via file
    # need to delete the sync file before running init_process_group()
    # see https://pytorch.org/docs/stable/distributed.html
    syncfile = args.dist_url.split('file:\\')[1]
    if os.path.exists(syncfile):
        os.remove(syncfile)

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    print("----- logs at {} -----".format(args.dump_path))
    print("spawning {} processes".format(world_size))

    mp.spawn(fn=demo_fn,
             args=(world_size, args,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    args = parser.parse_args()
    # run_demo(demo_basic, world_size)
    # run_demo(demo_checkpoint, world_size)
    run_demo(args, demo_model_parallel)

