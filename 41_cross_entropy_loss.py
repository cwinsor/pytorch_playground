# DistributedDataParallel tutorial
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

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader

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

def setup(args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # # initialize the process group (Linux)
    # dist.init_process_group(
    #     "gloo",
    #     rank=rank,
    #     world_size=world_size)

    # initialize the process group (Windows)
    print("init process group (start) - rank {} world_size {} method {}".format(
        args.rank, args.world_size, args.dist_url))
    dist.init_process_group(
        "gloo",
        rank=args.rank,
        init_method=args.dist_url,
        world_size=args.world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    print("init process group (end)".format(args.rank))

def cleanup():
    dist.destroy_process_group()

# Combining DDP with Model Parallelism

# DDP also works with multi-GPU models. DDP wrapping multi-GPU models is especially helpful when training large models with a huge amount of data.

class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1, in_size, out_size):
        super(ToyMpModel, self).__init__()
        # self.dev0 = dev0
        # self.dev1 = dev1
        self.net1 = torch.nn.Linear(in_size, 10)
        # self.net1 = self.net1.to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, out_size)
        # self.net2 = self.net2.to(dev1)

    def forward(self, x):
        # x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        # x = x.to(self.dev1)
        return self.net2(x)

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
        self.labels = torch.randn(length)

    def __getitem__(self, index):
        return (self.data[index], self.labels[index])

    def __len__(self):
        return self.len


# When passing a multi-GPU model to DDP, device_ids and output_device must NOT be set.
# Input and output data will be placed in proper devices by either the application or 
# the model forward() method.

def test_01(rank, args):

    args.rank = rank

    setup(args)

    # create a logger
    # logger = create_logger(os.path.join(args.dump_path, "log"), rank=rank)

    # logger.info("============ Initialized logger ============")
    # logger.info(
    #     "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    # )

    # setup mp_model and devices for this process
    dev0 = (rank * 2) % args.world_size
    dev1 = (rank * 2 + 1) % args.world_size
    args.in_size = 10
    args.out_size = 5
    mp_model = ToyMpModel(dev0, dev1, in_size, out_size)
    ddp_mp_model = DistributedDataParallel(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    args.data_length = 100
    args.batch_size = 50
    rand_loader = DataLoader(
        dataset=RandomDataset(
            size=args.input_size,
            length=args.data_size),
        batch_size=args.batch_size,
        shuffle=True)

    optimizer.zero_grad()  # zona - unclear why and if necessary...

    for minibatch, (data, labels) in enumerate(rand_loader):
        # zona data = data.to(device)
        output = ddp_mp_model(data)
        print("minibatch ",minibatch, " input size ", data.size(), " output size ", output.size(), " labels size ", labels.size())

        # zona labels = labels.to(dev1)
        loss_fn(outputs, labels).backward()
        optimizer.step()

    cleanup()

if __name__ == "__main__":

    args = parser.parse_args()

    # when using torch distributed (ddp) initialization via file
    # confirm the sync file doesn't exist before running init_process_group()
    # if so - delete it
    # see https://pytorch.org/docs/stable/distributed.html
    syncfile = args.dist_url.split('file:\\')[1]
    if os.path.exists(syncfile):
        os.remove(syncfile)
        print("removed")
    else:
        print("not there")

    # zona - how many gpus...
    args.world_size = torch.cuda.device_count()
    # args.world_size = 1

    print("----- logs at {} -----".format(args.dump_path))
    print("spawning {} processes".format(args.world_size))

    mp.spawn(fn=test_01,
             args=(args,),
             nprocs=args.world_size,
             join=True)
