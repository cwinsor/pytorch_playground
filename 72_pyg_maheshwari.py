'''
Graph Self-supervised Learning example using PyG

Large portions of code from Maheshwari et al.
https://medium.com/stanford-cs224w/self-supervised-learning-for-graphs-963e03b9f809

'''

# # we use content from Maheshwari (GraphSSL)
import sys
sys.path.append(r"D:\code_pytorch\my_playground\GraphSSL")

import datetime
import argparse
import logging
import wandb
from tqdm import trange

import torch

from GraphSSL.data import load_dataset, split_dataset, build_loader
from GraphSSL.model import Encoder
from GraphSSL.loss import infonce

# runtime arguments
parser = argparse.ArgumentParser()

parser.add_argument("--run_name", type=str,
                    default=datetime.datetime.now().strftime("%m%d_%H%M%S"),
                    help="unique name for this run")
parser.add_argument("--output_dir", type=str, default="output",
                    help="folder where model snapshots and logs are saved")
parser.add_argument("--device", type=str,
                    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

parser.add_argument("--lr", dest="lr", action="store", default=0.001, type=float)
parser.add_argument("--epochs", dest="epochs", action="store", default=20, type=int)
parser.add_argument("--batch_size", dest="batch_size", action="store", default=64, type=int)
parser.add_argument("--num_workers", dest="num_workers", action="store", default=8, type=int)
parser.add_argument("--dataset", dest="dataset", action="store", required=True, type=str,
                    choices=["proteins", "enzymes", "collab", "reddit_binary", "reddit_multi", "imdb_binary",
                             "imdb_multi", "dd", "mutag", "nci1"],
                             help="dataset on which you want to train the model")
parser.add_argument("--model", dest="model", action="store", default="gcn", type=str,
                    choices=["gcn", "gin", "resgcn", "gat", "graphsage", "sgc"],
                    help="he model architecture of the GNN Encoder")
parser.add_argument("--feat_dim", dest="feat_dim", action="store", default=128, type=int,
                    help="dimension of node features in GNN")
parser.add_argument("--layers", dest="layers", action="store", default=3, type=int,
                    help=" number of layers of GNN Encoder")
parser.add_argument("--loss", dest="loss", action="store", default="infonce", type=str,
                    choices=["infonce", "jensen_shannon"],
                    help="loss function for contrastive training")
parser.add_argument("--augment_list", dest="augment_list", nargs="*",
                    default=["edge_perturbation", "node_dropping"], type=str,
                    choices=["edge_perturbation", "diffusion", "diffusion_with_sample", "node_dropping",
                             "random_walk_subgraph", "node_attr_mask"],
                              help="augmentations to be applied as space separated strings")
parser.add_argument("--train_data_percent", dest="train_data_percent", action="store", default=1.0, type=float)


def run_batch(args, epoch, mode, dataloader, model, optimizer):
    if mode == "train":
        model.train()
    elif mode == "val" or mode == "test":
        model.eval()
    else:
        assert False, "Wrong Mode:{} for Run".format(mode)

    losses = []
    contrastive_fn = eval(args.loss + "()")

    with trange(len(dataloader), desc="{}, Epoch {}: ".format(mode, epoch)) as t:
        for data in dataloader:
            data.to(args.device)

            # readout_anchor is the embedding of the original datapoint x on passing through the model
            readout_anchor = model((data.x_anchor, data.edge_index_anchor, data.x_anchor_batch))

            # readout_positive is the embedding of the positively augmented x on passing through the model
            readout_positive = model((data.x_pos, data.edge_index_pos, data.x_pos_batch))

            # negative samples for calculating the contrastive loss is computed in contrastive_fn
            loss = contrastive_fn(readout_anchor, readout_positive)

            if mode == "train":
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # keep track of loss values
            losses.append(loss.item())
            t.set_postfix(loss=losses[-1])
            t.update()

    # gather the results for the epoch
    epoch_loss = sum(losses) / len(losses)
    return epoch_loss


def checkpoint_filename(step, args):
    file_name = f"{args.output_dir}/{args.run_name}_{step}.pt"
    return file_name


def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc
    logger.info(f"args: {args}")

    # Initialize wandb as soon as possible to log all stdout to the cloud
    wandb.init(project="pyg_practice", config=args)

    dataset, input_dim, num_classes = load_dataset(args.dataset)

    # split the data into train / val / test sets
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, args.train_data_percent)

    # build_loader is a dataloader which gives a paired sampled - the original x and the positively
    # augmented x obtained by applying the transformations in the augment_list as an argument
    train_loader = build_loader(args, train_dataset, "train")
    val_loader = build_loader(args, val_dataset, "val")
    test_loader = build_loader(args, test_dataset, "test")

    # easy initialization of the GNN model encoder to map graphs to embeddings needed for contrastive training
    model = Encoder(input_dim, args.feat_dim, n_layers=args.layers, gnn=args.model)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_train_loss, best_val_loss = float("inf"), float("inf")

    for epoch in range(args.epochs):

        train_loss = run_batch(args, epoch, "train", train_loader, model, optimizer)
        logger.info(f"epoch {epoch} train loss {train_loss}")

        val_loss = run_batch(args, epoch, "val", val_loader, model, optimizer)
        logger.info(f"epoch {epoch}   val loss {train_loss}")

        # save model
        if val_loss < best_val_loss:
            best_epoch, best_train_loss, best_val_loss, is_best_loss = epoch, train_loss, val_loss, True

            model.save_checkpoint(args.output_dir,
                                  optimizer, epoch, best_train_loss, best_val_loss, is_best_loss)

            wandb.log(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                step=epoch * args.batch_size
            )

    logger.info(f"Best validation loss at epoch {epoch}: val {best_val_loss:.3f} train {best_train_loss:.3f}")
    model.eval()
    test_loss = run_batch(args, best_epoch, "test", test_loader, model, optimizer)
    logger.info(f"Test loss using model from epoch {best_epoch}: {test_loss:.3f}")
    wandb.log(
        {
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "test_loss (final)": test_loss,
        })

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
