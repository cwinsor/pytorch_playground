
'''
Experiment:
Pytorch Dataset using PyG HeteroData
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html#
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html

'''

import os
import numpy as np
import argparse
import logging
import pandas as pd
import ijson

import torch
from torch_geometric.data import Data, Dataset, HeteroData, download_url
from torch_geometric.utils import to_networkx
import networkx as nx
from matplotlib import pyplot as plt

# runtime arguments
parser = argparse.ArgumentParser(description="Pytorch Geometric Twitter Dataset Preprocessing")

parser.add_argument("--a_choice", type=str, default="10", choices=["10", "11"],
                    help="nominal argument")
parser.add_argument("--a_string", type=str, default="foobar",
                    help="string argument")
parser.add_argument("--a_number", type=int, default=31,
                    help="numeric argument")
parser.add_argument("--debug", default=False, action="store_true",
                    help="an example of a 'switch', not recommended preferring choice with a default")


class GeoCoV19GraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root

    @property
    def raw_file_names(self):
        # return ['ids_geo_2020-02-01.jsonl']
        return ['ids_geo_2020-02-01.jsonl', 'ids_geo_2020-02-02.jsonl']

    @property
    def processed_file_names(self):
        # return ['ids_geo_2020-02-01.pt']
        # return ['ids_geo_2020-02-01.pt', 'ids_geo_2020-02-02.pt']
        return ['data_0.pt', 'data_1.pt']

    def download(self):
        rfn = self.raw_file_names
        assert False, (f'did not find raw data {self.root}\\raw\\{rfn} and download is not implemented - refer to \n'
                       'https://paperswithcode.com/dataset/geocov19 and \n'
                       'https://crisisnlp.qcri.org/covid19 and \n'
                       'https://github.com/docnow/hydrator')

    def _is_retweet(self, tweet):
        is_retweet = "retweeted_status" in tweet.keys()
        return is_retweet

    def _load_node_original_tweets(self, retweets, encoders=None):
        mapping = {tweet["retweeted_status"]["id"]: i for i, tweet in enumerate(retweets)}

        x = None
        x = torch.rand(len(mapping), 2)  # zona temporary
        # if encoders is not None:
        #     xs = [encoder(retweets[col]) for col, encoder in encoders.items()]
        #     x = torch.cat(xs, dim=-1)
        return x, mapping

    def _load_node_users(self, retweets, encoders=None):
        mapping = {tweet["user"]["id"]: i for i, tweet in enumerate(retweets)}

        x = None
        x = torch.rand(len(mapping), 3)  # zona temporary
        # if encoders is not None:
        #     xs = [encoder(retweets[col]) for col, encoder in encoders.items()]
        #     x = torch.cat(xs, dim=-1)
        return x, mapping

    def _load_edge_user_retweets_original_tweet(self, retweets, src_mapping, dst_mapping, encoders=None):
        src_dst = [
            (src_mapping[tweet["user"]["id"]],
             dst_mapping[tweet["retweeted_status"]["id"]]) for tweet in retweets
        ]
        edge_index = torch.tensor(src_dst)

        edge_attr = None
        # if encoders is not None:
        #     edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        #     edge_attr = torch.cat(edge_attrs, dim=-1)

        return edge_index, edge_attr

    def process(self):

        file_idx = -1
        for raw_path in self.raw_paths:
            file_idx += 1
            print(f"reading {raw_path}")

            # Read data from file
            with open(raw_path, "r", encoding="utf-8") as f:
                tweets = ijson.items(f, "", multiple_values=True)

                # just retweets
                retweets = [
                    tweet for tweet in tweets if self._is_retweet(tweet)
                    ]

                # NODES
                original_tweet_x, original_tweet_mapping = self._load_node_original_tweets(retweets=retweets, encoders=None)
                users_x, users_mapping = self._load_node_users(retweets=retweets, encoders=None)
                # EDGES
                edge_index, edge_label = self._load_edge_user_retweets_original_tweet(
                    retweets=retweets,
                    src_mapping=users_mapping,
                    dst_mapping=original_tweet_mapping,
                    encoders=None)

                data = HeteroData()
                data['user'].x = users_x
                data['original_tweet'].x = original_tweet_x
                data['user', 'retweets', 'original_tweet'].edge_index = edge_index
                print(data)

                torch.save(data, os.path.join(self.processed_dir, f'data_{file_idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc
    logger.info(f"args: {args}")

    dataset = GeoCoV19GraphDataset(root=r'D:\dataset_covid_GeoCovGraph')

    logger.info("------ data[0] ------")
    data = dataset[0]
    logger.info(data)

    logger.info("------ data[1] ------")
    data = dataset[0]
    logger.info(data)

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
