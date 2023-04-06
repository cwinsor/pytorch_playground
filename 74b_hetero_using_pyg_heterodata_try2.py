
'''
Experiment:
Pytorch Dataset using PyG HeteroData
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html#
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html

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
        return ['ids_geo_2020-02-01.jsonl', 'ids_geo_2020-02-02.jsonl']

    @property
    def processed_file_names(self):
        return ['ids_geo_2020-02-01.pt', 'ids_geo_2020-02-02.pt']

    def download(self):
        rfn = self.raw_file_names
        assert False, (f'did not find raw data {self.root}\\raw\\{rfn} and download is not implemented - refer to \n'
                       'https://paperswithcode.com/dataset/geocov19 and \n'
                       'https://crisisnlp.qcri.org/covid19 and \n'
                       'https://github.com/docnow/hydrator')

    def _is_retweet(self, tweet):
        is_retweet = "retweeted_status" in tweet.keys()
        return is_retweet

    def _map_and_encode_original_tweet(self, tweet):
                #                 original_tid = object["retweeted_status"]["id"]
        #                 tweet_id_set = tweet_id_set.union(original_tid)
        #                 retweet_count[retweet_original_tid] += 1
        return "zona"

    def _map_and_encode_user(self, tweet):
        return "zona"

    def _establish_edge_user_retweets_original_tweet(self, tweet):
        return "zona"

    def _load_node_original_tweets(self, retweets, encoders=None):
        mapping = {tweet["retweeted_status"]["id"]: i for i, tweet in enumerate(retweets)}

        x = None
        # foo = [retweets[n]["retweeted_status"]["user"]["name"] for n in mapping.values()]
        xs = torch.rand(len(mapping))
        x = torch.cat(xs, dim=-1)
        if encoders is not None:
            xs = [encoder(retweets[col]) for col, encoder in encoders.items()]
            x = torch.cat(xs, dim=-1)
        return x, mapping

    def process(self):

        file_idx = -1
        for raw_path in self.raw_paths:
            file_idx += 1

            # Read data from file
            with open(raw_path, "r", encoding="utf-8") as f:
                tweets = ijson.items(f, "", multiple_values=True)

                # just retweets
                retweets = [
                    tweet for tweet in tweets if self._is_retweet(tweet)
                    ]

                original_tweet_x, original_tweet_mapping = self._load_node_original_tweets(retweets=retweets, encoders=None)

                # # NODES
                # original_tweet_x, original_tweet_mapping = [
                #     self._map_and_encode_original_tweet(tweet) for tweet in retweets if tweet not in 
                #     ]
                # user_x, user_mapping = [
                #     self._map_and_encode_user(tweet) for tweet in retweets
                #     ]

                data = HeteroData()
                # data['user'].num_nodes = user_x
                data['original_tweet'].x = original_tweet_x
                print(data)

                # # EDGES
                # edge_index = [
                #     self._establish_edge_user_retweets_original_tweet(tweet) for tweet in retweets
                #     ]
            
                data['user', 'retweets', 'original_tweet'].edge_index = edge_index
                print(data)

                torch.save(data, os.path.join(self.processed_dir, f'data_{file_idx}.pt'))

        ################################3
        # CUTOFF = 5  # math.inf
        # file_idx = -1
        # for raw_path in self.raw_paths:
        #     file_idx += 1

        #     # Read data from file
        #     with open(raw_path, "r", encoding="utf-8") as f:
        #         objects = ijson.items(f, "", multiple_values=True)

        #         tweet_count = 0
        #         for object in objects:
        #             tweet_count += 1
        #             if tweet_count > CUTOFF:
        #                 break

        #             # common parsing
        #             tid = object["id"]
        #             full_text = object["full_text"]

        #             tweet_id_set = tweet_id_set.union(tid)

        #             if "retweeted_status" in object.keys():
        #                 original_tid = object["retweeted_status"]["id"]
        #                 tweet_id_set = tweet_id_set.union(original_tid)
        #                 retweet_count[retweet_original_tid] += 1

        #         node_feats = torch.rand((18,8))
        #         # data object - a single graph
        #         data = Data(x=node_feats,
        #                     edge_index=edge_index,
        #                     edge_attr=edge_feats,
        #                     y=label,
        #                     smiles=mol['smiles']
        #         )

        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue

        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    # # zona - experimental
    # @property
    # def num_nodes(self):
    #     return NUM_ORIGINAL_TWEETS + NUM_TWEET_DAILY_SUMMARIES + NUM_USERS



def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc
    logger.info(f"args: {args}")

    dataset = GeoCoV19GraphDataset(root=r'D:\dataset_covid_GeoCovGraph')
    data = dataset[0]
    print(data)

    # check ...
    # <nothing here...>

    # visualize
    data_h = data.to_homogeneous()
    data_h_nx = to_networkx(data_h, to_undirected=True)
    nx.draw(data_h_nx)

    # graph = dataset[1]
    # print(graph)

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
