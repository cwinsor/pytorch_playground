
import os
import numpy as np
import argparse
import logging

import torch
from torch_geometric.data import Data, Dataset, HeteroData, download_url


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

    def process(self):

        NUM_ORIGINAL_TWEETS = 2
        NUM_ORIGINAL_TWEET_FEATURES = 2

        NUM_TWEET_DAILY_SUMMARIES = 6
        NUM_TWEET_DAILY_SUMMARY_FEATURES = 2

        NUM_USERS = 11
        NUM_USER_FEATURES = 3

        # NODES
        # 2 original tweets, two features each
        original_tweets = np.random.randint(100, size=(NUM_ORIGINAL_TWEETS, NUM_ORIGINAL_TWEET_FEATURES))
        # 6 daily summaries 2 features each
        daily_summaries = np.random.randint(100, size=(NUM_TWEET_DAILY_SUMMARIES, NUM_TWEET_DAILY_SUMMARY_FEATURES))
        # 9 users 3 features each
        users = np.random.randint(100, size=(NUM_USERS, NUM_USER_FEATURES))

        # EDGES
        user_retweet_to_daily_tweet_activity = np.array([
            [0, 0],
            [1, 0],
            [3, 1],
            [4, 1],
            [5, 1],
            [6, 1],
            [8, 2],
            [9, 2],
            [10, 2],
            [1, 3],
            [2, 3],
            [7, 4]])

        summary_to_original_tweet = np.array([
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 1],
            [4, 1],
            [5, 1]])

        data = HeteroData()
        data['original_tweet'].x = torch.tensor(original_tweets)
        data['daily_tweet_activity'].x = torch.tensor(daily_summaries)
        data['user'].x = torch.tensor(users)
        data['user', 'retweeted_contributing_to', 'daily_tweet_activity'] = torch.tensor(user_retweet_to_daily_tweet_activity)
        data['daily_tweet_activity', 'refers_to', 'original_tweet'] = torch.tensor(summary_to_original_tweet)

        file_idx = -1
        for raw_path in self.raw_paths:
            file_idx += 1

            data = HeteroData()
            data['original_tweet'].x = torch.tensor(original_tweets)
            data['daily_tweet_activity'].x = torch.tensor(daily_summaries)
            data['user'].x = torch.tensor(users)
            data['user', 'retweeted_contributing_to', 'daily_tweet_activity'] = torch.tensor(user_retweet_to_daily_tweet_activity)
            data['daily_tweet_activity', 'refers_to', 'original_tweet'] = torch.tensor(summary_to_original_tweet)

            torch.save(data, os.path.join(self.processed_dir, f'data_{file_idx}.pt'))

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


def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc
    logger.info(f"args: {args}")

    dataset = GeoCoV19GraphDataset(root=r'D:\dataset_covid_GeoCovGraph')
    data = dataset[0]
    print(data)

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
