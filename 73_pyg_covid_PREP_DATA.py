'''
Pytorch Geometric Twitter Dataset Preprocessing

Process the raw GeoCoV19 data (https://arxiv.org/abs/2005.11177)
into TUDataset format as described at https://chrsmrrs.github.io/datasets/docs/format/

It is assumed the GeoCoV19 data has been rehydrated, i.e. we have
access to (ids_geo_2020-02-01.jsonl)

TUDataset format will give us access to "Dataset" and "DataLoader".
'''

import os
import argparse
import logging

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import ijson
import json

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange

import matplotlib.pyplot as plt
from wordcloud import WordCloud


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


def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc
    logger.info(f"args: {args}")

    assert pyro.__version__.startswith('1.8.4')

    # read the raw .json parsing out the info we want
    # ijson references:
    #   git: https://github.com/ICRAR/ijson
    #   memory: https://pythonspeed.com/articles/json-memory-streaming/

    # Reference for utf-8
    # https://www.freecodecamp.org/news/what-is-utf-8-character-encoding/

    # We observe the following fields (in all entries or in some entries)
    #  -------- in_all 24 --------

    # user

    # id
    # full_text
    # retweet_count
    # favorite_count

    # retweeted
    # favorited

    # in_reply_to_status_id_str
    # in_reply_to_status_id
    # in_reply_to_user_id
    # in_reply_to_screen_name
    # in_reply_to_user_id_str

    # truncated
    # entities
    # id_str
    # place
    # source
    # contributors
    # display_text_range
    # created_at
    # is_quote_status
    # coordinates
    # lang
    # geo

    #  -------- in_any 35 --------
    # quoted_status_permalink
    # retweeted_status <--- in event this is a retweet - this dictionary includes reference to original tweet
    # quoted_status_id_str
    # quoted_status
    # withheld_copyright
    # quoted_status_id
    # scopes
    # withheld_in_countries
    # withheld_scope
    # extended_entities
    # possibly_sensitive

    logger.info("reading...")
    local_filename = r"D:\dataset_covid_GeoCoV19\2020_02_01_05\ids_geo_2020-02-01.jsonl"
    tweet_data = []
    count = 0
    CUTOFF = math.inf  # math.inf

    in_all = set()
    in_any = set()

    tweet_id_set = {}
    list_of_cotton = list()

    with open(local_filename, "r", encoding="utf-8") as f:
        objects = ijson.items(f, "", multiple_values=True)
        for object in objects:
            count += 1
            if count > CUTOFF:
                break

            # identify what keys are always present...
            if count == 1:
                in_all = set(object.keys())
            else:
                in_all = in_all.intersection(set(object.keys()))
            in_any = in_any.union(set(object.keys()))

            # common parsing
            tid = object["id"]
            full_text = object["full_text"]

            # identify unique tweets
            tweet_id_set = tweet_id_set.union(tid)

            if "retweeted_status" in object.keys():
                retweet_original_tid = object["retweeted_status"]["id"]
                tweet_id_set = tweet_id_set.union(tid)
                retweet_count[retweet_original_tid] += 1

            # if tid in [1223491082425684000, 1223519633711386600]:
            #     # print individual field values
            #     print(f"-----{tid}----------")
            #     for k, v in object.items():
            #         print(f"{k}: {v}")
            #     retweeted_status = object["retweeted_status"]
            #     print(f"-----retweeted_status----------")
            #     for k, v in retweeted_status.items():
            #         print(f"{k}: {v}")



            if "The coronavirus has become a global pandemic" in full_text:
                list_of_cotton.append(tid)

            this_tweet = {}
            this_tweet["tid"] = object["id"]
            # created = object["created_at"]
            this_tweet["text"] = object["full_text"]
            tweet_data.append(this_tweet)
    logger.info(f"number of tweets {len(tweet_data)}")

    print(f" -------- in_all {len(in_all)} --------")
    [print(x) for x in in_all]
    print(f" -------- in_any {len(in_any)} --------")
    adders = in_any.difference(in_all)
    [print(x) for x in adders]
    # to json file
    logger.info("writing...")
    local_filename = "covid_tweets_preprocessed.json"
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(tweet_data, f, ensure_ascii=False, indent=4)

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
