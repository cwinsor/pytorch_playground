'''
Pyro (PGM using Torch) - following http://pyro.ai/examples/prodlda.html

The code implements the ProdLDA topic model from Autoencoding Variational
Inference For Topic Models by Akash Srivastava and Charles Sutton.

It is based on above tutorial.

The dataset is the GeoCoV19, specifically the first week in February.
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
parser = argparse.ArgumentParser(description="Implementation of DeepCluster-v2")

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

    logger.info("reading...")
    local_filename = r"D:\dataset_covid_GeoCoV19\2020_02_01_05\ids_geo_2020-02-01.jsonl"
    tweet_data = []
    count = 0
    CUTOFF = 5000

    with open(local_filename, "r", encoding="utf-8") as f:
        objects = ijson.items(f, "", multiple_values=True)
        for object in objects:
            count += 1
            if count > CUTOFF:
                break

            this_tweet = {}
            this_tweet["tid"] = object["id"]
            # created = object["created_at"]
            this_tweet["text"] = object["full_text"]
            tweet_data.append(this_tweet)
    logger.info(f"number of tweets {len(tweet_data)}")

    # to json file
    logger.info("writing...")
    local_filename = "covid_tweets_preprocessed.json"
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(tweet_data, f, ensure_ascii=False, indent=4)

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
