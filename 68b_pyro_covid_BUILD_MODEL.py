'''
Pyro (PGM using Torch) - following http://pyro.ai/examples/prodlda.html

The code implements the ProdLDA topic model from Autoencoding Variational
Inference For Topic Models by Akash Srivastava and Charles Sutton.

It is based on above tutorial.

The dataset is the GeoCoV19, specifically the first week in February.
'''

import os
import sys
import datetime
import argparse
import logging
import wandb

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange
import ijson

import matplotlib.pyplot as plt
from wordcloud import WordCloud


# runtime arguments
parser = argparse.ArgumentParser(description="Implementation of DeepCluster-v2")



parser.add_argument("--run_name", type=str, default=datetime.datetime.now().strftime("%m%d_%H%M%S"),
                    help="unique name for this run")
parser.add_argument("--smoke_test", default=False, action="store_true",
                    help="run the short smoke test")
parser.add_argument("--output_dir", type=str, default="output",
                    help="folder to save snapshots of model")
parser.add_argument("--save_period_in_batches", type=int, default=2000,
                    help="number of batches between model saves")

parser.add_argument("--num_topics", type=int, default=18,
                    help="number of topics (hidden representation, embeddings)")
parser.add_argument("--batch_size", type=int, default=32,
                    help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3,
                    help="learning rate")
parser.add_argument("--num_epochs", type=int, default=50,
                    help="number of epochs")
                    




parser.add_argument("--z_example_switch", default=False, action="store_true",
                    help="an example of a 'switch'")
parser.add_argument("--z_example_choice", type=str, default="10", choices=["10", "11"],
                    help="nominal argument")
parser.add_argument("--z_example_string", type=str, default="foobar",
                    help="string argument")
parser.add_argument("--z_example_number", type=int, default=31,
                    help="numeric argument")

class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        # NB: here we set `affine=False` to reduce the number of learning parameters
        # See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        # for the effect of this flag in BatchNorm1d
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        return logtheta_loc, logtheta_scale


class Decoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.encoder = Encoder(vocab_size, num_topics, hidden, dropout)
        self.decoder = Decoder(vocab_size, num_topics, dropout)

    def model(self, docs):
        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
            logtheta_loc = docs.new_zeros((docs.shape[0], self.num_topics))
            logtheta_scale = docs.new_ones((docs.shape[0], self.num_topics))
            logtheta = pyro.sample(
                "logtheta", pyro.distributions.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta = F.softmax(logtheta, -1)

            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            count_param = self.decoder(theta)
            # Currently, PyTorch Multinomial requires `total_count` to be homogeneous.
            # Because the numbers of words across documents can vary,
            # we will use the maximum count accross documents here.
            # This does not affect the result because Multinomial.log_prob does
            # not require `total_count` to evaluate the log probability.
            total_count = int(docs.sum(-1).max())
            pyro.sample(
                'obs',
                pyro.distributions.Multinomial(total_count, count_param),
                obs=docs
            )

    def guide(self, docs):
        pyro.module("encoder", self.encoder)
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution,
            # where μ and Σ are the encoder network outputs
            logtheta_loc, logtheta_scale = self.encoder(docs)
            logtheta = pyro.sample("logtheta", pyro.distributions.Normal(logtheta_loc, logtheta_scale).to_event(1))

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T


def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc
    logger.info(f"args: {args}")

    # Initialize wandb as soon as possible to log all stdout to the cloud
    wandb.init(project="pyro_twitter", config=args)

    def save_checkpoint(step):
        file_name = f"{args.output_dir}/{args.run_name}_{step}.pt"
        logger.info(f"saving checkpoint to {file_name}")
        torch.save(prodLDA, file_name)

    def restore_checkpoint(step):
        file_name = f"{args.output_dir}/{args.run_name}_{step}.pt"
        logger.info(f"restoring checkpoint {file_name}")
        # m = torch.load(prodLDA, file_name)
        m = torch.load(file_name)
        return m

    assert pyro.__version__.startswith('1.8.4')

    # read the preprocessed tweet data
    logger.info("reading...")
    local_filename = r"covid_tweets_preprocessed.json"

    raw_tid_to_ttext = {}
    with open(local_filename, "r", encoding="utf-8") as f:
        objects = ijson.items(f, "item", multiple_values=True)
        for object in objects:
            tweet_id = object["tid"]
            tweet_text = object["text"]
            # sender ...
            # recipient ...
            # if tweet_id not in raw_tid_to_ttext:
            #     raw_tid_to_ttext[tweet_id] = set()
            # raw_tid_to_ttext[tweet_id].add(tweet_text)

            if tweet_id not in raw_tid_to_ttext:
                raw_tid_to_ttext[tweet_id] = tweet_text

    # sanity check the encoding - should be mostly 1 or 2 bytes/character
    # n = 0
    # for k, v in raw_tid_to_ttext.items():
    #     n+=1
    #     if n>20:
    #         break
    #     # print(f"key {k}")
    #     print(f"value: {v}")
    #     # v = "a" * 1000
    #     # v = "t"
    #     size = sys.getsizeof(v)-49
    #     length = len(v)
    #     bytes_per_char = size / length
    #     print(f"bytes_per_char {bytes_per_char}")

    # bytes_per_char = [(sys.getsizeof(v)-49)/len(v) for k, v in raw_tid_to_ttext.items()]
    # hgram = np.histogram(bytes_per_char)
    # print(hgram)

    # Vectorize the corpus. This means:
    # Creating a dictionary where each word corresponds to an (integer) index
    # Removing rare words (words that appear in less than 20 documents)
    # Removing common words (words that appear in more than 50% of the documents)
    # Counting how many times each word appears in each document

    # vectorizer = CountVectorizer(max_df=0.5, min_df=20, stop_words='english')
    # vectorizer = CountVectorizer().fit(raw_tid_to_ttext.values())

    min_document_frequency = 2. / len(raw_tid_to_ttext)  # term needs to be used at least N times in the corpus
    max_document_frequency = 0.05  # term should show up in less than specified % of documents
    vectorizer = CountVectorizer(
        min_df=min_document_frequency,
        max_df=max_document_frequency).fit(raw_tid_to_ttext.values())

    document_word = torch.from_numpy(vectorizer.transform(raw_tid_to_ttext.values()).toarray())  # document-to-term

    # document_word = torch.from_numpy(vectorizer.fit_transform(raw_tid_to_ttext.values()).toarray())

    logger.info(f"number of tweets {document_word.shape[0]}")
    logger.info(f"vocabulary size  {document_word.shape[1]}")

    # sanity check torch memory usage...
    logger.info("document_word as as int ----------")
    logger.info(f"document_word.dtype {document_word.dtype})")
    logger.info(f"document_word.shape[0] * document_word.shape[0] = {document_word.shape[0] * document_word.shape[1]}")
    logger.info(f"document_word (in bytes) {document_word.element_size() * document_word.nelement()}")
    logger.info("document_word as float ----------")
    document_word = document_word.float()
    logger.info(f"document_word.dtype {document_word.dtype})")
    logger.info(f"document_word.shape[0] * document_word.shape[0] = {document_word.shape[0] * document_word.shape[1]}")
    logger.info(f"document_word (in bytes) {document_word.element_size() * document_word.nelement()}")

    # logger.info(f"vocabulary_ {vectorizer.vocabulary_}")

    vocabulary = pd.DataFrame(columns=['word', 'index'])
    vocabulary['word'] = vectorizer.get_feature_names()
    vocabulary['index'] = vocabulary.index
    logger.info('Dictionary size: %d' % len(vocabulary))
    logger.info('Corpus size: {}'.format(document_word.shape))

    logger.info(f'dictionary[50:55]\n{vocabulary[50:55]}')
    logger.info(f'Corpus[0]\n{document_word[5]}')

    # implement the model in Pyro ...

    # setting global variables
    seed = 0
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # training
    pyro.clear_param_store()
    prodLDA = ProdLDA(
        vocab_size=document_word.shape[1],
        num_topics=args.num_topics,
        hidden=100 if not args.smoke_test else 10,
        dropout=0.2
    )
    prodLDA.to(device)

    optimizer = pyro.optim.Adam({"lr": args.learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    num_batches = int(math.ceil(document_word.shape[0] / args.batch_size)) if not args.smoke_test else 1

    bar = trange(args.num_epochs)
    batch = 0
    for epoch in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch += 1
            batch_docs = document_word[i * args.batch_size:(i + 1) * args.batch_size, :]
            batch_docs = batch_docs.to(device)
            loss = svi.step(batch_docs)
            running_loss += loss / batch_docs.size(0)

            wandb.log(
                {
                    "train_loss": loss,
                    "running_loss": running_loss,
                    "optimizer_args": optimizer.pt_optim_args,
                    "epoch": epoch,
                },
                step=batch,
            )

            if batch % args.save_period_in_batches == 0:
                save_checkpoint(f"{batch:04}")

        bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))

    # save final model
    save_checkpoint("FINAL")

    # restore from checkpoint
    prodLDA = restore_checkpoint("FINAL")

    # visualization using word cloud
    def plot_word_cloud(b, ax, v, n):
        sorted_, indices = torch.sort(b, descending=True)
        df = pd.DataFrame(indices[:100].numpy(), columns=['index'])
        words = pd.merge(df, vocabulary[['index', 'word']],
                         how='left', on='index')['word'].values.tolist()
        sizes = (sorted_[:100] * 1000).int().numpy().tolist()
        freqs = {words[i]: sizes[i] for i in range(len(words))}
        wc = WordCloud(background_color="white", width=800, height=500)
        wc = wc.generate_from_frequencies(freqs)
        ax.set_title('Topic %d' % (n + 1))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")

    # if not smoke_test:
    if True:
        logger.info("----- visualize the wordcloud -----")
        beta = prodLDA.beta()
        fig, axs = plt.subplots(7, 3, figsize=(14, 24))
        for n in range(beta.shape[0]):
            i, j = divmod(n, 3)
            plot_word_cloud(beta[n], axs[i, j], vocabulary, n)
        axs[-1, -1].axis('off')

    plt.show()

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
