'''
Pyro (PGM using Torch) - following http://pyro.ai/examples/prodlda.html

"This tutorial implements the ProdLDA topic model from Autoencoding Variational
Inference For Topic Models by Akash Srivastava and Charles Sutton. "

'''

import os
import argparse
import logging

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS


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
    # Enable smoke test - run the notebook cells on CI.
    smoke_test = 'CI' in os.environ
    print(f"smoke_test {smoke_test}")
          
    def model(counts):
        theta = pyro.sample('theta', dist.Dirichlet(torch.ones(6)))
        total_count = int(counts.sum())
        pyro.sample('counts', dist.Multinomial(total_count, theta), obs=counts)

    data = torch.tensor([5, 4, 2, 5, 6, 5, 3, 3, 1, 5, 5, 3, 5, 3, 5, \
                        3, 5, 5, 3, 5, 5, 3, 1, 5, 3, 3, 6, 5, 5, 6])
    counts = torch.unique(data, return_counts=True)[1].float()

    nuts_kernel = NUTS(model)
    num_samples, warmup_steps = (1000, 200) if not smoke_test else (10, 10)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(counts)
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    means = hmc_samples['theta'].mean(axis=0)
    stds = hmc_samples['theta'].std(axis=0)
    print('Inferred dice probabilities from the data (68% confidence intervals):')
    for i in range(6):
        print('%d: %.2f ± %.2f' % (i + 1, means[i], stds[i]))

    logger.info("done here")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
