'''
PyTorch PGM - following Northeastern course by Robert O. Ness
https://bookdown.org/robertness/causalml/docs/

Got side-tracked into attempting a more simple example - that of:
The example however is taken from example 1.33, Neapolitan, page 44,45
which was unsuccessful (see "got about as far as...)

Also see 
Blei, D., Carin, L., & Dunson, D. (2010). Probabilistic Topic Models. IEEE Signal Processing Magazine, 27(6), 55–65. https://doi.org/10.1109/MSP.2010.938079

Got about as far as forward()...
the question becomes...
for .forward() we need to decide in advance what the input 'data' is (x) and what is it predicting
is this the generative path?
is this in the "posterior" (a.k.a. diagnostic) path?
If there are multiple variables we could predict a lot of different combinations..

It also begs the question - is this supervised data? Or would we use a pretext task / context prediction approach?
'''

import torch
import torch.nn.functional as F
import random
import argparse
import logging
# from logging import getLogger
import numpy as np


# runtime arguments
parser = argparse.ArgumentParser(description="Implementation of DeepCluster-v2")
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")


parser.add_argument("--a_choice", type=str, default="10", choices=["10", "11"],
                    help="nominal argument")
parser.add_argument("--a_string", type=str, default="foobar",
                    help="string argument")
parser.add_argument("--a_number", type=int, default=31,
                    help="numeric argument")
parser.add_argument("--debug", default=False, action="store_true",
                    help="an example of a 'switch' that can be set, I do not recommend this format, preferring choice with a default")


class BayesianNetwork(torch.nn.Module):
    def __init__(self):
        super(BayesianNetwork).__init__()

        # random variables - sizes and initial values all are in [T, F]
        # h = history of smoking
        # b = bronchitis
        # l = lung cancer
        # c = chest x-ray
        # f = fatigue
        hs = torch.tensor([.2, .8])
        br = torch.tensor([[.25, .75], [.05, .95]])
        lc = torch.tensor([[.003, .997], [.00005, .99995]])
        cx = torch.tensor([[.6, .4], [.02, .98]])
        fa = torch.tensor([[.75, .25], [.1, .9], [.5, .5], [.05, .95]])

        self.history_of_smoking = torch.nn.Parameter(data=hs, requires_grad=True)
        self.bronchitis = torch.nn.Parameter(data=br, requires_grad=True)
        self.lung_cancer = torch.nn.Parameter(data=lc, requires_grad=True)
        self.chest_xray = torch.nn.Parameter(data=cx, requires_grad=True)
        self.fatigue = torch.nn.Parameter(data=fa, requires_grad=True)

    def forward(self):

        # we are given 
        p_hs = F.sigmoid(self.history_of_smoking())
        p_br = F.sigmoid(self )
        p_

def main(args):

    # logger explained: https://stackoverflow.com/questions/7016056/python-logging-not-outputting-anything
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # also use DEBUG, WARNING, NOTSET, INFO etc

    logger.info(f"args: {args}")

    # Section 2.3.3 ("Gaussian Mixture Model") illustrates the underlying assumptions of our model
    # https://bookdown.org/robertness/causalml/docs/causal-inference-overview-and-course-goals.html#case-studies

    # Gaussian Mixture Model assumes there is a latent variable 

    # the following defines a dirichlet distribution having 3 classes and specific distribution.
    # We sample 10 instances from that distribution
    # in other words - we have 10 urns, each having a distinct distribution such that if there were many urns they would have our "alpha" distribuion 
    alpha = [9, 3, 1]
    num_samples = 10
    dirichlet_dist = np.random.dirichlet(alpha=alpha, size=num_samples)
    logger.info(f"{dirichlet_dist}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
