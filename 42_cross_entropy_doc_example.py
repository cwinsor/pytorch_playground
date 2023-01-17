# torcy.nn.CrossEntropy
# from https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html


import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy

def test_01():

    # # Example of target with class indices
    # loss = nn.CrossEntropyLoss()
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.empty(3, dtype=torch.long).random_(5)
    # output = loss(input, target)
    # print("---example 1---")
    # print("input\n{}".format(input))
    # print("target\n{}".format(target))
    # print("output {}".format(output))
    # output.backward()

    # # Example of target with class probabilities
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.randn(3, 5).softmax(dim=1)
    # output = loss(input, target)
    # print("---example 2---")
    # print("input\n{}".format(input))
    # print("target\n{}".format(target))
    # print("output {}".format(output))
    # output.backward()


    # Example of target with class indices
    pk = np.array([ 0.0001, 0.9998, 0.0001])
    qk = np.array([ 0.0001, 0.9998, 0.0001])

    input = torch.tensor(pk, requires_grad=True)
    target = torch.tensor(qk)
    loss = nn.CrossEntropyLoss(reduction='none')
    output = loss(input, target)
    print("pytorch output {}".format(output))

    output = entropy(pk, qk, base=2)
    print("scipy output {}".format(output))




if __name__ == "__main__":

    test_01()
