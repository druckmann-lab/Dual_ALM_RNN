# Requires PyTorch v1.0

import argparse, os, math, pickle, json
from dual_alm_rnn_exp import DualALMRNNExp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--generate_dataset', action='store_true', default=False)

    parser.add_argument('--train_type_uniform', action='store_true', default=False)

    parser.add_argument('--train_type_modular', action='store_true', default=False)

    parser.add_argument('--plot_cd_traces', action='store_true', default=False)


    return parser.parse_args()

def main():

    args = parse_args()


    if args.generate_dataset:
        exp = DualALMRNNExp()
        exp.generate_dataset()

    if args.train_type_uniform:
        exp = DualALMRNNExp()
        exp.train_type_uniform()

    if args.train_type_modular:
        exp = DualALMRNNExp()
        exp.train_type_modular()

    if args.plot_cd_traces:
        exp = DualALMRNNExp()
        exp.plot_cd_traces()



if __name__ == '__main__':
    main()

