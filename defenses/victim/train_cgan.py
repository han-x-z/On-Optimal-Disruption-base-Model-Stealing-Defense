#!/usr/bin/python
"""Code to evaluate the test accuracy of the (defended) blackbox model.
Note: The perturbation utility metric is logged by the blackbox in distance{test, transfer}.log.tsv
"""
import argparse
import os.path as osp
import os
import sys
import json
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


sys.path.append(os.getcwd())
from defenses import datasets
import defenses.utils.model as model_utils
import defenses.utils.utils as knockoff_utils
import defenses.config as cfg


from defenses.victim import *




def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('defense', metavar='TYPE', type=str, help='Type of defense to use',
                        choices=knockoff_utils.BBOX_CHOICES)
    parser.add_argument('defense_args', metavar='STR', type=str, help='Blackbox arguments in format "k1:v1,k2:v2,..."')
    parser.add_argument('--quantize', type=int, help="Whether using quantized defense", default=0)
    parser.add_argument('--quantize_args', type=str, help='Quantization arguments in format "k1:v1,k2:v2,..."')
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=64)
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs for CGAN')
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    knockoff_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
    device = torch.device('cuda' if args.device_id >= 0 else 'cpu')

    # Initialize blackbox
    blackbox = CGANDefense.from_modeldir(args.victim_model_dir, device, output_type='probs', **knockoff_utils.parse_defense_kwargs(args.defense_args))

    # Optional: Initialize Quantizer
    if args.quantize and 'epsilon' in args.quantize_args:
        blackbox = incremental_kmeans(blackbox, **knockoff_utils.parse_defense_kwargs(args.quantize_args))

    # Train CGAN
    #print('=> Starting CGAN training...')
    #blackbox.train_cgan(epochs=args.epochs)

    # Evaluate performance
    print('=> Evaluating performance...')
    blackbox.test()
    # Assuming test_step function exists and evaluates the model's performance
    #loss, acc, _ = model_utils.test_step(blackbox, blackbox.real_data_loader, nn.CrossEntropyLoss(), device)

    #print(f'Test Loss: {loss}, Test Accuracy: {acc}')

if __name__ == '__main__':
    main()
