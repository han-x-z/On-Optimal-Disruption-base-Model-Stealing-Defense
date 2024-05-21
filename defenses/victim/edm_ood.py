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
from torch.multiprocessing import set_start_method, get_context



def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('policy', metavar='PI', type=str, help='Policy to use while training',
                        choices=['random', 'adaptive'])
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')
    parser.add_argument('defense', metavar='TYPE', type=str, help='Type of defense to use',
                        choices=knockoff_utils.BBOX_CHOICES, default='none')
    parser.add_argument('defense_args', metavar='STR', type=str, help='Blackbox arguments in format "k1:v1,k2:v2,..."')

    parser.add_argument('--defense_aware',type=int,help="Whether using defense-aware attack",default = 0)
    parser.add_argument('--recover_args',type=str,help='Recover arguments in format "k1:v1,k2:v2,..."')
    parser.add_argument('--hardlabel',type=int,help="Whether only use hard label for extraction",default= 0)

    parser.add_argument('--quantize',type=int,help="Whether using quantized defense",default=0)
    parser.add_argument('--quantize_args',type=str,help='Quantization arguments in format "k1:v1,k2:v2,..."')

    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--budget', metavar='N', type=int, help='# images',
                        default=None)
    parser.add_argument('--nqueries', metavar='N', type=int, help='# queries to blackbox using budget images',
                        default=None)
    parser.add_argument('--qpi', metavar='N', type=int, help='# queries per image',
                        default=1)
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=1)
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--train_transform', type=int, help='Whether to perform data augmentation when querying', default=0)
    parser.add_argument('--train_transform_blur', type=int, help='Whether to perform data Gaussian Blur augmentation', default=0)
    parser.add_argument('--only_recovery', action='store_true', help='Perform prediction recovery only', default=False)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    knockoff_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
    device = torch.device('cuda' if args.device_id >= 0 else 'cpu')
    # ----------- Initialize blackbox
    blackbox_dir = params['victim_model_dir']
    defense_type = params['defense']
    # Initialize blackbox
    blackbox = EDM_device.from_modeldir(args.victim_model_dir, device, output_type='probs', **knockoff_utils.parse_defense_kwargs(args.defense_args))

    # Optional: Initialize Quantizer
    if args.quantize and 'epsilon' in args.quantize_args:
        blackbox = incremental_kmeans(blackbox, **knockoff_utils.parse_defense_kwargs(args.quantize_args))

    # ----------- Set up queryset
    queryset_name = params['queryset']

    modelfamily = datasets.dataset_to_modelfamily[queryset_name]
    if params['train_transform']:
        print('=> Using data augmentation while querying')
        transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    else:
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    # ----------- Evaluate
    queryset = datasets.__dict__[queryset_name](train=True, transform=transform)
    print('=> Evaluating on {} ({} samples)'.format(queryset, len(queryset)))
    batch_size = params['batch_size']
    nworkers = params['nworkers']
    test_loader = DataLoader(queryset, num_workers=nworkers, shuffle=False, batch_size=batch_size)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            #print(inputs.shape)
            labels_list = []
            out_list = []
            for model in blackbox.model_list:
                with torch.no_grad():
                    out = model(inputs)
                    out = F.softmax(out,dim=1).detach()
                    labels = torch.argmax(out, dim=1)
                    labels_list.append(labels)
                    out_list.append(out)
            out_all = torch.stack(out_list, dim=0)
            # 检查OOD样本
            for i in range(inputs.shape[0]):
                unique_labels = torch.unique(torch.stack([labels[i] for labels in labels_list]))  # 获取唯一标签
                if len(unique_labels) > 1:  # 如果有多个唯一标签，它是OOD
                    blackbox.ood_count += 1
                    blackbox.ood_samples.append(x[i])
                    blackbox.queryset_ood_data_list.append(queryset.data[i])

                else:
                    self.id_samples.append(x[i])
                    blackbox.queryset_id_data_list.append(queryset.data[i])



if __name__ == '__main__':
    main()
