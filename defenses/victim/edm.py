import os.path
import os.path as osp
import json
import pickle
from typing import List
from defenses import datasets
import defenses.models.zoo as zoo
from defenses.victim import Blackbox
import torch
import torch.nn as nn
import torch.nn.functional as F
from defenses.victim.utils_EDM import get_model
cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

model_choices = {
    'mnist':('lenet','conv3', 'kmnist'),
    'cifar10':('vgg16_bn', 'conv3', 'svhn'),
    'gtsrb':('vgg16_bn', 'conv3', 'svhn'),
    'imagenette':('resnet34', 'conv3', 'flowers17')
}

class EDM_device(Blackbox):
    def __init__(
        self,
        #root = '',
        model,
        task,
        num_classes,
        model_dir,
        *args, **kwargs
    ):
        super().__init__(model=model, *args, **kwargs)
        #self.root = root
        self.model_dir = model_dir
        self.call_count=0
        self.dataset_tar = task
        self.model_tar, self.arch_hash, self.hash_ds = model_choices[task]
        self.num_classes = num_classes
        self.device = torch.device('cuda')
        self.model_list = self.load_models()
        self.top1_preserve = False
        self.n_models = len(self.model_list)
        for model in self.model_list:
            model = model.to(self.device)
            model.eval()
        self.bounds = [-1, 1]
        self.n_queries = 0
        self.model_hash = self.load_hash_model()
        self.hash_list = []
        self.ood_count = 0
        self.ood_samples = []
        self.id_samples = []
        self.merged_queryset_list = []
        self.queryset_ood_data_list = []
        self.queryset_id_data_list = []
        if self.model_hash is not None:
            self.model_hash = self.model_hash.to(self.device)

    @classmethod
    def from_modeldir(cls, model_dir, device, output_type='probs', **kwargs):
        device = torch.device('cuda') if device is None else device
        # Load parameters from JSON configuration
        param_path = osp.join(model_dir, 'params.json')
        with open(param_path, 'r') as f:
            params = json.load(f)

        # Instantiate the model based on the architecture specified in the JSON
        model_arch = params['model_arch']
        num_classes = params['num_classes']
        if 'queryset' in params:
            dataset_name = params['queryset']
        elif 'testdataset' in params:
            dataset_name = params['testdataset']
        elif 'dataset' in params:
            dataset_name = params['dataset']
        modelfamily = datasets.dataset_to_modelfamily[dataset_name]
        model = zoo.get_net(model_arch, modelfamily, num_classes=num_classes)
        # Load model weights
        # Load weights
        checkpoint_path = osp.join(model_dir, 'checkpoint.pth.tar')
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))
        blackbox = cls(model=model, output_type=output_type, dataset_name=dataset_name,
                       modelfamily=modelfamily, model_arch=model_arch, num_classes=num_classes, model_dir=model_dir, device=device,
                       **kwargs)
        print("Blackbox model initialized with the following parameters:")
        print(f"Model architecture: {model_arch}, Model family: {modelfamily}, Num classes: {num_classes}, Dataset name: {dataset_name}")
        return blackbox

    def load_models(self) -> List:
        T_list = []
        path_exp = os.path.join(self.model_dir, f'edm')

        #print(f"model_dir:{self.model_dir}")
        #print(path_exp)
        for i in range(5):
            T_path = os.path.join(path_exp, 'T' + str(i) + '.pt')
            print("**",self.dataset_tar)
            T = get_model(self.model_tar, self.dataset_tar).to(self.device)
            T.load_state_dict(torch.load(T_path))
            T_list.append(T)

        return T_list

    def load_hash_model(self):
        path_hash = os.path.join(self.model_dir, f'hash.pt')
        #print(path_hash)
        H = get_model(self.arch_hash, self.hash_ds)
        H.load_state_dict(torch.load(path_hash))
        return H

    def eval(self):
        for model in self.model_list:
            model.eval()

    def coherence(self, x):
        pred_list = []
        cs_list = []
        for model in self.model_list:
            pred = F.softmax(model(x), dim=-1)
            pred_list.append(pred)
        pred_list_batch = torch.stack(pred_list, dim=0)  # n x batch x 10

        for i, pred_i in enumerate(pred_list_batch):
            for j in range(i + 1, len(self.model_list)):
                pred_j = pred_list_batch[j]
                cs = cosine_similarity(pred_i, pred_j)  # batch x 10
                cs_list.append(cs.detach())

        cs_batch = torch.max(torch.stack(cs_list, dim=0), dim=0)[0]
        return cs_batch

    def to(self, device: str):
        for model in self.model_list:
            model = model.to(device)
        return self

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.round(x * 128.0) / 128.0
        return x

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        """ clamp """
        x = torch.clamp(x, self.bounds[0], self.bounds[1])
        return x


    def get_hash_list(self, x: torch.Tensor) -> List[int]:
        y = F.softmax(self.model_hash(x), dim=-1)
        y_class = torch.argmax(y, dim=-1)
        M = 10
        m = 2
        y_hash = (
            (m * y_class) + torch.floor((M * torch.max(y, dim=-1)[0] - 1) * m / (M - 1))
        ) % self.n_models
        return y_hash.detach().cpu().numpy().tolist()


    def update_stats(self, hash_list):
        self.hash_list += hash_list


    def detect_ood_samples(self, x):
        labels_list = []
        out_list = []
        for model in self.model_list:
            with torch.no_grad():
                out = model(x)

                out = F.softmax(out,dim=1).detach()
                labels = torch.argmax(out, dim=1)
                labels_list.append(labels)
                out_list.append(out)
        out_all = torch.stack(out_list, dim=0)
        # 检查OOD样本
        for i in range(x.shape[0]):  # 遍历每个样本
            unique_labels = torch.unique(torch.stack([labels[i] for labels in labels_list]))  # 获取唯一标签
            if len(unique_labels) > 1:  # 如果有多个唯一标签，它是OOD
                self.ood_count += 1
                self.ood_samples.append(x[i])
            else:
                self.id_samples.append(x[i])

        return self.ood_samples

    def __call__(self, x: torch.Tensor, return_origin=False) -> torch.Tensor:
        self.n_queries += x.shape[0]
        self.call_count += x.shape[0]
        out_list = []
        labels_list = []
        for model in self.model_list:
            with torch.no_grad():
                out = model(x)
                out = F.softmax(out,dim=1).detach()
                labels = torch.argmax(out, dim=1)
                labels_list.append(labels)
                out_list.append(out)
        out_all = torch.stack(out_list, dim=0)
        hash_list = self.get_hash_list(x)
        out1 = out_all[hash_list, range(x.shape[0])]
        self.update_stats(hash_list)
        if return_origin:
            return out1, out
        else:
            return out1
    def get_n_queries(self) -> int:
        return self.n_queries

