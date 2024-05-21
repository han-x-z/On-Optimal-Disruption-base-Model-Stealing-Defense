import os.path as osp
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import defenses.models.zoo as zoo
from defenses import datasets
from defenses.victim import Blackbox
import pickle
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

def create_real_data_loader(dataset_name, modelfamily, batch_size, num_workers, use_subset=None):
    # 根据模型家族设置数据转换
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']

    # 创建数据集实例
    dataset = datasets.__dict__[dataset_name]
    trainset = dataset(train=True, transform=train_transform, download=True)
    testset = dataset(train=False, transform=test_transform, download=True)

    # 如果需要，只使用训练集的一个子集
    if use_subset is not None:
        idxs = np.arange(len(trainset))
        idxs = np.random.choice(idxs, size=use_subset, replace=False)
        trainset = Subset(trainset, idxs)

    # 创建 DataLoader
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader


class Generator(nn.Module):

    def __init__(self, latent_dim, num_classes):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),  # 将噪音维度加入生成器输入维度
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, noise, labels):
        gen_input = self.label_embed(labels)
        print(gen_input.requires_grad)
        gen_input_with_noise = torch.cat((gen_input, noise), -1)  # 将噪音和标签嵌入向量连接起来
        print(gen_input_with_noise.requires_grad)
        class_probs = self.generator(gen_input_with_noise)
        print(class_probs.requires_grad)
        return class_probs



class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        # 调整输入层维度为 num_classes * 2，因为我们的输入是两组概率分布
        self.model = nn.Sequential(
            nn.Linear(num_classes * 2, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # 添加 Dropout 层减少过拟合
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, real, fake):
        # 将真实数据和生成数据串联作为输入
        x = torch.cat([real, fake], dim=1)
        return self.model(x)



class CGANDefense(Blackbox):
    def __init__(self, model, num_classes, device, model_dir, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.device = device
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.model = model.to(device)
        self.generator = Generator(100, num_classes).to(device)
        self.discriminator = Discriminator(num_classes).to(device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.real_data_loader = create_real_data_loader(self.dataset_name, self.modelfamily, batch_size = 64, num_workers = 4, use_subset=None)
        self.load_models()  # 尝试加载模型



    @classmethod
    def from_modeldir(cls, model_dir, device=None, output_type='probs', **kwargs):
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
        return blackbox

    def load_models(self):
        generator_path = os.path.join(self.model_dir, 'generator.pth')
        discriminator_path = os.path.join(self.model_dir, 'discriminator.pth')

        if os.path.exists(generator_path) and os.path.exists(discriminator_path):
            self.generator.load_state_dict(torch.load(generator_path))
            self.discriminator.load_state_dict(torch.load(discriminator_path))
            self.training_complete = True
            print("Loaded trained models.")
        else:
            self.training_complete = False
            print("No trained models found, need training.")

    def save_models(self):
        torch.save(self.generator.state_dict(), os.path.join(self.model_dir, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(self.model_dir, 'discriminator.pth'))
        print("Models saved.")

    # 在鉴别器处理前后打印梯度状态
    def train_cgan(self, epochs=50):
        if not self.training_complete:
            print("Starting CGAN training...")
            for epoch in range(epochs):
                for real_data, labels in self.real_data_loader:
                    real_data, labels = real_data.to(self.device), labels.to(self.device)
                    batch_size = real_data.shape[0]
                    real_probs = F.softmax(self.model(real_data), dim=1)

                    # Train Discriminator
                    noise = torch.randn(batch_size, 100, device=self.device, requires_grad=True)
                    gen_labels = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)
                    fake_data = self.generator(noise, gen_labels)
                    print(f"Fake data requires_grad: {fake_data.requires_grad}")  # 确保 fake_data 有梯度
                    self.dis_optimizer.zero_grad()
                    d_real = self.discriminator(real_probs.detach(), fake_data)
                    d_fake = self.discriminator(fake_data, real_probs.detach())
                    d_real_loss = self.criterion(d_real, torch.ones_like(d_real))
                    d_fake_loss = self.criterion(d_fake, torch.zeros_like(d_fake))
                    d_loss = d_real_loss + d_fake_loss
                    print(f"D Loss requires_grad: {d_loss.requires_grad}")  # 检查 D Loss 梯度状态

                    d_loss.backward()  # 检查反向传播是否引起错误
                    self.dis_optimizer.step()

                    # Train Generator
                    self.gen_optimizer.zero_grad()
                    g_loss = self.criterion(self.discriminator(fake_data, real_probs.detach()), torch.ones_like(d_real))
                    g_loss.backward()
                    self.gen_optimizer.step()

                print(f'Epoch {epoch + 1}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

            self.save_models()  # 保存模型
            self.training_complete = True

    def __call__(self, x):
        if not self.training_complete:
            print("Training required. Training now...")
            self.train_cgan(epochs=50)

        with torch.no_grad():
            x = x.to(self.device)
            real_probs = self.model(x)
            noise = torch.randn(x.shape[0], 100, device=self.device)
            random_labels = torch.randint(0, num_classes, (batch_size,), device=noise.device)
            manipulated_probs = self.generator(noise, random_labels)
            return manipulated_probs
