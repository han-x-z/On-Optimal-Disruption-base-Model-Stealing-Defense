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
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def create_real_data_loader(dataset_name, modelfamily, batch_size, num_workers, use_subset=None):

    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']


    dataset = datasets.__dict__[dataset_name]
    trainset = dataset(train=True, transform=train_transform, download=True)
    testset = dataset(train=False, transform=test_transform, download=True)


    if use_subset is not None:
        #idxs = np.arange(len(trainset))
        #idxs = np.random.choice(idxs, size=use_subset, replace=False)
        #trainset = Subset(trainset, idxs)
        idxs = np.arange(len(testset))
        idxs = np.random.choice(idxs, size=use_subset, replace=False)
        testset = Subset(testset, idxs)
    # 创建 DataLoader
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return test_loader


class Generator(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=50):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim + embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, noise, labels):
        labels = self.label_embedding(labels)
        x = torch.cat([noise, labels], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)


        self.model = nn.Sequential(
            nn.Linear(num_classes * 2 + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, real, fake, labels):

        label_embedding = self.label_embedding(labels)


        x = torch.cat([real, fake, label_embedding], dim=1)
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
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.call_count=0
        self.criterion = nn.BCELoss()
        self.real_data_loader = create_real_data_loader(self.dataset_name, self.modelfamily, batch_size = 32, num_workers = 4, use_subset=None)
        self.load_models()



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
        print("Blackbox model initialized with the following parameters:")
        print(f"Model architecture: {model_arch}, Model family: {modelfamily}, Num classes: {num_classes}, Dataset name: {dataset_name}")
        return blackbox

    def load_models(self):

        generator_file_name = f'{self.dataset_name}_generator.pth'
        discriminator_file_name = f'{self.dataset_name}_discriminator.pth'


        generator_path = os.path.join(self.model_dir, generator_file_name)
        discriminator_path = os.path.join(self.model_dir, discriminator_file_name)

        if os.path.exists(generator_path) and os.path.exists(discriminator_path):
            self.generator.load_state_dict(torch.load(generator_path))
            self.discriminator.load_state_dict(torch.load(discriminator_path))
            self.generator.eval()
            self.discriminator.eval()
            self.training_complete = True
            #print("Loaded trained models from:", generator_file_name, "and", discriminator_file_name)
        else:
            self.training_complete = False
            print("No trained models found for the dataset:", self.dataset_name, ". Need training.")


    def save_models(self):

        generator_file_name = f'{self.dataset_name}_generator.pth'
        discriminator_file_name = f'{self.dataset_name}_discriminator.pth'


        torch.save(self.generator.state_dict(), os.path.join(self.model_dir, generator_file_name))
        torch.save(self.discriminator.state_dict(), os.path.join(self.model_dir, discriminator_file_name))
        print(f"Generator model saved as {generator_file_name}")
        print(f"Discriminator model saved as {discriminator_file_name}")


    def train_cgan(self, epochs=50):

        gen_scheduler = StepLR(self.gen_optimizer, step_size=10, gamma=0.1)
        dis_scheduler = StepLR(self.dis_optimizer, step_size=10, gamma=0.1)
        if not self.training_complete:
            print("Starting CGAN training...")
            print(f"real_data_loader: {self.real_data_loader}")
            for epoch in range(epochs):
                label_matches = []
                for real_data, _ in self.real_data_loader:
                    real_data = real_data.to(self.device)
                    batch_size = real_data.shape[0]
                    real_probs = self.model(real_data)
                    real_probs = F.softmax(real_probs, dim=1)
                    predicted_labels = torch.argmax(real_probs, dim=1)
                    # Train Discriminator
                    noise = torch.randn(batch_size, 100, device=self.device)
                    fake_data = self.generator(noise, predicted_labels)
                    #print(fake_data)

                    self.dis_optimizer.zero_grad()
                    d_real = self.discriminator(real_probs.detach(), fake_data, predicted_labels)
                    d_fake = self.discriminator(fake_data, real_probs.detach(), predicted_labels)
                    d_loss = (self.criterion(d_real, torch.ones_like(d_real)) + self.criterion(d_fake, torch.zeros_like(d_fake)) ) / 2


                    d_loss.backward(retain_graph=True)
                    self.dis_optimizer.step()

                    # Train Generator
                    self.gen_optimizer.zero_grad()
                    g_loss = self.criterion(self.discriminator(fake_data, real_probs.detach(), predicted_labels), torch.ones_like(d_fake))
                    g_loss.backward()
                    self.gen_optimizer.step()


                    # Evaluate label matching
                    fake_class = torch.argmax(fake_data, dim=1)
                    matches = (fake_class == predicted_labels).float().sum().item()
                    label_matches.append(matches / batch_size)
                gen_scheduler.step()
                dis_scheduler.step()
                avg_match = sum(label_matches) / len(label_matches)
                print(f'Epoch {epoch + 1}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, Avg Label Match: {avg_match:.2f}')
                print(f'Real data example from Epoch {epoch + 1}: {predicted_labels} {real_probs}')
                print(f'Fake data example from Epoch {epoch + 1}: {fake_class} {fake_data}')
            self.save_models()
            self.training_complete = True

    def test(self):
        self.load_models()
        if not self.training_complete:
            print("Model training not completed. Please train the model first.")
            return
        all_real_probs = []
        all_generated_probs = []
        all_labels = []
        for real_data, labels in create_real_data_loader(self.dataset_name, self.modelfamily, batch_size = 32, num_workers = 4, use_subset=500):
            real_data, labels = real_data.to(self.device), labels.to(self.device)
            batch_size = real_data.shape[0]
            #print(batch_size)

            noise = torch.randn(batch_size, 100, device=self.device)
            with torch.no_grad():

                real_probs = F.softmax(self.model(real_data), dim=1)
                predicted_labels = torch.argmax(real_probs, dim=1)
                generated_probs = self.generator(noise, predicted_labels)
            all_real_probs.append(real_probs.cpu().numpy())
            all_generated_probs.append(generated_probs.cpu().numpy())
            all_labels.append(predicted_labels.cpu().numpy())



        all_real_probs = np.concatenate(all_real_probs, axis=0)
        all_generated_probs = np.concatenate(all_generated_probs, axis=0)
        all_labels = np.concatenate(all_labels)


        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')


        combined_probs = np.concatenate([all_real_probs, all_generated_probs])
        tsne_results = tsne.fit_transform(combined_probs)


        real_tsne_result = tsne_results[:len(all_real_probs)]
        generated_tsne_result = tsne_results[len(all_real_probs):]


        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(real_tsne_result[:, 0], real_tsne_result[:, 1], c='blue', alpha=0.5, label='Real Data')
        ax.scatter(generated_tsne_result[:, 0], generated_tsne_result[:, 1], c='red', alpha=0.5, label='Generated Data')
        ax.legend()
        plt.title('t-SNE of Real vs Generated Data Class Distributions')
        plt.savefig('tsne_real_vs_generated.png')  # Make sure to use the correct path for saving
        #plt.show()







    def __call__(self, x, return_origin=False):
        self.load_models()
        if not self.training_complete:
            print("Training required. Training now...")
            self.train_cgan(epochs=50)

        with torch.no_grad():
            self.call_count += x.shape[0]
            x = x.to(self.device)
            batch_size = x.size(0)
            z_v = self.model(x)
            y_v = F.softmax(z_v, dim=1)
            noise = torch.randn(batch_size, 100, device=self.device)
            #labels = torch.argmax(real_probs, dim=1)
            random_labels = torch.randint(0, self.num_classes, (batch_size,), device=noise.device)
            fake_probs = self.generator(noise, random_labels)
            generator_ratio = 1.0     #TODO
            fake_count = int(batch_size * generator_ratio)
            real_count = batch_size - fake_count
            fake_probs = torch.cat((y_v[:real_count], fake_probs[:fake_count]), dim=0)

            #print(random_labels[0])
            #print(fake_probs[0])
        if return_origin:
            return fake_probs, y_v
        else:
            return fake_probs

