from collections import defaultdict
import csv
import pickle
import torch
import torch.utils.data

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

from models.resnet import ResNet18, SupConResNet18, ResNet34, SupConResNet34, ResNet50, SupConResNet50
from models.trigger_generator_vae import Generator
from torchvision.utils import save_image

from models.word_model import RNNModel
from utils.text_load import *
from utils.utils import SubsetSampler
import random

logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0
class ImageHelper(Helper):


    def poison(self):
        return

    def create_model(self):
        if self.params['dataset']=='cifar10':
            generator = Generator(helper=self).cuda()
            if self.params['model_type']=='ResNet18':
                local_model = ResNet18(name='Local',
                    created_time=self.params['current_time'])
                local_model.cuda()
                target_model = ResNet18(name='Target',
                    created_time=self.params['current_time'])
                target_model.cuda()
                replace_model = ResNet18(name='Local',
                    created_time=self.params['current_time'])
        
                contrastive_model = SupConResNet18(name='Contrastive',
                    created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet34':
                local_model = ResNet34(name='Local',
                    created_time=self.params['current_time'])
                local_model.cuda()
                target_model = ResNet34(name='Target',
                    created_time=self.params['current_time'])
                target_model.cuda()
                replace_model = ResNet34(name='Local',
                    created_time=self.params['current_time'])
        
                contrastive_model = SupConResNet34(name='Contrastive',
                    created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet50':
                local_model = ResNet50(name='Local',
                    created_time=self.params['current_time'])
                local_model.cuda()
                target_model = ResNet50(name='Target',
                    created_time=self.params['current_time'])
                target_model.cuda()
                replace_model = ResNet50(name='Local',
                    created_time=self.params['current_time'])
        
                contrastive_model = SupConResNet50(name='Contrastive',
                    created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()

        elif self.params['dataset']=='cifar100':
            generator = Generator(helper=self)
            if self.params['model_type']=='ResNet18':
                local_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                local_model.cuda()
                target_model = ResNet18(name='Target',
                            created_time=self.params['current_time'], num_classes=100)
                target_model.cuda()
                replace_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                contrastive_model = SupConResNet18(name='Contrastive',
                            created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet34':
                local_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                local_model.cuda()
                target_model = ResNet34(name='Target',
                            created_time=self.params['current_time'], num_classes=100)
                target_model.cuda()
                replace_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                contrastive_model = SupConResNet34(name='Contrastive',
                            created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet50':
                local_model = ResNet50(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                local_model.cuda()
                target_model = ResNet50(name='Target',
                            created_time=self.params['current_time'], num_classes=100)
                target_model.cuda()
                replace_model = ResNet50(name='Local',
                            created_time=self.params['current_time'], num_classes=100)
                contrastive_model = SupConResNet50(name='Contrastive',
                            created_time=self.params['current_time'] )
                replace_model.cuda()
                contrastive_model.cuda()
        
        elif self.params['dataset']=='emnist':
            generator = Generator(helper=self).cuda()
            if self.params['model_type']=='ResNet18':
                local_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                local_model.cuda()
                target_model = ResNet18(name='Target',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                target_model.cuda()
                replace_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                contrastive_model = SupConResNet18(name='Contrastive',
                                created_time=self.params['current_time'], dataset='emnist')
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet34':
                local_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                local_model.cuda()
                target_model = ResNet34(name='Target',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                target_model.cuda()
                replace_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=10, dataset='emnist')
                contrastive_model = SupConResNet34(name='Contrastive',
                                created_time=self.params['current_time'], dataset='emnist')
                replace_model.cuda()
                contrastive_model.cuda()
        elif self.params['dataset']=='gtsrb':
            generator = Generator(helper=self).cuda()
            if self.params['model_type']=='ResNet18':
                local_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=43, dataset='gtsrb')
                local_model.cuda()
                target_model = ResNet18(name='Target',
                            created_time=self.params['current_time'], num_classes=43, dataset='gtsrb')
                target_model.cuda()
                replace_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=43, dataset='gtsrb')
                contrastive_model = SupConResNet18(name='Contrastive',
                                created_time=self.params['current_time'], dataset='gtsrb')
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet34':
                local_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=43, dataset='gtsrb')
                local_model.cuda()
                target_model = ResNet34(name='Target',
                            created_time=self.params['current_time'], num_classes=43, dataset='gtsrb')
                target_model.cuda()
                replace_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=43, dataset='gtsrb')
                contrastive_model = SupConResNet34(name='Contrastive',
                                created_time=self.params['current_time'], dataset='gtsrb')
                replace_model.cuda()
                contrastive_model.cuda()
        elif self.params['dataset']=='BelgiumTS':
            generator = Generator(helper=self).cuda()
            if self.params['model_type']=='ResNet18':
                local_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=62, dataset='BelgiumTS')
                local_model.cuda()
                target_model = ResNet18(name='Target',
                            created_time=self.params['current_time'], num_classes=62, dataset='BelgiumTS')
                target_model.cuda()
                replace_model = ResNet18(name='Local',
                            created_time=self.params['current_time'], num_classes=62, dataset='BelgiumTS')
                contrastive_model = SupConResNet18(name='Contrastive',
                                created_time=self.params['current_time'], dataset='BelgiumTS')
                replace_model.cuda()
                contrastive_model.cuda()
            elif self.params['model_type']=='ResNet34':
                local_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=62, dataset='BelgiumTS')
                local_model.cuda()
                target_model = ResNet34(name='Target',
                            created_time=self.params['current_time'], num_classes=62, dataset='BelgiumTS')
                target_model.cuda()
                replace_model = ResNet34(name='Local',
                            created_time=self.params['current_time'], num_classes=62, dataset='BelgiumTS')
                contrastive_model = SupConResNet34(name='Contrastive',
                                created_time=self.params['current_time'], dataset='BelgiumTS')
                replace_model.cuda()
                contrastive_model.cuda()

        if self.params['resumed_model']:
            loaded_params = torch.load(f"saved_models/{self.params['resumed_model_path']}")
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch'] + 1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model
        self.replace_model = replace_model
        self.contrastive_model = contrastive_model
        self.generator = generator

        
        self.visual = True
        self.count = 0

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                if self.params['semantic_backdoor']:
                    continue
            
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
   
    def sample_dirichlet_train_data_without_8(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label == 8:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        class_size = len(cifar_classes[1])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes+1):
            if n!=8:
                random.shuffle(cifar_classes[n])
                sampled_probabilities = class_size * np.random.dirichlet(
                    np.array(no_participants * [alpha]))
                for user in range(no_participants):
                    no_imgs = int(round(sampled_probabilities[user]))
                    sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                    per_participant_list[user].extend(sampled_list)
                    cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def sample_dirichlet_train_data_without_2(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label == 2:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes+1):
            if n != 2:
                random.shuffle(cifar_classes[n])
                sampled_probabilities = class_size * np.random.dirichlet(
                    np.array(no_participants * [alpha]))
                for user in range(no_participants):
                    no_imgs = int(round(sampled_probabilities[user]))
                    sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                    per_participant_list[user].extend(sampled_list)
                    cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def sample_dirichlet_train_data_without_1(self, no_participants, alpha=0.9):
        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label == 1:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes+1):
            if n != 1:
                random.shuffle(cifar_classes[n])
                sampled_probabilities = class_size * np.random.dirichlet(
                    np.array(no_participants * [alpha]))
                for user in range(no_participants):
                    no_imgs = int(round(sampled_probabilities[user]))
                    sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                    per_participant_list[user].extend(sampled_list)
                    cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
    
    def add_trigger(self, data, pattern_diffusion=0):
        
        new_data = np.copy(data)
        channels, height, width = new_data.shape
        if self.params['pattern_type'] == 1:
            print('image helper 1')
            for c in range(channels):
                if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100' or self.params['dataset'] == 'gtsrb':
                    new_data[c, height-3, width-3] = 255
                    new_data[c, height-2, width-4] = 255
                    new_data[c, height-4, width-2] = 255
                    new_data[c, height-2, width-2] = 255
                elif self.params['dataset'] == 'emnist' or self.params['dataset'] == 'BelgiumTS':
                    new_data[c, height-3, width-3] = 1
                    new_data[c, height-2, width-4] = 1
                    new_data[c, height-4, width-2] = 1
                    new_data[c, height-2, width-2] = 1
        
        elif self.params['pattern_type'] == 2:
            change_range = 4
            
            if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100' or self.params['dataset'] == 'gtsrb':
                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(6+diffusion), width-(6+diffusion)] = 255
                new_data[1, height-(6+diffusion), width-(6+diffusion)] = 255
                new_data[2, height-(6+diffusion), width-(6+diffusion)] = 255

                diffusion = 0
                new_data[0, height-(5+diffusion), width-(5+diffusion)] = 255
                new_data[1, height-(5+diffusion), width-(5+diffusion)] = 255
                new_data[2, height-(5+diffusion), width-(5+diffusion)] = 255

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(4-diffusion), width-(6+diffusion)] = 255
                new_data[1, height-(4-diffusion), width-(6+diffusion)] = 255
                new_data[2, height-(4-diffusion), width-(6+diffusion)] = 255

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(6+diffusion), width-(4-diffusion)] = 255
                new_data[1, height-(6+diffusion), width-(4-diffusion)] = 255
                new_data[2, height-(6+diffusion), width-(4-diffusion)] = 255

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(4-diffusion), width-(4-diffusion)] = 255
                new_data[1, height-(4-diffusion), width-(4-diffusion)] = 255
                new_data[2, height-(4-diffusion), width-(4-diffusion)] = 255
            elif self.params['dataset'] == 'emnist' or self.params['dataset'] == 'BelgiumTS':
                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(6+diffusion), width-(6+diffusion)] = 1

                diffusion = 0
                new_data[0, height-(5+diffusion), width-(5+diffusion)] = 1

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(4-diffusion), width-(6+diffusion)] = 1

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(6+diffusion), width-(4-diffusion)] = 1

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, height-(4-diffusion), width-(4-diffusion)] = 1
        elif self.params['pattern_type'] == 3:
            
            x = np.random.randint(6, height)
            y = np.random.randint(6, width)
            
            change_range = 4
            if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100' or self.params['dataset'] == 'gtsrb':
                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, x - (6 + diffusion), y - (6 + diffusion)] = 255
                new_data[1, x - (6 + diffusion), y - (6 + diffusion)] = 255
                new_data[2, x - (6 + diffusion), y - (6 + diffusion)] = 255

                diffusion = 0
                new_data[0, x - (5 + diffusion), y - (5 + diffusion)] = 255
                new_data[1, x - (5 + diffusion), y - (5 + diffusion)] = 255
                new_data[2, x - (5 + diffusion), y - (5 + diffusion)] = 255

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, x - (4 - diffusion), y - (6 + diffusion)] = 255
                new_data[1, x - (4 - diffusion), y - (6 + diffusion)] = 255
                new_data[2, x - (4 - diffusion), y - (6 + diffusion)] = 255

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, x - (6 + diffusion), y - (4 - diffusion)] = 255
                new_data[1, x - (6 + diffusion), y - (4 - diffusion)] = 255
                new_data[2, x - (6 + diffusion), y - (4 - diffusion)] = 255

                diffusion = int(random.random() * pattern_diffusion * change_range)
                new_data[0, x - (4 - diffusion), y - (4 - diffusion)] = 255
                new_data[1, x - (4 - diffusion), y - (4 - diffusion)] = 255
                new_data[2, x - (4 - diffusion), y - (4 - diffusion)] = 255
        elif self.params['pattern_type'] == 4:
            x = np.random.randint(4, height)
            y = np.random.randint(4, width)
            for c in range(channels):
                if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100' or self.params['dataset'] == 'gtsrb':
                    new_data[c, x - 3, y - 3] = 255
                    new_data[c, x - 2, y - 4] = 255
                    new_data[c, x - 4, y - 2] = 255
                    new_data[c, x - 2, y - 2] = 255
                elif self.params['dataset'] == 'emnist' or self.params['dataset'] == 'BelgiumTS':
                    new_data[c, x - 3, y - 3] = 1
                    new_data[c, x - 2, y - 4] = 1
                    new_data[c, x - 4, y - 2] = 1
                    new_data[c, x - 2, y - 2] = 1
        elif self.params['pattern_type'] == 5:
            
            trigger_size = self.params['trigger_size']
            pixel_num = self.params['pixel_num']
           
            z = np.random.choice(trigger_size * trigger_size, pixel_num, replace=False)
            x = []
            y = []
            for i in range(pixel_num):
                x.append(z[i] // trigger_size+1)
                y.append(z[i] % trigger_size+1)
            for i in range(pixel_num):
                if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100' or self.params['dataset'] == 'gtsrb':
                    new_data[0, height-x[i], width-y[i]] = 255
                    new_data[1, height-x[i], width-y[i]] = 255
                    new_data[2, height-x[i], width-y[i]] = 255
                elif self.params['dataset'] == 'emnist' or self.params['dataset'] == 'BelgiumTS':
                    new_data[0, height-x[i], width-y[i]] = 1
                    new_data[1, height-x[i], width-y[i]] = 1
                    new_data[2, height-x[i], width-y[i]] = 1
            
        elif self.params['pattern_type'] == 6:
            
            for c in range(channels):
                if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100' or self.params['dataset'] == 'gtsrb':
                    new_data[c, height-3, width-3] = 255
                    new_data[c, height-2, width-4] = 255
                    new_data[c, height-4, width-2] = 255
                    new_data[c, height-2, width-2] = 255
                    
                elif self.params['dataset'] == 'emnist' or self.params['dataset'] == 'BelgiumTS':
                    new_data[c, height-3, width-3] = 1
                    new_data[c, height-2, width-4] = 1
                    new_data[c, height-4, width-2] = 1
                    new_data[c, height-2, width-2] = 1
        return torch.Tensor(new_data)

   

    def add_dynamic_trigger(self, data, trigger, pattern_diffusion=0):
        
        channels, height, width = data.size()
        trigger_size = self.params['trigger_size']
        new_data = torch.tensor(data).cuda()
        
        all_mask = torch.ones_like(new_data[0, :, :])
        if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100' or self.params['dataset'] == 'gtsrb':
            mask = torch.ones_like(trigger)-(trigger/255)
        else:
            mask = torch.ones_like(trigger)-(trigger)
        
        all_mask[-trigger_size:, -trigger_size:] = mask
        new_data[0, :, :] = new_data[0, :, :] * all_mask
        if channels == 3:
            new_data[1, :, :] = new_data[1, :, :] * all_mask
            new_data[2, :, :] = new_data[2, :, :] * all_mask
        new_data[:, -trigger_size:, -trigger_size:] += trigger
        
        return new_data

    
    def visual_posion_data(self, dynamic, origin):
        if self.params['dataset'] == "cifar10":
            std = [0.247, 0.243, 0.261]
            mean = [0.4914, 0.4822, 0.4465]
        tensor = origin.clone()
        for t, m, s in zip(range(tensor.size(0)), mean, std):
            tensor[t] = (tensor[t]).mul_(s).add_(m)
        origin = torch.clamp(tensor, 0, 1)

        tensor = dynamic.clone()
        for t, m, s in zip(range(tensor.size(0)), mean, std):
            tensor[t] = (tensor[t]).mul_(s).add_(m)
        dynamic = torch.clamp(tensor, 0, 1)

        self.count += 1
        figure_name = str(self.count)
        save_image(origin, 'saved_images/1/' + figure_name + '_origin.jpg')
        origin = origin.cpu()
        new_data = np.copy(origin)
        channels, height, width = new_data.shape
        for c in range(channels):
            new_data[c, height-3, width-3] = 255
            new_data[c, height-2, width-4] = 255
            new_data[c, height-4, width-2] = 255
            new_data[c, height-2, width-2] = 255
        save_image(torch.tensor(new_data), 'saved_images/1/' + figure_name + '_pixel.jpg')
        save_image(dynamic, 'saved_images/1/' + figure_name + '_dynamic.jpg')

    def label_dataset(self):
        label_inter_dataset_list = []
        label_fac_dataset_list = []
        pos = self.params['poison_images'][0]
        _,label_pos = self.train_dataset[pos]
        if self.params['edge_case'] and self.params['dataset']=='cifar10':
            self.inter_label = 0
        else:
            self.inter_label = label_pos

        self.fac_label = self.params['poison_label_swap']

        for ind,x in enumerate(self.train_dataset):
            _, label = x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            elif label == label_pos:
                label_inter_dataset_list.append(x)
            elif label == self.params['poison_label_swap']:
                label_fac_dataset_list.append(x)

        
        return label_inter_dataset_list, label_fac_dataset_list


    
    def visualize_dataset(self):
        indices = list()
        if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'emnist':
            range_no_id = list(range(50000))
        elif self.params['dataset'] == 'gtsrb':
            range_no_id = list(range(39209))
        elif self.params['dataset'] == 'BelgiumTS':
            range_no_id = list(range(4575))
        for image in self.params['poison_images'] + self.params['poison_images_test']:
            if image in range_no_id and self.params['semantic_backdoor']:
                range_no_id.remove(image)
        
        
        for batches in range(0, self.params['size_of_visualize_dataset']):
            range_iter = random.sample(range_no_id,
                                       self.params['batch_size'])
            indices.extend(range_iter)
            

        
        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def poison_dataset(self):
        indices = list()
        if self.params['dataset'] == 'cifar10' or self.params['dataset'] == 'emnist':
            range_no_id = list(range(50000))
        elif self.params['dataset'] == 'gtsrb':
            range_no_id = list(range(39209))
        elif self.params['dataset'] == 'BelgiumTS':
            range_no_id = list(range(4575))
        for image in self.params['poison_images'] + self.params['poison_images_test']:
            if image in range_no_id and self.params['semantic_backdoor']:
                range_no_id.remove(image)

        
        for batches in range(0, self.params['size_of_secret_dataset']):
            range_iter = random.sample(range_no_id,
                                       self.params['batch_size'])
            
            indices.extend(range_iter)
            
        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def poison_dataset_label_5(self):
        indices = list()
        poison_indices = list()
        for ind,x in enumerate(self.test_dataset):
            _,label = x
            if label == 5:
                poison_indices.append(ind)
        
        while len(indices)<self.params['size_of_secret_dataset_label_flip']:
            range_iter = random.sample(poison_indices,np.min([self.params['batch_size'], len(poison_indices) ]))
            indices.extend(range_iter)

        self.poison_images_ind = indices

        
        return torch.utils.data.DataLoader(self.test_dataset,
                               batch_size=self.params['batch_size'],
                               sampler=torch.utils.data.sampler.SubsetRandomSampler(self.poison_images_ind))

    def poison_test_dataset_label_5(self):

        return torch.utils.data.DataLoader(self.test_dataset,
                            batch_size=self.params['test_batch_size'],
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                self.poison_images_ind
                            ))


    def poison_test_dataset_with_9(self):
        indices = list()
        count = 0
        for ind, x in enumerate(self.train_dataset):
            _, label =  x
            if label == 9:
                count += 1
                indices.append(ind)
            if count == 1000:
                break
        
        return torch.utils.data.DataLoader(self.train_dataset,
                            batch_size=self.params['batch_size'],
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

        

    def poison_test_dataset(self):
        logger.info('get poison test loader')
        
        test_classes = {}
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_no_id = list(range(0, len(self.test_dataset)))
        count = 0
        for image_ind in test_classes[self.params['poison_label_swap']]:
            if image_ind in range_no_id:
                range_no_id.remove(image_ind)
                count += 1
        
        print("remove " + str(count) +" test data.")
        self.remove_label = count

        return torch.utils.data.DataLoader(self.test_dataset,
                           batch_size=self.params['batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                               range_no_id))
    
    def get_test_without_label_9(self):
        indices = list()
        for ind, x in enumerate(self.test_dataset):
            _, label = x
            if label == 9:
                continue
            indices.append(ind)
        
        return torch.utils.data.DataLoader(self.test_dataset,
                            batch_size=self.params['batch_size'],
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def load_edge_case(self):
        with open('./data/edge-case/southwest_images_new_train.pkl', 'rb') as train_f:
            saved_southwest_dataset_train = pickle.load(train_f)
        with open('./data/edge-case/southwest_images_new_test.pkl', 'rb') as test_f:
            saved_southwest_dataset_test = pickle.load(test_f)        

        return saved_southwest_dataset_train, saved_southwest_dataset_test

    def load_data(self):
        logger.info('Loading data')

        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_train_poison = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_train_grad = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        

        transform_test = transforms.Compose([
        
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_emnist = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])

        self.transform_poison = transform_train_poison
        self.transform_test = transform_test

        
        if self.params['dataset'] == 'cifar10':
            self.train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                         transform=transform_train)

            self.test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
        elif self.params['dataset'] == 'cifar100':
            self.train_dataset = datasets.CIFAR100('./data', train=True, download=True,
                                            transform=transform_train)

            self.test_dataset = datasets.CIFAR100('./data', train=False, transform=transform_test)
        elif self.params['dataset'] == 'emnist':
            self.train_dataset = datasets.EMNIST('./data', train=True, split="mnist", download=True,
                                            transform=transform_emnist)
            self.test_dataset = datasets.EMNIST('./data', train=False, split="mnist", transform=transform_emnist)
        elif self.params['dataset'] == 'gtsrb':
            self.train_dataset = GTSRB(
                True,
                transforms=transforms.Compose([
                        transforms.Resize((32, 32)), 
                        transforms.ToTensor()
                    ]),
                )
            self.test_dataset = GTSRB(
                False,
                transforms=transforms.Compose([
                        transforms.Resize((32, 32)), 
                        transforms.ToTensor()
                    ]),
                )
        elif self.params['dataset'] == 'BelgiumTS':
            self.train_dataset = BELGIUMTS(
                True,
                transforms=transforms.Compose([
                        transforms.Resize((32, 32)), 
                        transforms.ToTensor()
                    ]),
                )
            self.test_dataset = BELGIUMTS(
                False,
                transforms=transforms.Compose([
                        transforms.Resize((32, 32)), 
                        transforms.ToTensor()
                    ]),
                )
        
        if self.params['sampling_dirichlet']:
            
            
            indices_per_participant = self.sample_dirichlet_train_data(
                    self.params['number_of_total_participants'],
                    alpha=self.params['dirichlet_alpha'])
            
            
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
            benign_train_data = [self.get_train(indices) for pos, indices in
                             indices_per_participant.items()]
        else:
           
            all_range = list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]
        
        self.benign_train_data = benign_train_data
        self.train_data = train_loaders
       
        self.test_data = self.get_test()
        self.poisoned_data_for_train = self.poison_dataset()
        self.test_data_poison = self.poison_test_dataset()
        self.label_inter_dataset, self.label_fac_dataset = self.label_dataset()

        
        


    def get_train(self, indices):
       
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
        return train_loader

    def get_train_old(self, all_range, model_no):
        

        data_len = int(len(self.train_dataset) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               sub_indices))
        return train_loader


    def get_secret_loader(self):
        
        indices = list(range(len(self.train_dataset)))
        random.shuffle(indices)
        shuffled_indices = indices[:self.params['size_of_secret_dataset']]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=SubsetSampler(shuffled_indices))
        return train_loader

    def get_test(self):

        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)

        return test_loader


    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.cuda()
        target = target.cuda()
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target
    



class GTSRB(torch.utils.data.Dataset):
    def __init__(self, train, transforms, data_root=None, min_width=0):
        super(GTSRB, self).__init__()
        if data_root is None:
            data_root = "data"
        if train:
            self.data_folder = os.path.join(data_root, "GTSRB/Final_Training/Images")
            self.images, self.labels = self._get_data_train_list(min_width=min_width)
            if min_width > 0:
                print(f'Loading GTSRB Train greater than {min_width} width. Loaded {len(self.images)} images.')
        else:
            self.data_folder = os.path.join(data_root, "GTSRB/Final_Test/Images")
            self.images, self.labels = self._get_data_test_list(min_width)
            print(f'Loading GTSRB Test greater than {min_width} width. Loaded {len(self.images)} images.')

        self.transforms = transforms

    def _get_data_train_list(self, min_width=0):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                if int(row[1]) >= min_width:
                    images.append(prefix + row[0])
                    labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self, min_width=0):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            if int(row[1]) >= min_width: 
                images.append(self.data_folder + "/" + row[0])
                labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label
    

class BELGIUMTS(torch.utils.data.Dataset):
    def __init__(self, train, transforms, data_root=None, min_width=0):
        super(BELGIUMTS, self).__init__()
        if data_root is None:
            data_root = "data"
        if train:
            self.data_folder = os.path.join(data_root, "BelgiumTS/Training")
            self.images, self.labels = self._get_data_list(min_width=min_width)
            if min_width > 0:
                print(f'Loading BelgiumTS Train greater than {min_width} width. Loaded {len(self.images)} images.')
        else:
            self.data_folder = os.path.join(data_root, "BelgiumTS/Testing")
            self.images, self.labels = self._get_data_list(min_width)
            print(f'Loading BelgiumTS Test greater than {min_width} width. Loaded {len(self.images)} images.')

        self.transforms = transforms

    def _get_data_list(self, min_width=0):
        images = []
        labels = []
        for c in range(0, 62):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                if int(row[1]) >= min_width:
                    images.append(prefix + row[0])
                    labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label