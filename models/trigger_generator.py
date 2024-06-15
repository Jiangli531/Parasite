

import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, helper):
        super(Generator, self).__init__()

        encoder_lis = [
            
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)
        
        self.fc = nn.Linear(32*32, helper.params['trigger_size']*helper.params['trigger_size'])
        self.trigger_size = helper.params['trigger_size']


    def forward(self, helper, x):
        if self.trigger_size == 32:
            x = self.encoder(x)
            x = self.bottle_neck(x)
            x = self.decoder(x)
            return x
        else:
            x = self.encoder(x)
            x = self.bottle_neck(x)
            x = self.decoder(x)
            trigger = self.fc(x.view(x.size(0) * x.size(1), x.size(2) * x.size(3)))
            trigger = trigger.view(x.size(0), x.size(1), self.trigger_size, self.trigger_size)
            
            trigger = torch.mean(trigger, dim=1, keepdim=True)
            trigger = torch.squeeze(trigger, dim=1)
            
            temp_trigger = torch.flatten(trigger, start_dim=1)
            for i in range(temp_trigger.size(0)):
                topk_values, topk_indices = torch.topk(temp_trigger[i], k=5)
                fifth_largest_value = topk_values[-1]
                trigger[i, :, :] = ((trigger[i, :, :] >= fifth_largest_value).float() - trigger[i, :, :]).detach() + trigger[i, :, :]
            if helper.params['dataset'] == 'cifar10' or self.params['dataset'] == 'cifar100':
                return trigger*255
            else:
                return trigger

    def add_trigger(self, poisoned_inputs, trigger):
        new_data = poisoned_inputs.detach().clone()
        _, channels, height, width = new_data.size()
        for i in range(trigger.size(0)):
            for j in range(trigger.size(1)):
                for k in range(trigger.size(2)):
                    if trigger[i, j, k] == 0:
                        new_data[i, :, height-self.trigger_size+j, width-self.trigger_size+k] = trigger[i, j, k] + new_data[i, :, height-self.trigger_size+j, width-self.trigger_size+k]
                    else:
                        new_data[i, :, height-self.trigger_size+j, width-self.trigger_size+k] = trigger[i, j, k]
        return new_data

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out




class DivLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward2(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        chunk_size = layer.size(0) // 2

        
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer=layer.view((layer.size(0), -1))
        chunk_size=layer.size(0) // 2

       
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))

