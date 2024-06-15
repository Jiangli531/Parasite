

import numpy as np
import torch
import torch.nn as nn

import torch
from .base_vae import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from .maskunit import GumbelLinear

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size
        self.batch_size = self.size[0]
        self.size = (self.batch_size, *self.size[1:])

    def forward(self, tensor):
        return tensor.view(self.size)

class Generator(BaseVAE):

    num_iter = 0 

    def __init__(self, helper, z_dim=10, nc=3):
        super(Generator, self).__init__()

        self.nc = nc
        self.z_dim = z_dim
        self.trigger_size = helper.params['trigger_size']
        self.dataset = helper.params['dataset']

        
        if self.dataset == 'emnist':
            nc = 1
            self.nc = nc

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  
            nn.ReLU(True),
            
        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(128, 256),  
            nn.ReLU(True),
            nn.Linear(256, 256),  
            nn.ReLU(True),
            nn.Linear(256, z_dim * 2),  
        )

        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(z_dim, 256),  
            nn.ReLU(True),
            nn.Linear(256, 256),  
            nn.ReLU(True),
            nn.Linear(256, 32 * 2 * 2),  
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(z_dim * 2, z_dim)  
        self.fc_var = nn.Linear(z_dim * 2, z_dim)
        
        self.fc = nn.Linear(32 * 32, helper.params['trigger_size'] * helper.params['trigger_size'])
        self.gumbel = GumbelLinear(threshold=helper.params['gumbel_threshold']).cuda()

    def encode(self, input: Tensor) -> List[Tensor]:
        
        result = self.encoder(input)
        temp = result.shape[0]
        result = result.view(temp, 32 * 2 * 2)
        result = self.encoder_linear(result)
        result = torch.flatten(result, start_dim=1)

        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)


        return [mu, log_var]

    def decode(self, z: Tensor, input: Tensor) -> Tensor:
        
        result = self.decoder_linear(z)
        temp = result.shape[0]
        result = result.view(temp, 32, 2, 2)
        result = self.decoder(result)
       

        trigger = self.fc(result.view(result.size(0) * result.size(1), result.size(2) * result.size(3)))
        trigger = trigger.view(result.size(0), result.size(1), self.trigger_size, self.trigger_size)
        
        trigger = torch.mean(trigger, dim=1, keepdim=True)
        trigger = torch.squeeze(trigger, dim=1)
        
        trigger = self.gumbel.forward(h=trigger, input=input)
        if self.dataset == 'cifar10' or self.dataset == 'cifar100' or self.dataset == 'gtsrb' :
            trigger = trigger * 255
        elif self.dataset == 'emnist' or self.dataset == 'BelgiumTS':
            trigger = trigger * 1
        else:
            trigger = trigger * 255
        
        

        return trigger

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        bs = input.shape[0]
        input_temp = input
        if input.shape[0] != 64:
            num = 64-input.shape[0]
            for i in range(num // input.shape[0]):
                input_temp = torch.cat((input_temp, input), dim=0)
            num = 64 - input_temp.shape[0]
            input_temp = torch.cat((input_temp, input[0:num]), dim=0)
            input = input_temp

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
       
        return [self.decode(z, input)[0:bs], input, mu, log_var]

    def loss_function(self, recons, input,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
       
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z, input)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        
        return self.forward(x)[0]

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