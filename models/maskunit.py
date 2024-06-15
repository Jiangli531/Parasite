import torch
import torch.nn as nn
import torch.nn.functional as F

class Gumbel(nn.Module):
    

    def __init__(self, threshold, eps=torch.tensor(1e-8)):
        super(Gumbel, self).__init__()
        self.eps = eps
        self.gumbel_temp = torch.tensor(1.0)
        self.x = nn.Parameter(torch.randn(64, 100))
        self.threshold = threshold
        


    def forward(self, gumbel_noise=True):
        if not self.training:  
            return (self.x >= 0).float()

        if gumbel_noise:
            eps = self.eps
            U1 = torch.rand_like(self.x)
            U2 = torch.rand_like(self.x)
            g1 = -torch.log(-torch.log(U1 + eps) + eps)
            g2 = -torch.log(-torch.log(U2 + eps) + eps)
            temp = self.x + g1 - g2

        soft = torch.sigmoid(temp / self.gumbel_temp)

        

        hard = ((soft >= self.threshold).float() - soft).detach() + soft
        
        assert not torch.any(torch.isnan(hard))
        return hard


class GumbelLinear(nn.Module):
    

    def __init__(self, threshold, eps=torch.tensor(1e-8)):
        super(GumbelLinear, self).__init__()
        self.eps = eps
        self.gumbel_temp = torch.tensor(1.0)
        self.w_p = nn.Parameter(torch.randn(16, 16))
        self.bias = nn.Parameter(torch.randn(64, 16))
        self.threshold = threshold



    def forward(self, h, input, gumbel_noise=True):
       
        h = h.reshape(input.shape[0], 16)
        
        if (torch.max(h) > 100 or torch.min(h) < -100):
            
            h_min = torch.min(h)
            h_max = torch.max(h)
            
            mapped_h = ((h - h_min) / (h_max - h_min)) * 0.6 - 0.3

            
            mapped_h = torch.clamp(mapped_h, -0.3, 0.3)
            
            h = mapped_h
            
       

        
        mask = torch.mm(h, self.w_p) + self.bias
        mask = mask.reshape(64, 4, 4)
        

        if gumbel_noise:
            eps = self.eps
            U1 = torch.rand_like(mask)
            U2 = torch.rand_like(mask)
            g1 = -torch.log(-torch.log(U1 + eps) + eps)
            g2 = -torch.log(-torch.log(U2 + eps) + eps)
            temp = mask + g1 - g2
        
        soft = torch.sigmoid(temp / self.gumbel_temp)
        
        temp_soft = torch.flatten(soft, start_dim=1)
        hard = torch.zeros_like(soft)
        for i in range(soft.size(0)):
            topk_values, topk_indices = torch.topk(temp_soft[i], k=5)
            fifth_largest_value = topk_values[-1]
            hard[i, :, :] = ((soft[i, :, :] >= fifth_largest_value).float() - soft[i, :, :]).detach() + soft[i, :, :]
        
        return hard


