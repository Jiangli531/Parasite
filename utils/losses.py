
from __future__ import print_function

import torch
import torch.nn as nn

import logging
from torch.nn.modules.distance import PairwiseDistance
import torch.nn.functional as F

logger = logging.getLogger("logger")


class SupConLoss(nn.Module):
    
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, poison_per_batch=None, poison_images_len=None, scale_weight=1, down_scale_weight=1, helper=None):
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            
            mask_scale = mask.clone().detach()
            mask_cross_feature = torch.ones_like(mask_scale).to(device)
            
            label_flatten = labels.view(-1)
            for ind, label in enumerate(labels.view(-1)):
                if label == helper.fac_label:
                    mask_scale[ind, :] = mask[ind, :] * scale_weight
                if ind < poison_per_batch*poison_images_len:
                    label_1_row_eq = torch.eq(label_flatten, 1).float().to(device) * down_scale_weight 
                    label_1_row_nq = torch.ne(label_flatten, 1).float().to(device)
                    label_1_row = label_1_row_eq + label_1_row_nq
                    mask_cross_feature[ind,:] = mask_cross_feature[ind,:] * label_1_row
                if label == 1:
                    mask_cross_feature[ind, 0:poison_per_batch*poison_images_len] = mask_cross_feature[ind, 0:poison_per_batch*poison_images_len] * down_scale_weight

        else:
            mask = mask.float().to(device)

        contrast_feature = features
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = features
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) * mask_cross_feature 
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        mask_scale = mask_scale * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos_mask = (mask_scale * log_prob).sum(1)
        mask_check = mask.sum(1)
        for ind, mask_item in enumerate(mask_check):
            if mask_item == 0:
                continue
            else:
                mask_check[ind] = 1 / mask_item
        mask_apply = mask_check
        mean_log_prob_pos = mean_log_prob_pos_mask * mask_apply
        
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(batch_size).mean()

        return loss

class OnlineTripletLoss(nn.Module):
    

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)