import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from itertools import combinations


class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dict_html(dict_obj, current_time):
    out = ''
    for key, value in dict_obj.items():

        
        if key in ['poisoning_test', 'test_batch_size', 'discount_size', 'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold' ]:
            continue

        out += f'<tr><td>{key}</td><td>{value}</td></tr>'
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output



def poison_random(batch, target, poisoned_number, poisoning, test=False):

    batch = batch.clone()
    target = target.clone()
    for iterator in range(0,len(batch)-1,2):

        if random.random()<poisoning:
            x_rand = random.randrange(-2,20)
            y_rand = random.randrange(-23, 2)
            batch[iterator + 1] = batch[iterator]
            batch[iterator+1][0][ x_rand + 2][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 2][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 2][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator+1][0][ x_rand + 6][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 6][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 6][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator+1][0][ x_rand + 5][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 4][ y_rand + 23] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 3][ y_rand + 24] = 2.5 + (random.random()-0.5)

            target[iterator+1] = poisoned_number
    return (batch, target)

def poison_test_random(batch, target, poisoned_number, poisoning, test=False):
    for iterator in range(0,len(batch)):
            x_rand = random.randrange(-2,20)
            y_rand = random.randrange(-23, 2)
            batch[iterator] = batch[iterator]
            batch[iterator][0][ x_rand + 2][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 2][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 2][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator][0][ x_rand + 6][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 6][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 6][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator][0][ x_rand + 5][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 4][ y_rand + 23] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 3][ y_rand + 24] = 2.5 + (random.random()-0.5)


            target[iterator] = poisoned_number
    return (batch, target)


class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def clip_grad_norm_dp(named_parameters, target_params, max_norm, norm_type=2):
    
    parameters = list(filter(lambda p: p[1]-target_params[p[0]], named_parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletSelector:
    

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

class FunctionNegativeTripletSelector(TripletSelector):
    

    def __init__(self, margin, negative_selection_fn, fac_label):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.fac_label = fac_label

    def get_triplets(self, embeddings, labels):
        
        embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        
        label = self.fac_label
        label_mask = (labels == label)
        label_indices = np.where(label_mask)[0]
        
        negative_indices = np.where(np.logical_not(label_mask))[0]
        
        anchor_positives = list(combinations(label_indices, 2))  
        anchor_positives = np.array(anchor_positives)
        ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
        for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
            loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
            loss_values = loss_values.data.cpu().numpy()
            
            hard_negative = self.negative_selection_fn(loss_values)
            if hard_negative is not None:
                hard_negative = negative_indices[hard_negative]
                triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        

        return torch.LongTensor(triplets)
    
def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def RandomNegativeTripletSelector(margin, fac_label): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                fac_label=fac_label)

