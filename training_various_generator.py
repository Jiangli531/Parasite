import argparse
import json
import datetime
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import copy
from MyDataset import MyDataset
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset

from torchvision import transforms

from image_helper import ImageHelper
from text_helper import TextHelper
import utils.csv_record as csv_record

from utils.utils import dict_html, RandomNegativeTripletSelector
logger = logging.getLogger("logger")

import yaml
import time
import visdom
import numpy as np
import pandas as pd

import random
from utils.text_load import *

from utils.losses import SupConLoss, OnlineTripletLoss
from torch.nn.modules.distance import PairwiseDistance

from models.autoencoder import Autoencoder


SupConLoss = SupConLoss().cuda()
criterion = torch.nn.CrossEntropyLoss()
poison_dir_params_variables = dict()
l2_dist = PairwiseDistance(2)

torch.manual_seed(0) #0
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(helper, epoch, poison_lr, train_data_sets, local_model, target_model, contrastive_model, is_poison, last_weight_accumulator=None, mask_grad_list=None):
    
    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        
        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
            continue
        weight_accumulator[name] = torch.zeros_like(data)

   
    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)

    current_number_of_adversaries = 0
    for model_id, _ in train_data_sets:
        if model_id == -1 or model_id in helper.params['adversary_list']:
            current_number_of_adversaries += 1
    logger.info(f'There are {current_number_of_adversaries} adversaries in the training.')

    for model_id in range(helper.params['no_models']):
        model = local_model
        
        model.copy_params(target_model.state_dict())
        for params in model.named_parameters():
            params[1].requires_grad = True

        target_lr = helper.params['target_lr']
        
        lr_init = helper.params['lr']
        if epoch <= 500:
            lr = epoch*(target_lr - lr_init)/499.0 + lr_init - (target_lr - lr_init)/499.0
        else:
            lr = epoch*(-target_lr)/1500 + target_lr*4.0/3.0 #1500 4 3 2000 5 4
            if lr <= 0.0001: #0.01
                lr = 0.0001

        if epoch > helper.params['poison_epochs'][-1]: # -1
            lr = helper.params['persistence_diff'] #0.005

        
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()

        start_time = time.time()
        if helper.params['type'] == 'text':
            current_data_model, train_data = train_data_sets[model_id]
            ntokens = len(helper.corpus.dictionary)
            hidden = model.init_hidden(helper.params['batch_size'])
        else:
            _, (current_data_model, train_data) = train_data_sets[model_id]
        batch_size = helper.params['batch_size']
        

        if current_data_model == -1:
            continue
        if is_poison and current_data_model in helper.params['adversary_list'] and \
                (epoch in helper.params['poison_epochs'] or helper.params['random_compromise']):
            logger.info('poison_now')
            contrastive_model = contrastive_model
            contrastive_model.copy_params(target_model.state_dict())
            
            poisoned_data = helper.poisoned_data_for_train
            _, acc = test(helper=helper, epoch=epoch,
                        data_source=helper.test_data,
                        model=model, is_poison=False, visualize=False)

            retrain_no_times = helper.params['retrain_poison_contrastive']
            step_lr = helper.params['poison_step_lr_contrastive']

            poison_lr = helper.params['poison_lr_contrastive']
            
            if helper.params['is_frozen_params_contrastive']:
                for params in contrastive_model.named_parameters():
                    if params[0] in helper.params['forzen_params']:
                        params[1].requires_grad = False


            poison_optimizer_contrastive = torch.optim.SGD(filter(lambda p:p.requires_grad, contrastive_model.parameters()), lr=poison_lr,
                                               momentum=helper.params['momentum_contrastive'],
                                               weight_decay=helper.params['decay_contrastive'])
            scheduler_contrastive = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer_contrastive,
                                                             milestones=helper.params['milestones_conrtastive'],
                                                             gamma=helper.params['lr_gamma_contrastive'])

            try:
                for internal_epoch in range(1, retrain_no_times + 1):
                    if step_lr:
                        scheduler_contrastive.step()
                    if helper.params['type'] == 'text':
                        data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
                    else:
                        
                        _, data_iterator = helper.train_data[0]




                    for batch_id, batch in enumerate(data_iterator):
                        poison_batch_list = np.zeros(helper.params['batch_size'])

                        if helper.params['label_flip_backdoor']:
                            batch_copy = copy.deepcopy(batch)

                        if helper.params['type'] == 'image':
                            if helper.params['regularize_batch']:
                                for i in range(helper.params['regularize_len']):

                                    if helper.params['semantic_backdoor']:
                                        if i%2:
                                            label_inter_pos = random.choice(range(len(helper.label_inter_dataset)))
                                            batch[0][i]=helper.label_inter_dataset[label_inter_pos][0]

                                            pos = helper.params['poison_images'][0]
                                            _, label = helper.train_dataset[pos]
                                            batch[1][i]=label
                                        else:
                                            label_fac_pos = random.choice(range(len(helper.label_fac_dataset)))
                                            batch[0][i]=helper.label_fac_dataset[label_fac_pos][0]
                                            batch[1][i]=helper.params['poison_label_swap']
                                    elif helper.params['pixel_pattern']:
                                        if i >= helper.params['poison_batch_len']:
                                            label_fac_pos = random.choice(range(len(helper.label_fac_dataset)))
                                            batch[0][i]=helper.label_fac_dataset[label_fac_pos][0]
                                            batch[1][i]=helper.params['poison_label_swap']


                            for i in range(helper.params['poisoning_per_batch_contrastive']):

                                if helper.params['semantic_backdoor'] or helper.params['pixel_pattern']:
                                    poison_batch_list = helper.params['poison_images'].copy()
                                elif helper.params['label_flip_backdoor']:
                                    poison_batch_list = helper.label_5_poison_dataset.copy()

                                check_list = helper.params['poison_images']
                                random.shuffle(poison_batch_list)
                                poison_batch_list = poison_batch_list[0 : min( helper.params['poison_batch_len'], len(batch[0]) )]

                                if helper.params['pattern_type'] == 6:
                                    batch_temp = []
                                    for pos, image in enumerate(poison_batch_list):
                                        poison_pos = len(poison_batch_list) * i + pos
                                        temp = torch.tensor(batch[0][poison_pos]).cuda()
                                        batch_temp.append(temp)
                                    for pos, image in enumerate(poison_batch_list):
                                        poison_pos = len(poison_batch_list) * i + pos
                                        temp = torch.tensor(batch[0][poison_pos]).cuda()
                                        batch_temp.append(temp)
                                    for pos, image in enumerate(poison_batch_list):
                                        poison_pos = len(poison_batch_list) * i + pos
                                        temp = torch.tensor(batch[0][poison_pos]).cuda()
                                        batch_temp.append(temp)
                                        if pos == 13:
                                            break
                                    inputs = torch.stack(batch_temp)
                                    with torch.no_grad():
                                        trigger = helper.generator(inputs)[0]
                                    for pos, image in enumerate(poison_batch_list):
                                        poison_pos = len(poison_batch_list) * i + pos
                                        batch[0][poison_pos] = helper.add_dynamic_trigger(batch[0][poison_pos], trigger[pos], helper.params['pattern_diff'])
                                        batch[1][poison_pos] = helper.params['poison_label_swap']

                                else:
                                    for pos, image in enumerate(poison_batch_list):
                                        poison_pos = len(poison_batch_list) * i + pos
                                       
                                        if helper.params['semantic_backdoor']:
                                            if helper.params['edge_case']:
                                                edge_pos = random.choice(range(len(helper.edge_poison_train)))
                                                image = helper.edge_poison_train[edge_pos]
                                                batch[0][poison_pos] = helper.transform_poison(image)
                                            else:
                                                batch[0][poison_pos] = helper.train_dataset[image][0]

                                        elif helper.params['label_flip_backdoor']:
                                            batch[0][poison_pos] = helper.test_dataset[image][0]
                                        elif helper.params['pixel_pattern']:
                                            batch[0][poison_pos] = helper.add_trigger(batch[0][poison_pos], helper.params['pattern_diff'])

                                        
                                        batch[1][poison_pos] = helper.params['poison_label_swap']

                        if helper.params['label_flip_backdoor']:
                            for i in range(helper.params['batch_size']):
                                if batch_copy[1][i] == 5:
                                    batch_copy[1][i] = helper.params['poison_label_swap']

                        data, targets = helper.get_batch(data_iterator, batch, False)
                        

                        if helper.params['label_flip_backdoor']:
                            data_copy, targets_copy = helper.get_batch(data_iterator, batch_copy, False)
                            data = torch.cat((data,data_copy))
                            targets = torch.cat((targets,targets_copy))

                        poison_optimizer_contrastive.zero_grad()
                        if helper.params['type'] == 'text':
                            hidden = helper.repackage_hidden(hidden)
                            output, hidden = contrastive_model(data, hidden)
                            class_loss = criterion(output[-1].view(-1, ntokens),
                                                   targets[-batch_size:])

                       
                        else:
                            output = contrastive_model(data)
                            contrastive_loss = SupConLoss(output, targets,
                                                        poison_per_batch=helper.params['poisoning_per_batch_contrastive'],
                                                        poison_images_len=len(helper.params['poison_images']),
                                                        scale_weight = helper.params['contrastive_loss_scale_weight'],
                                                        down_scale_weight = helper.params['contrastive_loss_down_scale_weight'],
                                                        helper=helper)

                        loss = helper.params['contrastive_loss_weight'] * contrastive_loss
                        loss_data_vis = float(loss.data)


                        
                        pos_class = helper.params['poison_label_swap']
                        if type(output) not in (tuple, list):
                            outputs = (output,)
                        loss_inputs = outputs
                        if targets is not None:
                            target = (targets,)
                            loss_inputs += target
                        loss_fn = OnlineTripletLoss(helper.margin, RandomNegativeTripletSelector(helper.margin, fac_label=pos_class))
                        loss_outputs = loss_fn(*loss_inputs)
                        
                        triplet_loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                        loss = helper.triplet_loss_scale_weight * triplet_loss
                        loss.backward()

                        if helper.params['gradmask_ratio'] != 1 and mask_grad_list and not helper.params['mask_contrastive']:
                            apply_grad_mask(contrastive_model, mask_grad_list)

                        if helper.params['diff_privacy_contrastive']:
                            poison_optimizer_contrastive.step()

                            model_norm = helper.model_dist_norm(contrastive_model, target_params_variables)
                            if model_norm > helper.params['s_norm_contrastive']:
                                logger.info(
                                    f'The limit reached for distance: '
                                    f'{helper.model_dist_norm(contrastive_model, target_params_variables)}')
                                norm_scale = helper.params['s_norm'] / ((model_norm))
                                for name, layer in contrastive_model.named_parameters():
                                    
                                    if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                        continue
                                    clipped_difference = norm_scale * (
                                    layer.data - target_model.state_dict()[name])
                                    layer.data.copy_(
                                        target_model.state_dict()[name] + clipped_difference)

                        elif helper.params['type'] == 'text':
                            torch.nn.utils.clip_grad_norm_(contrastive_model.parameters(),
                                                           helper.params['clip'])
                            poison_optimizer_contrastive.step()
                        else:
                            poison_optimizer_contrastive.step()
            except ValueError:
                logger.info('Converged earlier')

            model.copy_params(contrastive_model.state_dict())

            poisoned_data = helper.poisoned_data_for_train
            if helper.params['test_forgetting']:
                _, acc_p = test_poison(helper=helper, epoch=epoch,
                                   data_source=helper.poison_test_data_with_9,
                                   model=model, is_poison=True, visualize=False)
                _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data_without_9,
                             model=model, is_poison=False, visualize=False)
            else:
                _, acc_p = test_poison(helper=helper, epoch=epoch,
                                     data_source=helper.test_data_poison,
                                     model=model, is_poison=True, visualize=False)
                _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=model, is_poison=False, visualize=False)

           
            retrain_no_times = helper.params['retrain_poison']
            step_lr = helper.params['poison_step_lr']

            poison_lr = helper.params['poison_lr']
            
            if helper.params['is_frozen_params']:
                for params in model.named_parameters():
                    
                    if params[0] != 'linear.weight' and params[0] != 'linear.bias':
                        params[1].requires_grad = False

            if helper.params['anticipate']:
                anticipate_steps = helper.params['anticipate_steps']
                attack_model = copy.deepcopy(model)
                vis_model = copy.deepcopy(model)
                _, attack_params, attack_buffers = make_functional_with_buffers(attack_model)
                _, weight_names, _ = functorch._src.make_functional.extract_weights(attack_model)
                _, buffer_names, _ = functorch._src.make_functional.extract_buffers(attack_model)

                poison_optimizer = torch.optim.SGD(attack_params + attack_buffers, lr=poison_lr,
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            else:
                poison_optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=poison_lr,
                                               momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=helper.params['milestones'],
                                                             gamma=helper.params['lr_gamma'])

            acc = acc_initial
            try:

                for internal_epoch in range(1, retrain_no_times + 1):
                    if step_lr:
                        scheduler.step()
                    if helper.params['type'] == 'text':
                        data_iterator = range(0, poisoned_data.size(0) - 1, helper.params['bptt'])
                    else:
                        
                        _, data_iterator = helper.train_data[0]



                    for batch_id, batch in enumerate(data_iterator):

                        poison_batch_list = np.zeros(helper.params['batch_size'])
                        if helper.params['label_flip_backdoor']:
                            batch_copy=copy.deepcopy(batch)

                        if helper.params['type'] == 'image':
                            if helper.params['regularize_batch']:
                                for i in range(helper.params['regularize_len']):
                                    if helper.params['semantic_backdoor']:
                                        if i%2:
                                            label_inter_pos = random.choice(range(len(helper.label_inter_dataset)))
                                            batch[0][i]=helper.label_inter_dataset[label_inter_pos][0]

                                            pos = helper.params['poison_images'][0]
                                            _, label = helper.train_dataset[pos]
                                            batch[1][i]=label
                                        else:
                                            label_fac_pos = random.choice(range(len(helper.label_fac_dataset)))
                                            batch[0][i]=helper.label_fac_dataset[label_fac_pos][0]
                                            batch[1][i]=helper.params['poison_label_swap']

                                    elif helper.params['pixel_pattern']:
                                        label_fac_pos = random.choice(range(len(helper.label_fac_dataset)))
                                        batch[0][i]=helper.label_fac_dataset[label_fac_pos][0]
                                        batch[1][i]=helper.params['poison_label_swap']

                            for i in range(helper.params['poisoning_per_batch']):
                                if helper.params['semantic_backdoor'] or helper.params['pixel_pattern']:
                                    poison_batch_list = helper.params['poison_images'].copy()
                                elif helper.params['label_flip_backdoor']:
                                    poison_batch_list = helper.label_5_poison_dataset.copy()

                                random.shuffle(poison_batch_list)
                                poison_batch_list = poison_batch_list[0:min(helper.params['poison_batch_len'], len(batch[0]))]

                                if helper.params['pattern_type'] == 6:
                                    batch_temp = []
                                    for pos, image in enumerate(poison_batch_list):
                                        poison_pos = len(poison_batch_list) * i + pos
                                        temp = torch.tensor(batch[0][poison_pos]).cuda()
                                        batch_temp.append(temp)
                                    for pos, image in enumerate(poison_batch_list):
                                        poison_pos = len(poison_batch_list) * i + pos
                                        temp = torch.tensor(batch[0][poison_pos]).cuda()
                                        batch_temp.append(temp)
                                    for pos, image in enumerate(poison_batch_list):
                                        poison_pos = len(poison_batch_list) * i + pos
                                        temp = torch.tensor(batch[0][poison_pos]).cuda()
                                        batch_temp.append(temp)
                                        if pos == 13:
                                            break
                                    inputs = torch.stack(batch_temp)
                                    with torch.no_grad():
                                        trigger = helper.generator(inputs)[0]
                                    for pos, image in enumerate(poison_batch_list):
                                        poison_pos = len(poison_batch_list) * i + pos
                                        batch[0][poison_pos] = helper.add_dynamic_trigger(batch[0][poison_pos], trigger[pos], helper.params['pattern_diff'])
                                        batch[1][poison_pos] = helper.params['poison_label_swap']
                                else:
                                    for pos, image in enumerate(poison_batch_list):
                                        poison_pos = len(poison_batch_list) * i + pos
                                        if helper.params['semantic_backdoor']:
                                            if helper.params['edge_case']:
                                                edge_pos = random.choice(range(len(helper.edge_poison_train)))
                                                image = helper.edge_poison_train[edge_pos]
                                                batch[0][poison_pos] = helper.transform_poison(image)
                                            else:
                                                batch[0][poison_pos] = helper.train_dataset[image][0]
                                        elif helper.params['label_flip_backdoor']:
                                            batch[0][poison_pos] = helper.test_dataset[image][0]
                                        elif helper.params['pixel_pattern']:
                                            batch[0][poison_pos] = helper.add_trigger(batch[0][poison_pos], helper.params['pattern_diff'])
                                        batch[1][poison_pos] = helper.params['poison_label_swap']

                        if helper.params['label_flip_backdoor']:
                            for i in range(helper.params['batch_size']):
                                if batch_copy[1][i] == 5:
                                    batch_copy[1][i] = helper.params['poison_label_swap']

                        data, targets = helper.get_batch(data_iterator, batch, False)
                        if helper.params['label_flip_backdoor']:
                            data_copy, targets_copy = helper.get_batch(data_iterator, batch_copy, False)
                            data = torch.cat((data, data_copy))
                            targets = torch.cat((targets, targets_copy))

                        poison_optimizer.zero_grad()

                        if helper.params['anticipate']:
                            func_model, curr_params, curr_buffers = make_functional_with_buffers(model)
                            loss = None
                            for anticipate_i in range(anticipate_steps):
                                if anticipate_i == 0:
                                    
                                    curr_params = train_with_functorch(helper, epoch + anticipate_i, func_model, curr_params, curr_buffers, data_iterator, num_users=helper.params['no_models']-1)

                                    
                                    curr_params = [(attack_params[i] + curr_params[i] * (helper.params['no_models'] - 1)) / helper.params['no_models'] for i in range(len(curr_params))]
                                    curr_buffers = [(attack_buffers[i] + curr_buffers[i] * (helper.params['no_models'] - 1)) / helper.params['no_models'] for i in range(len(curr_buffers))]
                                else:
                                    
                                    curr_params = train_with_functorch(helper, epoch + anticipate_i, func_model, curr_params, curr_buffers, data_iterator, num_users=helper.params['no_models'])

                                
                                logits = func_model(curr_params, curr_buffers, data)
                                y = targets

                                if loss is None:
                                    loss = nn.functional.cross_entropy(logits, y).mean()
                                else:
                                    loss += nn.functional.cross_entropy(logits, y).mean()

                        else:
                            if helper.params['type'] == 'text':
                                hidden = helper.repackage_hidden(hidden)
                                output, hidden = model(data, hidden)
                                class_loss = criterion(output[-1].view(-1, ntokens),
                                                    targets[-batch_size:])
                            else:
                                output = model(data)
                                class_loss = nn.functional.cross_entropy(output, targets)

                            all_model_distance = helper.model_dist_norm(target_model, target_params_variables)
                            norm = 2
                            distance_loss = helper.model_dist_norm_var(model, target_params_variables)
                            loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss

                            similarity_loss_with_target = helper.poison_dir_cosine_similarity(model, target_params_variables)


                        loss.backward()

                        

                        if helper.params['diff_privacy']:
                            poison_optimizer.step()

                            model_norm = helper.model_dist_norm(model, target_params_variables)
                            if model_norm > helper.params['s_norm']:
                                logger.info(
                                    f'The limit reached for distance: '
                                    f'{helper.model_dist_norm(model, target_params_variables)}')
                                norm_scale = helper.params['s_norm'] / ((model_norm))
                                for name, layer in model.named_parameters():
                                    
                                    if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                        continue
                                    clipped_difference = norm_scale * (
                                    layer.data - target_model.state_dict()[name])
                                    layer.data.copy_(
                                        target_model.state_dict()[name] + clipped_difference)

                        elif helper.params['type'] == 'text':
                            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                           helper.params['clip'])
                            poison_optimizer.step()
                        else:
                            poison_optimizer.step()

                        if helper.params['anticipate']:
                            functorch._src.make_functional.load_weights(vis_model, weight_names, attack_params)
                            functorch._src.make_functional.load_buffers(vis_model, buffer_names, attack_buffers)

                    if helper.params['test_forgetting']:
                        loss, acc = test(helper=helper, epoch=epoch, data_source=helper.test_data_without_9,
                                     model=model, is_poison=False, visualize=False)
                        loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                            data_source=helper.poison_test_data_with_9,
                                            model=model, is_poison=True, visualize=False, log_poison_choice=True)
                    else:
                        loss, acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                        model=model, is_poison=False, visualize=False)
                        loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                        data_source=helper.test_data_poison,
                                        model=model, is_poison=True, visualize=False, log_poison_choice=True)


                    if loss_p<=0.0001:
                        if helper.params['type'] == 'image' and acc<acc_initial:
                            if step_lr:
                                scheduler.step()
                            continue

                        raise ValueError()

            except ValueError:
                logger.info('Converged earlier')

            if helper.params['anticipate']:
                functorch._src.make_functional.load_weights(model, weight_names, attack_params)
                functorch._src.make_functional.load_buffers(model, buffer_names, attack_buffers)

            logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
            logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')



            if helper.params['diff_privacy']:
                model_norm = helper.model_dist_norm(model, target_params_variables)
                if model_norm > helper.params['s_norm']:
                    norm_scale = helper.params['s_norm'] / (model_norm)
                    for name, layer in model.named_parameters():
                       
                        if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                            continue
                        clipped_difference = norm_scale * (
                        layer.data - target_model.state_dict()[name])
                        layer.data.copy_(target_model.state_dict()[name] + clipped_difference)
                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning and clipping: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['track_distance'] and model_id < 10:
                distance = helper.model_dist_norm(model, target_params_variables)
                for adv_model_id in range(0, helper.params['number_of_adversaries']):
                    logger.info(
                        f'MODEL {adv_model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                        f'Distance to the global model: {distance:.4f}. '
                        f'Dataset size: {train_data.size(0)}')


            for key, value in model.state_dict().items():
                
                if helper.params.get('tied', False) and key == 'decoder.weight' or '__'in key:
                    continue
                target_value = target_model.state_dict()[key]
                new_value = target_value + (value - target_value) * current_number_of_adversaries
                model.state_dict()[key].copy_(new_value)

            distance = helper.model_dist_norm(model, target_params_variables)
            logger.info(f"Total norm for {current_number_of_adversaries} "
                        f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

            if epoch == helper.params['poison_epochs'][0]:
                for name, param in model.named_parameters():
                    poison_dir_params_variables[name] = model.state_dict()[name].clone().detach().requires_grad_(False)


        
        else:

            
            if helper.params['fake_participants_load']:
                continue
            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.
                if helper.params['type'] == 'text':
                    data_iterator = range(0, train_data.size(0) - 1, helper.params['bptt'])
                else:
                    data_iterator = train_data
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(train_data, batch,
                                                      evaluation=False)
                    if helper.params['type'] == 'text':
                        hidden = helper.repackage_hidden(hidden)
                        output, hidden = model(data, hidden)
                        loss = criterion(output.view(-1, ntokens), targets)
                    else:
                        output = model(data)
                        loss = criterion(output, targets)

                    loss.backward()

                    if helper.params['diff_privacy_benign']:
                        optimizer.step()
                        model_norm = helper.model_dist_norm(model, target_params_variables)

                        if model_norm > helper.params['s_norm']:
                            norm_scale = helper.params['s_norm'] / (model_norm)
                            for name, layer in model.named_parameters():
                                
                                if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                                    continue
                                clipped_difference = norm_scale * (
                                layer.data - target_model.state_dict()[name])
                                layer.data.copy_(
                                    target_model.state_dict()[name] + clipped_difference)
                    elif helper.params['type'] == 'text':
                        
                        torch.nn.utils.clip_grad_norm_(model.parameters(), helper.params['clip'])
                        optimizer.step()
                    else:
                        optimizer.step()

                    total_loss += loss.data

                    if helper.params["report_train_loss"] and batch_id % helper.params[
                        'log_interval'] == 0 and batch_id > 0:
                        cur_loss = total_loss.item() / helper.params['log_interval']
                        elapsed = time.time() - start_time
                        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
                        logger.info('{}: | model {} | epoch {:3d} | internal_epoch {:3d} '
                                    '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                    'loss {:5.2f} | ppl {:8.2f}'
                                            .format(current_time, model_id, epoch, internal_epoch,
                                            batch_id,len(batch[0]),
                                            helper.params['lr'],
                                            elapsed * 1000 / helper.params['log_interval'],
                                            cur_loss,
                                            math.exp(cur_loss) if cur_loss < 30 else -1.))
                        total_loss = 0
                        start_time = time.time()

            logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

            if helper.params['track_distance'] and model_id < 10:
                
                distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                    f'Distance to the global model: {distance_to_global_model:.4f}. '
                    f'Dataset size: {train_data.size(0)}')


        similarity_loss_with_target = helper.poison_dir_cosine_similarity(model, target_params_variables)
        if epoch >= helper.params['poison_epochs'][0]-1:
            logger.info(f'epoch:{epoch}, model:{model_id} similarity loss between model and global model is {similarity_loss_with_target}')

        copy_model_param = dict(model.named_parameters())
        copy_target_model_param = dict(helper.target_model.named_parameters())
        params_list = []
        for key, value in copy_model_param.items():
            new_value = value - copy_target_model_param[key]
            params_list.append(new_value.view(-1))
        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list.clone().detach().cuda())


        scale = max( 1.0, float(torch.abs(l2_norm/helper.params['norm_bound'] )))
        logger.info(f'epoch:{epoch},model:{model_id},l2_norm:{l2_norm},scale:{scale}')

        if helper.params['norm_clip']:
            for name, data in model.state_dict().items():
                if 'running_var' in name or 'running_mean' in name or 'num_batches_tracked' in name:
                    continue

                new_value = helper.target_model.state_dict()[name] + (model.state_dict()[name] - helper.target_model.state_dict()[name])/scale
                model.state_dict()[name].copy_(new_value)

        for name, data in model.state_dict().items():
            
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(data - target_model.state_dict()[name])


    if helper.params["fake_participants_save"]:
        torch.save(weight_accumulator,
                   f"{helper.params['fake_participants_file']}_"
                   f"{helper.params['s_norm']}_{helper.params['no_models']}")
    elif helper.params["fake_participants_load"]:
        fake_models = helper.params['no_models'] - helper.params['number_of_adversaries']
        fake_weight_accumulator = torch.load(
            f"{helper.params['fake_participants_file']}_{helper.params['s_norm']}_{fake_models}")
        logger.info(f"Faking data for {fake_models}")
        for name in target_model.state_dict().keys():
            
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__'in name:
                continue
            weight_accumulator[name].add_(fake_weight_accumulator[name])

    return weight_accumulator



def test(helper, epoch, data_source,
         model, is_poison=False, visualize=True):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0
    if helper.params['type'] == 'text':
        hidden = model.init_hidden(helper.params['test_batch_size'])
        random_print_output_batch = \
        random.sample(range(0, (data_source.size(0) // helper.params['bptt']) - 1), 1)[0]
        data_iterator = range(0, data_source.size(0)-1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        dataset_size = len(data_source.dataset)
        data_iterator = data_source

    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        
        if helper.params['type'] == 'text':
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, helper.n_tokens)
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = helper.repackage_hidden(hidden)
            pred = output_flat.data.max(1)[1]
            correct += pred.eq(targets.data).sum().to(dtype=torch.float)
            total_test_words += targets.data.shape[0]

            
            if batch_id == random_print_output_batch * helper.params['bptt'] and \
                    helper.params['output_examples'] and epoch % 5 == 0:
                expected_sentence = helper.get_sentence(targets.data.view_as(data)[:, 0])
                expected_sentence = f'*EXPECTED*: {expected_sentence}'
                predicted_sentence = helper.get_sentence(pred.view_as(data)[:, 0])
                predicted_sentence = f'*PREDICTED*: {predicted_sentence}'
                score = 100. * pred.eq(targets.data).sum() / targets.data.shape[0]
                logger.info(expected_sentence)
                logger.info(predicted_sentence)

        else:
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                              reduction='sum').item() 
            
            pred = output.data.max(1)[1]  
            
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    if helper.params['type'] == 'text':
        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / (dataset_size-1)
        logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                       total_l, correct, total_test_words,
                                                       acc))
        acc = acc.item()
        total_l = total_l.item()
    else:
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size
        current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
        


    model.train()
    return (total_l, acc)


def test_poison(helper, epoch, data_source,
                model, is_poison=False, visualize=True, log_poison_choice=False):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    total_test_words = 0.0
    batch_size = helper.params['test_batch_size']
    if helper.params['type'] == 'text':
        ntokens = len(helper.corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        data_iterator = range(0, data_source.size(0) - 1, helper.params['bptt'])
        dataset_size = len(data_source)
    else:
        data_iterator = data_source
        
        dataset_size = 9000

    for batch_id, batch in enumerate(data_iterator):
        poison_choice_list = []
        if helper.params['type'] == 'image':
            if not helper.params['test_forgetting']:

                if helper.params['pattern_type'] == 6:
                    batch_temp = []
                    for pos in range(len(batch[0])):
                        temp = torch.tensor(batch[0][pos]).cuda()
                        batch_temp.append(temp)
                    inputs = torch.stack(batch_temp)
                    with torch.no_grad():
                        trigger = helper.generator(inputs)[0]
                    for pos in range(len(batch[0])):
                        batch[0][pos] = helper.add_dynamic_trigger(batch[0][pos], trigger[pos], helper.params['pattern_diff'])
                        batch[1][pos] = helper.params['poison_label_swap']
                else:
                    for pos in range(len(batch[0])):
                        if helper.params['semantic_backdoor']:
                            if helper.params['edge_case']:
                                edge_pos = random.choice(range(len(helper.edge_poison_test)))
                                image = helper.edge_poison_test[edge_pos]
                                batch[0][pos] = helper.transform_test(image)
                            else:
                                poison_choice  = random.choice(helper.params['poison_images_test'])
                                batch[0][pos] = helper.train_dataset[poison_choice][0]
                        elif helper.params['label_flip_backdoor']:
                            poison_choice  = random.choice(helper.label_5_poison_dataset)
                            batch[0][pos] = helper.test_dataset[poison_choice][0]
                        elif helper.params['pixel_pattern']:
                            batch[0][pos] = helper.add_trigger(batch[0][pos], helper.params['pattern_diff'])
                        

                        batch[1][pos] = helper.params['poison_label_swap']


        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        
        if helper.params['type'] == 'text':
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += 1 * criterion(output_flat[-batch_size:], targets[-batch_size:]).data
            hidden = helper.repackage_hidden(hidden)

            
            pred = output_flat.data.max(1)[1][-batch_size:]


            correct_output = targets.data[-batch_size:]
            correct += pred.eq(correct_output).sum()
            total_test_words += batch_size
        else:
            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets,
                                              reduction='sum').data.item()  
            pred = output.data.max(1)[1] 
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float)

    if helper.params['type'] == 'text':
        acc = 100.0 * (correct / total_test_words)
        total_l = total_loss.item() / dataset_size
    else:
        acc = 100.0 * (correct / dataset_size)
        total_l = total_loss / dataset_size
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    

    model.train()
    return total_l, acc

def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)

def train_with_functorch(hlpr, epoch, func_model, params, buffers, train_loader, num_users=1):
    lr = hlpr.params['anticipate_lr'] * hlpr.params['anticipate_gamma'] ** (epoch)

    def compute_loss(params, buffers, x, y):
        logits = func_model(params, buffers, x)
        loss = nn.functional.cross_entropy(logits, y).mean()
        return loss

    for i, batch in enumerate(train_loader):
        for _ in range(hlpr.params['anticipate_local_epochs']):
            data, targets = hlpr.get_batch(train_loader, batch)
            grads = grad(compute_loss)(params, buffers, data, targets)
            params = [p - g * lr for p, g, in zip(params, grads)]
        break
    return params


def train_generator(helper, generator_lr, train_data, output_name):
    helper.generator.cuda()
    helper.generator.train()
    optimizer_generator = torch.optim.Adam(helper.generator.parameters(), generator_lr, betas=(0.5, 0.9))
    print(" Training Generator:", flush = True)

    
    
    auto_model = Autoencoder()
    auto_model.load_state_dict(torch.load('./saved_models/autoencoder/autoencoder_model.pth').state_dict())
    auto_model.eval()
    auto_model.cuda()

    feature_list = []
    for _, (inputs, labels) in enumerate(helper.nine_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        if inputs.shape[0] != 64:
            continue
        feature = auto_model.feature_forward(inputs)
        for i in range(feature.shape[0]):
            feature_list.append(feature[i])
    feature_list = torch.stack(feature_list, dim=0)
    
    center = torch.mean(feature_list, dim=0)
    
    distances = []
    for i in range(feature_list.shape[0]):
        distances.append(torch.sqrt(torch.sum(torch.pow(feature_list[i] - center, 2))))
    distances = torch.stack(distances, dim=0)
   
    indices = torch.topk(distances, k=64, largest=False)[1]
    
    clean_feature = feature_list[indices].detach()

   
    torch.save(helper.generator.state_dict(), './saved_models/' + output_name + '/init_generator.pth')
    init_generator_path = './saved_models/' + output_name + '/init_generator.pth'

    
    for lamda in [0, 0.25, 0.5, 0.75, 1]:
        logger.info('lamda: {}'.format(lamda))
        helper.generator.load_state_dict(torch.load(init_generator_path))
        
        losses_lf = []  
        losses_ld = []  
        for epoch in range(10):
            for i, (inputs, labels) in enumerate(train_data):
                if inputs.shape[0] != 64:
                    continue

                if i % 2 == 0:
                    inputs_1, labels_1 = inputs.cuda(), labels.cuda()
                    continue
                else:
                    inputs_2, labels_2 = inputs.cuda(), labels.cuda()
                    optimizer_generator.zero_grad()
                    out_1 = helper.generator(inputs_1)
                    out_2 = helper.generator(inputs_2)
                    trigger_1 = out_1[0]
                    mu_1 = out_1[2]
                    log_var_1 = out_1[3]
                    trigger_2 = out_2[0]
                    mu_2 = out_2[2]
                    log_var_2 = out_2[3]
                    poison_images_1 = helper.generator.add_trigger(inputs_1, trigger_1)
                    poison_images_2 = helper.generator.add_trigger(inputs_2, trigger_2)
                    poison_feature_1 = auto_model.feature_forward(poison_images_1)
                    poison_feature_2 = auto_model.feature_forward(poison_images_2)
                    distance_1 = torch.nn.functional.cosine_similarity(poison_feature_1, clean_feature)
                    distance_2 = torch.nn.functional.cosine_similarity(poison_feature_2, clean_feature)
                    l_f = torch.mean(distance_1+distance_2)
                    
                    if helper.params['dataset'] == 'emnist' or helper.params['dataset'] == 'BelgiumTS':
                        
                        l_d = (trigger_1-trigger_2).pow(2).mean()
                    else:
                        l_d = (trigger_1-trigger_2).pow(2).mean()/255
                    recons_loss = -l_d
                    kld_loss_1 = torch.mean(-0.5 * torch.sum(1 + log_var_1 - mu_1 ** 2 - log_var_1.exp(), dim=1), dim=0)
                    kld_loss_2 = torch.mean(-0.5 * torch.sum(1 + log_var_2 - mu_2 ** 2 - log_var_2.exp(), dim=1), dim=0)
                    kld_loss = kld_loss_1 + kld_loss_2


                    
                    scalar_ld = torch.log10(l_d+1)
                    scalar_lf = torch.log10(l_f+1)
                    scalar_kld = torch.log10(kld_loss+1)
                    loss_generator = scalar_kld - (1-lamda)*scalar_ld - lamda*scalar_lf 
                    if i % 100 == 1:
                        logger.info("Normalize: l_f: {:.4f}, l_d: {:.4f}, kld_loss: {:.4f}, s_l_f: {:.4f}, s_l_d: {:.4f}, s_kld_loss: {:.4f}".format(l_f.item(),l_d.item(),kld_loss.item(),scalar_lf.item(),scalar_ld.item(),scalar_kld.item()))
                    
                    optimizer_generator.zero_grad()
                    loss_generator.backward()
                    optimizer_generator.step()

                    if i % 100 == 1:
                        logger.info('Iteration {}, Generator loss: {:.4f}, l_f: {:.4f}, l_d: {:.4f}, recons_loss: {:.4f}, '
                              'kld_loss: {:.4f}'.format(i, loss_generator.item(), l_f.item(), l_d.item(),
                                                        recons_loss.item(), kld_loss.item()))
            
            losses_lf.append(l_f.item())
            losses_ld.append(l_d.item())
            logger.info(
                'Epoch {}, Generator loss: {:.4f}, l_f: {:.4f}, l_d: {:.4f}, recons_loss: {:.4f}, kld_loss: {:.4f}'
                .format(epoch, loss_generator.item(), l_f.item(), l_d.item(), recons_loss.item(), kld_loss.item()))
            torch.save(helper.generator.state_dict(), './saved_models/' + output_name + '/generator_lamda_' + str(lamda) + '_epoch_' + str(epoch) + '.pth')
        
        logger.info('--------------------')
        logger.info("This turn's lamda is : {}".format(lamda))
        


if __name__ == '__main__':
    print('Start training', flush=True)
    time_start_load_everything = time.time()
    mask_grad_list = None
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--is_frozen_params',
                        default=1,
                        type=int,
                        help='is_frozen_params')
    parser.add_argument('--retrain_poison_contrastive',
                        default=10,
                        type=int,
                        help='retrain_poison_contrastive')
    parser.add_argument('--retrain_poison',
                        default=3,
                        type=int,
                        help='retrain_poison')
    parser.add_argument('--poison_lr',
                        default=0.01,
                        type=float,
                        help='poison_lr')
    parser.add_argument('--gradmask_ratio',
                        default=1.0,
                        type=float,
                        help='gradmask_ratio')
    parser.add_argument('--lr_gamma',
                        default=0.005,
                        type=float,
                        help='lr_gamma')
    parser.add_argument('--GPU_id',
                        default="0",
                        type=str,
                        help='GPU_id')
    parser.add_argument('--pattern_diff',
                        default=0.0,
                        type=float,
                        help='pattern_diff')
    parser.add_argument('--pattern_type',
                        default=1,
                        type=int,
                        help='pattern_type')
    parser.add_argument('--trigger_size',
                        default=4,
                        type=int,
                        help='trigger_size')
    parser.add_argument('--pixel_num',
                        default=5,
                        type=int,
                        help='pixel_num')
    parser.add_argument('--size_of_secret_dataset',
                        default=25,
                        type=int,
                        help='size_of_secret_dataset')
    parser.add_argument('--persistence_diff',
                        default=0.005,
                        type=float,
                        help='persistence_diff')
    parser.add_argument('--mask_contrastive',
                        default=0,
                        type=int,
                        help='mask_contrastive')
    parser.add_argument('--contrastive_loss_scale_weight',
                        default=2.0,
                        type=float,
                        help='contrastive_loss_scale_weight')
    parser.add_argument('--model_type',
                        default="ResNet18",
                        type=str,
                        help='model_type')
    parser.add_argument('--anticipate',
                        default=0,
                        type=int,
                        help='anticipate')
    parser.add_argument('--anticipate_steps',
                        default=5,
                        type=int,
                        help='anticipate_steps')
    parser.add_argument('--anticipate_lr',
                        default=0.01,
                        type=float,
                        help='anticipate_lr')
    parser.add_argument('--anticipate_local_epochs',
                        default=2,
                        type=int,
                        help='anticipate_local_epochs')
    parser.add_argument('--epochs',
                        default=4000,
                        type=int,
                        help='epochs')
    parser.add_argument('--edge_case',
                        default=0,
                        type=int,
                        help='edge_case')
    parser.add_argument('--margin', default=0.5, type=float, metavar='MG',
                        help='margin (default: 0.5)')
    parser.add_argument('--triplet_loss_scale_weight',
                        default=2.0,
                        type=float,
                        help='triplet_loss_scale_weight')
    parser.add_argument('--output_name',
                        default="test",
                        type=str,
                        help='output_name')
    parser.add_argument('--generator_epoch',
                        default=25,
                        type=int,
                        help='generator_epoch')
    parser.add_argument('--generator_lr',
                        default=0.003,
                        type=float,
                        help='generator_lr')
    parser.add_argument('--alpha',
                        default=0.1,
                        type=float,
                        help='alpha')
    parser.add_argument('--gumbel_threshold',
                        default=0.3,
                        type=float,
                        help='gumbel_threshold')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id
    is_make_dir = False
    scale_running_var = 3
    scale_running_mean = 1.5
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.safe_load(f)

    params_loaded.update(vars(args))
    check = params_loaded['is_frozen_params']
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    if params_loaded['type'] == "image":
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'image'), output_name=args.output_name)

    helper.load_data()
    helper.create_model()
    helper.margin = args.margin
    helper.triplet_loss_scale_weight = args.triplet_loss_scale_weight
    
    if helper.params['is_poison']:
        helper.params['adversary_list'] = [0]+ \
                                random.sample(range(helper.params['number_of_total_participants']),
                                                      helper.params['number_of_adversaries']-1)
        logger.info(f"Poisoned following participants: {len(helper.params['adversary_list'])}")
    else:
        helper.params['adversary_list'] = list()

    best_loss = float('inf')
    logger.info(f"We use following environment for graphs:  {helper.params['environment_name']}")
    participant_ids = range(len(helper.train_data))
    logger.info(f"participant_ids are: {participant_ids}")
    mean_acc = list()

    results = {'poison': list(), 'number_of_adversaries': helper.params['number_of_adversaries'],
               'poison_type': helper.params['poison_type'], 'current_time': current_time,
               'sentence': helper.params.get('poison_sentences', False),
               'random_compromise': helper.params['random_compromise'],
               'baseline': helper.params['baseline']}

    weight_accumulator = None
    is_grad_log = True
    is_make_grad_dir = True
    forwardset_rate = helper.params['poisoning_per_batch_low']
   
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)
    dist_list = list()
    poison_lr = helper.params['poison_lr']
    
    higher_than_mark = False


    
    torch.save(helper.target_model.state_dict(), './saved_models/model_1800.pth')


    
    nine_data = []
    if helper.params['dataset'] == 'cifar10':
        target_poison_label = 9
    elif helper.params['dataset'] == 'emnist':
        target_poison_label = 2
    elif helper.params['dataset'] == 'gtsrb':
        target_poison_label = 7
    elif helper.params['dataset'] == 'BelgiumTS':
        target_poison_label = 7
    for data, label in tqdm(helper.train_dataset):
        if label == target_poison_label:
            nine_data.append((data, label))
    
    nine_dataset = Subset(MyDataset(nine_data), list(range(len(nine_data))))
    helper.nine_loader = DataLoader(nine_dataset, batch_size=64, shuffle=False)

   
    data_iterator_generator = torch.utils.data.DataLoader(helper.train_dataset, batch_size=helper.params['batch_size'])
    
    train_generator(helper=helper, generator_lr=helper.params['generator_lr'], train_data=data_iterator_generator, output_name=helper.params['output_name'])
    
    os._exit(0)
    
    
