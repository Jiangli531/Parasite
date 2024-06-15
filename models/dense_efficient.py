

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul
from collections import OrderedDict
from torch.autograd import Variable, Function



from models.simple import SimpleNet


class _EfficientDensenetBottleneck(nn.Module):
    
    def __init__(self, shared_allocation_1, shared_allocation_2, num_input_channels, num_output_channels):
        super(_EfficientDensenetBottleneck, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.num_input_channels = num_input_channels

        self.norm_weight = nn.Parameter(torch.Tensor(num_input_channels))
        self.norm_bias = nn.Parameter(torch.Tensor(num_input_channels))
        self.register_buffer('norm_running_mean', torch.zeros(num_input_channels))
        self.register_buffer('norm_running_var', torch.ones(num_input_channels))
        self.conv_weight = nn.Parameter(torch.Tensor(num_output_channels, num_input_channels, 1, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        self.norm_running_mean.zero_()
        self.norm_running_var.fill_(1)
        self.norm_weight.data.uniform_()
        self.norm_bias.data.zero_()
        stdv = 1. / math.sqrt(self.num_input_channels)
        self.conv_weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        if isinstance(inputs, Variable):
            inputs = [inputs]

        
        fn = _EfficientDensenetBottleneckFn(self.shared_allocation_1, self.shared_allocation_2,
                                            self.norm_running_mean, self.norm_running_var,
                                            training=self.training, momentum=0.1, eps=1e-5)
        relu_output = fn(self.norm_weight, self.norm_bias, *inputs)

        
        conv_output = F.conv2d(relu_output, self.conv_weight, bias=None, stride=1,
                               padding=0, dilation=1, groups=1)

       
        dummy_fn = _DummyBackwardHookFn(fn)
        output = dummy_fn(conv_output)

       
        return output


class _DenseLayer(nn.Sequential):
    def __init__(self, shared_allocation_1, shared_allocation_2,
                 num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.drop_rate = drop_rate

        self.add_module('bn', _EfficientDensenetBottleneck(shared_allocation_1, shared_allocation_2,
                                                           num_input_features, bn_size * growth_rate))
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, x):
        if isinstance(x, Variable):
            prev_features = [x]
        else:
            prev_features = x
        new_features = super(_DenseLayer, self).forward(prev_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Container):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, storage_size=1024):
        self.final_num_features = num_input_features + (growth_rate * num_layers)
        self.shared_allocation_1 = _SharedAllocation(storage_size)
        self.shared_allocation_2 = _SharedAllocation(storage_size)

        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(self.shared_allocation_1, self.shared_allocation_2,
                                num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        
        self.shared_allocation_1.type_as(x)
        self.shared_allocation_2.type_as(x)

       
        final_size = list(x.size())
        final_size[1] = self.final_num_features
        final_storage_size = reduce(mul, final_size, 1)
        self.shared_allocation_1.resize_(final_storage_size)
        self.shared_allocation_2.resize_(final_storage_size)

        outputs = [x]
        for module in self.children():
            outputs.append(module.forward(outputs))
        return torch.cat(outputs, dim=1)


class DenseNetEfficient(SimpleNet):
    
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, name=None, created_time=None):

        super(DenseNetEfficient, self).__init__(name=f'{name}_DE', created_time=created_time)
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

       
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            features.size(0), -1)
        out = self.classifier(out)
        return out





class _SharedAllocation(object):
    
    def __init__(self, size):
        self._cpu_storage = torch.Storage(size)
        self._gpu_storages = []
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                with torch.cuda.device(device_idx):
                    self._gpu_storages.append(torch.Storage(size).cuda())

    def type(self, t):
        if not t.is_cuda:
            self._cpu_storage = self._cpu_storage.type(t)
        else:
            for device_idx, storage in enumerate(self._gpu_storages):
                with torch.cuda.device(device_idx):
                    self._gpu_storages[device_idx] = storage.type(t)

    def type_as(self, obj):
        if isinstance(obj, Variable):
            if not obj.is_cuda:
                self._cpu_storage = self._cpu_storage.type(obj.data.storage().type())
            else:
                for device_idx, storage in enumerate(self._gpu_storages):
                    with torch.cuda.device(device_idx):
                        self._gpu_storages[device_idx] = storage.type(obj.data.storage().type())
        elif torch.is_tensor(obj):
            if not obj.is_cuda:
                self._cpu_storage = self._cpu_storage.type(obj.storage().type())
            else:
                for device_idx, storage in enumerate(self._gpu_storages):
                    with torch.cuda.device(device_idx):
                        self._gpu_storages[device_idx] = storage.type(obj.storage().type())
        else:
            if not obj.is_cuda:
                self._cpu_storage = self._cpu_storage.type(obj.storage().type())
            else:
                for device_idx, storage in enumerate(self._gpu_storages):
                    with torch.cuda.device(device_idx):
                        self._gpu_storages[device_idx] = storage.type(obj.type())

    def resize_(self, size):
        if self._cpu_storage.size() < size:
            self._cpu_storage.resize_(size)
        for device_idx, storage in enumerate(self._gpu_storages):
            if storage.size() < size:
                with torch.cuda.device(device_idx):
                    self._gpu_storages[device_idx].resize_(size)
        return self

    def storage_for(self, val):
        if val.is_cuda:
            with torch.cuda.device_of(val):
                curr_device_id = torch.cuda.current_device()
                return self._gpu_storages[curr_device_id]
        else:
            return self._cpu_storage


class _EfficientDensenetBottleneckFn(Function):
    
    def __init__(self, shared_allocation_1, shared_allocation_2,
                 running_mean, running_var,
                 training=False, momentum=0.1, eps=1e-5):

        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

       
        self.prev_running_mean = self.running_mean.new(self.running_mean.size())
        self.prev_running_var = self.running_var.new(self.running_var.size())

    def forward(self, bn_weight, bn_bias, *inputs):
        if self.training:
            
            self.prev_running_mean.copy_(self.running_mean)
            self.prev_running_var.copy_(self.running_var)

        
        all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in all_num_channels[1:]:
            size[1] += num_channels
        storage = self.shared_allocation_1.storage_for(inputs[0])
        bn_input_var = Variable(type(inputs[0])(storage).resize_(size), volatile=True)
        relu_output = type(inputs[0])(storage).resize_(size)

        
        torch.cat(inputs, dim=1, out=bn_input_var.data)

        
        bn_weight_var = Variable(bn_weight)
        bn_bias_var = Variable(bn_bias)
        bn_output_var = F.batch_norm(bn_input_var, self.running_mean, self.running_var,
                                     bn_weight_var, bn_bias_var, training=self.training,
                                     momentum=self.momentum, eps=self.eps)

        
        torch.clamp(bn_output_var.data, 0, 1e100, out=relu_output)

        self.save_for_backward(bn_weight, bn_bias, *inputs)
        if self.training:
            
            self.running_mean.copy_(self.prev_running_mean)
            self.running_var.copy_(self.prev_running_var)
        return relu_output

    def prepare_backward(self):
        bn_weight, bn_bias = self.saved_tensors[:2]
        inputs = self.saved_tensors[2:]

        
        all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in all_num_channels[1:]:
            size[1] += num_channels
        storage1 = self.shared_allocation_1.storage_for(inputs[0])
        self.bn_input_var = Variable(type(inputs[0])(storage1).resize_(size), requires_grad=True)
        storage2 = self.shared_allocation_2.storage_for(inputs[0])
        self.relu_output = type(inputs[0])(storage2).resize_(size)

        
        torch.cat(inputs, dim=1, out=self.bn_input_var.data)

       
        self.bn_weight_var = Variable(bn_weight, requires_grad=True)
        self.bn_bias_var = Variable(bn_bias, requires_grad=True)
        self.bn_output_var = F.batch_norm(self.bn_input_var, self.running_mean, self.running_var,
                                          self.bn_weight_var, self.bn_bias_var, training=self.training,
                                          momentum=self.momentum, eps=self.eps)

        
        torch.clamp(self.bn_output_var.data, 0, 1e100, out=self.relu_output)

    def backward(self, grad_output):
        

        grads = [None] * len(self.saved_tensors)
        inputs = self.saved_tensors[2:]

        
        if not any(self.needs_input_grad):
            return grads

        
        relu_grad_input = grad_output.masked_fill_(self.relu_output <= 0, 0)
        self.bn_output_var.backward(gradient=relu_grad_input)
        if self.needs_input_grad[0]:
            grads[0] = self.bn_weight_var.grad.data
        if self.needs_input_grad[1]:
            grads[1] = self.bn_bias_var.grad.data

        
        if any(self.needs_input_grad[2:]):
            all_num_channels = [input.size(1) for input in inputs]
            index = 0
            for i, num_channels in enumerate(all_num_channels):
                new_index = num_channels + index
                grads[2 + i] = self.bn_input_var.grad.data[:, index:new_index]
                index = new_index

        
        del self.bn_input_var
        del self.bn_weight_var
        del self.bn_bias_var
        del self.bn_output_var

        return tuple(grads)


class _DummyBackwardHookFn(Function):
   
    def __init__(self, fn):
        
        self.fn = fn

    def forward(self, input):
        
        size = input.size()
        res = input.new(input.storage()).view(*size)
        return res

    def backward(self, grad_output):
        self.fn.prepare_backward()
        return grad_output