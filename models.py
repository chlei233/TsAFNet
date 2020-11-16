import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

from baselineResnet3d import generate_model, generate_model_preAct
class TsAFNet(nn.Module):
    def __init__(self, num_class, num_segments, modality='rgb', base_model_deepth=50, dropout=0.4, attention=True):
        super(TsAFNet, self).__init__()
        self.num_class  = num_class
        self.num_segments = num_segments
        self.modality = modality
        #self.consensus_type = consensus_type
        self.dropout = dropout
        self.attention = attention
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        #self.attention_stage = attention_stage
        '''if not attention and consensus_type != 'att':
            raise ValueError("attention pattern needs 'att' consensus_type")
        if not attention and attention_stage:
            raise ValueError("attention pattern needs to specify the stage")'''
        print(("""
Initializing TSN with base model (resnet): {}.
TSN Configurations:
    input_modality('rgb' default):     {}
    num_segments:       {}
    dropout_ratio:      {}
    attention:          {}
        """.format(base_model_deepth, self.modality, self.num_segments, self.dropout, self.attention)))
        
        self.__prepare_base_model(base_model_deepth)

    def __prepare_base_model(self, base_modle_deepth):
        if self.attention:
            return generate_model_preAct(base_modle_deepth, conv1_t_size=3, n_classes=self.num_class)
        else:
            return generate_model(base_modle_deepth, conv1_t_size=3, n_classes=self.num_class)

    def train(self, mode=True):
        super(TsAFNet, self).train(mode)