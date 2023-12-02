import os

import torch
from torch import nn

from model import load_pretrained, Gated_Embedding_Unit

global_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransferModel(nn.Module):
    NAME = "TRANSFER_MODEL_CLUSTERING"
    def __init__(self,  pretrained_feature_extractors:list, embed_dim=96):
        super(TransferModel,self).__init__()
        pretrained_modals = list(map(lambda x: os.path.join(os.curdir, x), pretrained_feature_extractors))

        self.acc_ft_ext = load_pretrained(list(filter(lambda x: "acc" in x, pretrained_modals))[0])
        self.gyro_ft_ext = load_pretrained(list(filter(lambda x: "gyro" in x, pretrained_modals))[0])

        self.acc_gated = Gated_Embedding_Unit(1024,1024)
        self.gyro_gated = Gated_Embedding_Unit(1024,1024)

    def forward(self, acc, gyro):
        acc_extracted = self.acc_ft_ext(acc)
        gyro_extracted = self.gyro_ft_ext(gyro)
        acc = self.acc_gated(acc_extracted)
        gyro = self.gyro_gated(gyro_extracted)
        return torch.concat((acc,gyro), dim=1)
        # return acc, gyro #experiement with concatentating each of these


class TransferModelClassification(nn.Module):
    def __init__(self, core_model, num_classes):
        super(TransferModelClassification, self).__init__()
        self.transfer_core = core_model
        self.classification =  nn.Linear(1024, num_classes)


    def forward(self, acc, gyro):
        features = self.transfer_core(acc,gyro)
        # return self.classification(acc)
        return self.classification(features)

