import os

import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity

from model_impl.model import load_pretrained, Gated_Embedding_Unit, MultiModalTransformer

global_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransferModel(nn.Module):
    NAME = "TRANSFER_MODEL_CLUSTERING"
    def __init__(self,  pretrained_feature_extractors:list, clustering_model, embed_dim=128):
        super(TransferModel,self).__init__()
        pretrained_modals = list(map(lambda x: os.path.join(os.curdir, x), pretrained_feature_extractors))

        self.acc_ft_ext = load_pretrained(list(filter(lambda x: "acc" in x, pretrained_modals))[0])
        self.gyro_ft_ext = load_pretrained(list(filter(lambda x: "gyro" in x, pretrained_modals))[0])

        # need to reload
        self.acc_gated, self.gyro_gated = setup_clustering_gated(clustering_model, extractors=pretrained_feature_extractors)

    def forward(self, acc, gyro):
        acc_extracted = self.acc_ft_ext(acc)
        gyro_extracted = self.gyro_ft_ext(gyro)
        acc = self.acc_gated(acc_extracted)
        gyro = self.gyro_gated(gyro_extracted)
        return torch.concat((acc,gyro), dim=1)


class TransferModelClassification(nn.Module):
    def __init__(self, core_model, num_classes):
        super(TransferModelClassification, self).__init__()
        self.transfer_core = core_model
        self.classification =  nn.Linear(1024, num_classes)


    def forward(self, acc, gyro):
        features = self.transfer_core(acc,gyro)
        # return self.classification(acc)
        return self.classification(features)



def setup_clustering_gated(path, extractors):
    model = MultiModalTransformer((128,6), extractors, 2)
    model.load_state_dict(torch.load(path))
    return model.acc_gated, model.gyro_gated