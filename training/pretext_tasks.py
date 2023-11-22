"""
    This function performs the first pretext task of training a transformer on each modality to perform transformation classification
    all datasets will be combined to perform the classification
"""
import inspect
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from UserDataLoader import UserDataLoader
from model import TransformerMultiTaskBinaryClassificationModel, TransformerClassificationModel
from utils.model_utils import train_epoch, MultiTaskLoss, SingleClassificationFormatter, BinaryClassificationFormatter
from utils.transformation_utils import transform_funcs_vectorized, transform_funcs_names


def pretext_one():
    print('pretext one')


class Training_Task:
    # this will either load the model or create the model
    def __init__(self, dataset, save_path="./", previous_task_path=None, epochs=80, early_stop=False):
        self.model = None
        self.dataset = dataset
        self.save_path = save_path
        self.epochs = epochs
        self.model = None
        if previous_task_path is None:
            self.create_model()
        else:
            self.load_model(previous_task_path)
        pass

    def create_model(self):
        pass

    def get_model(self):
        return self.model

    def load_model(self, configuration_path):
        self.model = torch.load(configuration_path)

    def get_save_file_name(self) -> str:
        pass

    def save_model(self):
        torch.save(self.model, os.path.join(self.save_path, self.get_save_file_name()))

    def train_task_setup(self):
        pass



    def get_output_formatter(self):
        pass


    def get_training_data(self):
        pass

    def train(self):
        # move the device to gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.train_task_setup()  # will set up the training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        training, training_label = self.get_training_data()
        for epoch in range(1, self.epochs + 1):
            train_epoch(self.model, epoch, training, training_label, output_formatter=self.get_output_formatter(),
                        optimizer=optimizer, loss_fn=self.get_loss())
            # self.save_model()
        self.save_model()


    def get_loss(self):
        pass


class Classification_Task(Training_Task):
    TASK_NAME = "Classification_Task"
    def __init__(self, dataset: UserDataLoader, save_path="./", previous_task_path=None, epochs=80, early_stop=False):
        super().__init__(dataset, save_path=save_path, previous_task_path=previous_task_path, epochs=epochs)
        self.dataset = dataset
        self.model = None
        self.create_model()

    def create_model(self):
        self.model = TransformerClassificationModel((128,6), 13, modal_count=2)

    def train_task_setup(self):
        pass


    def get_training_data(self):
        return self.dataset.train, self.dataset.train_label

    def get_output_formatter(self):
        return SingleClassificationFormatter()


    def get_loss(self):
        return nn.CrossEntropyLoss()


class Transformation_Classification_Task(Training_Task):
    TASK_NAME = "Transformation_Classification_Task"
    def __init__(self, dataset: UserDataLoader, modal_range: range, save_path="./", previous_task_path=None, epochs=80, early_stop=False):
        super().__init__(dataset, save_path=save_path, previous_task_path=previous_task_path, epochs=epochs)
        self.modal_range = modal_range
        self.dataset = dataset
        self.model = None
        self.transformations = transform_funcs_vectorized
        self.create_model()


    def train_task_setup(self):
        # this will set up the trainin
        self.dataset.keep_modalities(self.modal_range)  # keep the modalities
        self.dataset.transform_sets(self.transformations)  # transform the data

    def create_model(self):
        #TODO: ALLLOW PASSING IN OTHER  transformations
        #TODO: remove the magic number
        self.model = TransformerMultiTaskBinaryClassificationModel((128,3), len(transform_funcs_vectorized))


    def get_training_data(self):
        return self.dataset.transform_train, self.dataset.transform_label


    def get_loss(self):
        return MultiTaskLoss(len(self.transformations))


    def get_output_formatter(self):
        return BinaryClassificationFormatter(len(self.transformations), transform_funcs_names)

def match_configuration(config, key):
    if config.get(key) is None:
        print(f"configuration does not have a {key}")
        return None
    config = config[key]
    config = config.lower()
    if PRETEXT_TASKS.get(config) is not None:
        return PRETEXT_TASKS.get(config)
    else:
        print("Not found!")
        return None

PRETEXT_TASKS = {
    "transformation_classification": Transformation_Classification_Task
}