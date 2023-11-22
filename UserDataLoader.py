import itertools
from typing import Tuple

import numpy as np
from sympy.physics.quantum.identitysearch import scipy
from torch.utils.data import Dataset, DataLoader

class UserDataLoader(Dataset):
    def __init__(self, train_data,  test_data,  classes, validation_data=None):
        self.train, self.train_label = train_data
        self.test, self.test_label = test_data
        self.classes = classes
        #TODO: perform the transformations
        if validation_data is not None:
            self.validation, self.validation_label = validation_data
        self.transform_train = None
        self.transform_label = None
        self.transform_validation = None
        self.transform_validation_label = None



    def transform_sets(self, transformation_data_target: int, transformation_functions: list):
        if transformation_data_target > 2:
            raise ValueError("The transformation data target must be less than 2")
        if transformation_data_target == 0:
            self.train,self.train_label = self.generate_transform_data(self.train, transformation_functions)
        if transformation_data_target == 1:
            self.test,self.test_label = self.generate_transform_data(self.test, transformation_functions)
        if transformation_data_target == 2:
            self.validation,self.validation_label = self.generate_transform_data(self.validation, transformation_functions)

    def generate_transform_data(self, data, transformation_functions: list, num_copies=1):
        # this will go through the data and then apply the transformations
        num_transformations = len(transformation_functions)
        transform_x = []
        transform_y = []
        for _ in range(num_copies): # the number of times you want to create the transformations
            transform_x.append(data)
            y_transforms = np.zeros((len(data), num_transformations),dtype=int) # create arrays to be the labels for all transformations
            transform_y.append(y_transforms) # add the transformations
            for i, transform in enumerate(transformation_functions):
                transform_x.append(transform(data))
                y_transforms = np.zeros((len(data), num_transformations),dtype=int) # create the labels for the transformation
                y_transforms[:,i] = 1 # mark that the transformation was applied
                transform_y.append(y_transforms)

        return (np.concatenate(transform_x, axis=0), np.concatenate(transform_y, axis=0)) # (data,label)





    def transform_train_done(self):
        # check if the obejct has the attribute self.transform_train and self.transform_label
        if hasattr(self, 'transform_train') and hasattr(self, 'transform_label'):
            del self.transform_train
            del self.transform_label
            if hasattr(self, 'transform_validation'):
                del self.transform_validation




    def __getitem__(self, index: int) -> Tuple:
        pass

    def keep_modalities(self, modal_range):
        self.train = self.train[:,:,modal_range]
        self.test = self.test[:,:,modal_range]
        if hasattr(self, 'validation'):
            self.validation = self.validation[:,:,modal_range]


