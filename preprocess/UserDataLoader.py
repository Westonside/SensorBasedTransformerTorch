from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

from preprocess.transformation_utils import transform_funcs_names


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


    def combine_training_validation(self):
        self.train = np.concatenate((self.train, self.validation), axis=0)
        self.train_label = np.concatenate((self.train_label, self.validation_label), axis=0)

    def transform_sets(self, transformation_functions: list):
        # this function will join the validation and the training together and then perform the transformations
        # then it will split the data back into the validation and training sets
        # self.train = np.concatenate((self.train, self.validation), axis=0)
        # self.train_label = np.concatenate((self.train_label, self.validation_label), axis=0)
        if hasattr(self, 'validation'):
            self.combine_training_validation()
            self.transform_train, self.transform_label = self.generate_transform_data(self.train, transformation_functions)
            # now that you have generated the transformations, you need to split the data back into the validation and training sets
            multi_task_transform = (self.transform_train, (map_multitask_y(self.transform_label, transform_funcs_names)))
            multi_task_split = multitask_train_test_split(multi_task_transform, test_size=0.1, random_seed=42) #split the tranformed data
            self.transform_train, self.transform_label, self.transform_validation, self.transform_validation_label = multi_task_split


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

def map_multitask_y(y, output_tasks):
    multitask_y = {} # create a dictionary
    for i, task in enumerate(output_tasks): # for the number of output labels that correspond to the transformation function
        multitask_y[task] = y[:, i] # add the value at the task to be all rows and select all values in column i and so multitask_y['noised']=[] this will return a 1D array that has a value that indicates if the coreresponding sample has had the corresponding transformation applied or not
    return multitask_y


def multitask_train_test_split(dataset, test_size=0.1, random_seed=42):
    dataset_size = len(dataset[0]) # get the size of the sensor data
    indices = np.arange(dataset_size) #creates array of ints from [0-dataset_size)
    np.random.seed(random_seed) #ccreate the seed so you can always have the same seed
    np.random.shuffle(indices) #shuffle the indicies
    test_dataset_size = int(dataset_size * test_size) #int of the total training set size using the split
    return dataset[0][indices[test_dataset_size:]], dict([(k, v[indices[test_dataset_size:]]) for k, v in dataset[1].items()]), dataset[0][indices[:test_dataset_size]], dict([(k, v[indices[:test_dataset_size]]) for k, v in dataset[1].items()])
