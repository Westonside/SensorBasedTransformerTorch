import os
import re

import hickle as hkl
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import utils.data_shortcuts
from UserDataLoader import UserDataLoader

"""
    Pass in a list of the form [val..valn] where val in [0,1] 1 meaning the dataset is to be loaded 0 otherwise not
"""
def load_datasets(container: list, path='../datasets/processed', validation=True):
    datasets = []
    UCI = [0, 1, 2, 3, 4, 5]
    REALWORLD_CLIENT = [2, 1, 6, 5, 7, 3, 4, 0]
    HHAR = [3, 4, 0, 1, 2, 8]
    Motion_Sense = [2, 1, 3, 4, 0, 7]
    SHL = [4, 0, 7, 8, 9, 10, 11, 12]


    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []
    # orientation_data = []
    # orientation_labels = []
    selected_datasets = []
    align_all_classes  = ['Walk', 'Upstair', 'Downstair', 'Sit', 'Stand', 'Lay', 'Jump', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']

    for i, dataset in enumerate(os.listdir(path)):
        if dataset in container:
            # if you are to load the dataset
            selected_datasets.append(dataset)
            loader, clients = dataset_classes_users_map[dataset]
            train, test, orientation = loader(os.path.join(path, dataset))
            training_data.append(train[0])
            training_labels.append(train[1])
            testing_data.append(test[0])
            testing_labels.append(test[1])
            # orientation_data.append(orientation[0])
            # orientation_labels.append(orientation[1])

    central_train_label_align = []
    central_test_label_align = []
    for i, dataset in enumerate(selected_datasets):
        if dataset == "MotionSense": #this converts the label to the correct label
            central_train_label_align.append([Motion_Sense[labelIndex] for labelIndex in training_labels[i]])
            central_test_label_align.append([Motion_Sense[label_index] for label_index in testing_labels[i]])
        elif (dataset == 'HHAR'):
            central_train_label_align.append(np.hstack([HHAR[labelIndex] for labelIndex in training_labels[i]]))
            central_test_label_align.append(np.hstack([HHAR[labelIndex] for labelIndex in training_labels[i]]))

    training_data = np.vstack(training_data)
    training_labels = np.hstack(central_train_label_align)
    testing_data = np.vstack(testing_data)
    testing_labels = np.hstack(central_test_label_align)


    # now that that data is loadded and aligned we can now put the data into the dataset as one hot encoding
    unique_labels = np.unique(training_labels)
    num_classes = unique_labels.size
    one_hot_training_labels = torch.nn.functional.one_hot(torch.from_numpy(training_labels),
                                                      num_classes=len(align_all_classes)).numpy()
    central_test_label = torch.nn.functional.one_hot(torch.from_numpy(testing_labels),
                                                     num_classes=len(align_all_classes)).numpy()

    # now that the data is one hot encoded we can now create the dataset
    if validation:
        X_train, X_val, y_train, y_val = train_test_split(training_data, one_hot_training_labels, test_size=0.10)

        dataset = UserDataLoader((X_train, y_train), (testing_data, central_test_label), align_all_classes, validation_data=(X_val, y_val))
    else:
        dataset = UserDataLoader((training_data, one_hot_training_labels), (testing_data, central_test_label), align_all_classes)
    return dataset




"""
    Client data is loaded from the data and folds are generated for each client
    The client's data will be stacked on top of each other and the labels will be stacked on top of each other as well
    There will be no shuffling of data to avoid data leakage
    K Folds is generated for each client to ensure a better representation of the data by having more balanced data
"""
def load_data(client_data:list, client_labels:list, test_split:int, valid_split=None, orientation_data=None):
   num_test_users = int(len(client_data) * test_split)
   train, test = best_samples(client_data, client_labels,num_test_users, num_validation=valid_split)
   return train, test, ()

def best_samples(user_data, user_labels, num_test_users:int, num_validation=None):
    #the validation can be users from the user set
    # the validation can be taken during training set
    # you can use k folds
    # you can use train test split for validation
    test_labels = []
    test_data = []
    selected = []
    class_occur_per_user = [np.mean([np.count_nonzero(classes==user) for classes in np.unique(user)]) for user in user_labels] # this is the number of occurances of each class per user
    # num_samples_per_user =  [user.shape[0] for user in user_data]
    # now we will take num_test users
    for _ in range(num_test_users):
        user_index = np.argmax(class_occur_per_user) # choose the user with the best class distribution
        test_data.append(user_data[user_index])
        test_labels.append(user_labels[user_index])
        class_occur_per_user[user_index] = 0 # set the user to 0 so that it is not chosen again
        selected.append(user_index)

    training_data = [user_data[i] for i in range(len(user_data)) if i not in selected]
    training_labels = [user_labels[i] for i in range(len(user_labels)) if i not in selected]
    return (training_data, training_labels), (test_data, test_labels)




def load_clients_data(path: str, num_clients: int):
    client_data = []
    client_labels = []
    for i in range(num_clients): # add the client data
        pat = re.compile(rf'\D{i}\D')  # match the client number
        data_file = [file for file in os.listdir(path) if "data" in file.lower() and pat.search(file)][0]
        data_label = [file for file in os.listdir(path) if "label" in file.lower() and pat.search(file)][0]

        client_data.append(hkl.load(os.path.join(path, data_file)))
        client_labels.append(hkl.load(os.path.join(path, data_label)))
        # print(data_label, data_file)
    # print('done loading data')
    return client_data, client_labels


def load_hhar(path):
    # get the data
    client_data, client_labels = load_clients_data(path, dataset_classes_users_map["HHAR"][1])
    orientations = hkl.load([os.path.join(path,file) for file in os.listdir(path) if "device" in file.lower()][0])
    orientationsNames = ['nexus4', 'lgwatch', 's3', 's3mini', 'gear', 'samsungold']
    train, test, orientation = load_data(client_data, client_labels, 0.1, 0.1)

    # for i in range(dataset_classes_users_map["HHAR"][1]): # for all clients
    #     client_orientation_train[i] = orientations[orientationsNames[i]][client_orientation_train[i]]
    #     client_orientation_test[i] = orientations[orientationsNames[i]][client_orientation_test[i]]


    train_data,train_labels, test_data, test_labels = utils.data_shortcuts.stack_train_test_orientation(train, test)
    return (train_data, train_labels), (test_data, test_labels), ()





def load_motion_sense(path: str):
    client_data, client_labels = load_clients_data(path, dataset_classes_users_map["MotionSense"][1])
    train, test, client_orientation= load_data(client_data, client_labels, 0.1,0.1 )
    print('testing')
    train_data, train_labels, test_data, test_labels = utils.data_shortcuts.stack_train_test_orientation(train, test)
    return (train_data, train_labels), (test_data, test_labels), client_orientation
def load_realworld(): # for now I ignore this because there are differing orientations
    pass


def load_uci():
    pass


dataset_classes_users_map ={
    #the map will contain a function to load the datset and the number of users
    "HHAR": (load_hhar, 51),
    "MotionSense": (load_motion_sense, 24),
    # "RealWorld": (load_realworld, 15)
    "UCI": (load_uci, 5)
}

dataset_training_classes = {
    "HHAR": [],
    "MotionSense": ['Downstairs', 'Upstairs', 'Sitting', 'Standing', 'Walking', 'Jogging'],
    "RealWorld": ['Downstairs','Upstairs', 'Jumping','Lay', 'Running', 'Sitting', 'Standing', 'Walking'],
    "UCI": []


}




if __name__ == '__main__':
    load_datasets(["MotionSense"])
    pass