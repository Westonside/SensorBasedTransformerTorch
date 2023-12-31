import os
import re

import hickle as hkl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import preprocess.data_shortcuts
from preprocess.UserDataLoader import UserDataLoader

"""
    IMPORTANT!! 
    This file takes inspiration from the following github repo:
    https://github.com/getalp/Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices
    
    
    All functions prefixed by load are used to load the data from the dataset
"""
def load_datasets(container: list, path='../datasets/processed', validation=True, balance_sets=False):
    datasets = []
    UCI = [0, 1, 2, 3, 4, 5]
    HHAR = [3, 4, 0, 1, 2, 8]
    Motion_Sense = [2, 1, 3, 4, 0, 7] # these provide the indicies
    SHL = [4, 0, 7, 8, 9, 10, 11, 12]
    PAMAP = {6: 5, 3: 3, 4: 4, 5: 0, 2: 7, 7: 8, 8: 26, 9:27}

    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []
    # orientation_data = []
    # orientation_labels = []
    selected_datasets = []
    #                      0        1           2           3       4       5       6       7       8    9       10      11     12        13        14              15     16      17       18          19          20
    align_all_classes  = ['Walk', 'Upstair', 'Downstair', 'Sit', 'Stand', 'Lay', 'Jump', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway', 'Typing', 'BrushTeeth', 'Soup', 'Chips', 'Pasta', 'Drinking','Sandwich', 'Kicking',
                           # 21     22             23        24         25            27         28
                          'Catch', 'Dribbling', 'Writing', 'Clapping', 'Folding', 'Nordic Walk' ,'Jump Rope']

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
            central_test_label_align.append(np.hstack([HHAR[labelIndex] for labelIndex in testing_labels[i]]))

        elif dataset == "UCI":
            central_train_label_align.append(np.hstack([UCI[labelIndex] for labelIndex in training_labels[i]]))
            central_test_label_align.append(np.hstack([UCI[labelIndex] for labelIndex in testing_labels[i]]))

        elif dataset == "SHL":
            central_train_label_align.append(np.hstack([SHL[labelIndex] for labelIndex in training_labels[i]]))
            central_test_label_align.append(np.hstack([SHL[labelIndex] for labelIndex in testing_labels[i]]))
        elif dataset == "PAMAP":
            central_train_label_align.append(np.hstack([PAMAP[labelIndex] for labelIndex in training_labels[i]]))
            central_test_label_align.append(np.hstack([PAMAP[labelIndex] for labelIndex in testing_labels[i]]))
        else: # already label translated wisdm kuhar
            central_train_label_align.append(np.hstack([x for x in training_labels[i]]))
            central_test_label_align.append(np.hstack([x for x in testing_labels[i]]))
            print('testing')

    training_data = np.vstack(training_data)
    training_labels = np.hstack(central_train_label_align)
    testing_data = np.vstack(testing_data)
    testing_labels = np.hstack(central_test_label_align)


    # now that that data is loadded and aligned we can now put the data into the dataset as one hot encoding
    unique_labels = np.unique(training_labels)
    num_classes = unique_labels.size

    training_labels, testing_labels, new_aligned_classes = remap_classes(training_labels,testing_labels, align_all_classes)


    one_hot_training_labels = torch.nn.functional.one_hot(torch.from_numpy(training_labels),
                                                      num_classes=len(new_aligned_classes)).numpy()
    central_test_label = torch.nn.functional.one_hot(torch.from_numpy(testing_labels),
                                                     num_classes=len(new_aligned_classes)).numpy()

    # now that the data is one hot encoded we can now create the dataset
    if validation: # it is importnat to note that the testing data has already been generated to avoid data leakage
        X_train, X_val, y_train, y_val = train_test_split(training_data, one_hot_training_labels, test_size=0.10)

        dataset = UserDataLoader((X_train, y_train), (testing_data, central_test_label), new_aligned_classes, validation_data=(X_val, y_val))
    else:
        dataset = UserDataLoader((training_data, one_hot_training_labels), (testing_data, central_test_label), new_aligned_classes)
    return dataset


def remap_classes(train_labels, test_labels, total_labels):
    # get the unique labels
    unique_labels = list(np.unique(np.concatenate((train_labels,test_labels))))
    new_labels = [total_labels[i] for i in unique_labels]

    # create a mapping from unique_labels to a new set of labels with a difference of 1
    label_mapping = {label: i for i, label in enumerate(unique_labels)}

    # replace the old labels with the new labels in train_labels and test_labels
    train_labels = np.array([label_mapping[label] for label in train_labels])
    test_labels = np.array([label_mapping[label] for label in test_labels])

    return train_labels, test_labels, new_labels



"""
    Client data is loaded from the data and folds are generated for each client
    The client's data will be stacked on top of each other and the labels will be stacked on top of each other as well
    There will be no shuffling of data to avoid data leakage and data is split by user
"""
def load_data(client_data:list, client_labels:list, test_split:int, total_classes:int, valid_split=None, orientation_data=None):
   num_test_users = int(len(client_data) * test_split)
   train, test = random_user_selection(client_data, client_labels,num_test_users, total_classes)
   return train, test, ()

def random_user_selection(user_data, user_labels, num_test_users, total_classes):
    users = set({})
    test_data = []
    test_labels = []
    while(len(users) != num_test_users):
        user = np.random.randint(0, len(user_data))
        users.add(user)
        test_data.append(user_data[user])
        test_labels.append(user_labels[user])

    # now you will get the users from the dict
    train_data = [user_data[x] for x in range(len(user_data))if x not in users]
    train_labels = [user_labels[x] for x in range(len(user_labels)) if x not in users]
    return (train_data,train_labels), (test_data, test_labels)

# code for splitting by user into test and train
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

    train, test, orientation = load_data(client_data, client_labels, 0.2, len(dataset_training_classes["HHAR"]))
    train_data,train_labels, test_data, test_labels = preprocess.data_shortcuts.stack_train_test_orientation(train, test)
    return (train_data, train_labels), (test_data, test_labels), ()





def load_motion_sense(path: str):
    client_data, client_labels = load_clients_data(path, dataset_classes_users_map["MotionSense"][1])
    train, test, client_orientation= load_data(client_data, client_labels, 0.2,len(dataset_training_classes["MotionSense"]))
    print('testing')
    train_data, train_labels, test_data, test_labels = preprocess.data_shortcuts.stack_train_test_orientation(train, test)
    return (train_data, train_labels), (test_data, test_labels), client_orientation



def load_uci(path: str):
    central_train_data = hkl.load(os.path.join(path,'trainX.hkl'))
    central_test_data = hkl.load(os.path.join(path,'testX.hkl'))
    central_train_label = hkl.load(os.path.join(path, 'trainY.hkl'))
    central_test_label = hkl.load(os.path.join(path,'testY.hkl'))

    return (central_train_data, central_train_label), (central_test_data,central_test_label), ()
    pass

def load_shl(path: str):
    client_data = os.path.join(path, [x for x in os.listdir(path) if "data" in x.lower()][0])
    client_labels = os.path.join(path, [x for x in os.listdir(path) if "label" in x.lower()][0])
    client_data = hkl.load(client_data)
    client_labels = hkl.load(client_labels)

    client_data_train = {i: data for i, data in enumerate(client_data)}
    client_train_label = {i: label for i, label in enumerate(client_labels)}
    train, test, orientation = load_data(client_data_train, client_train_label, 0.2, len(dataset_training_classes["SHL"]))
    train_data, train_labels, test_data, test_labels = preprocess.data_shortcuts.stack_train_test_orientation(train, test)
    return (train_data, train_labels), (test_data, test_labels), ()

def load_wisdm(path: str):
    vals = os.listdir(path)
    client_data = hkl.load(os.path.join(path,vals[0]))
    data = client_data['data']
    labels = client_data['labels']
    train, test, client_orientation = load_data(data, labels, 0.2,
                                                len(dataset_training_classes["WISDM"]))


    train_data, train_labels, test_data, test_labels = preprocess.data_shortcuts.stack_train_test_orientation(train, test)
    return (train_data, train_labels), (test_data, test_labels), ()


def load_pamap(path: str):
    vals = os.listdir(path)
    client_data = hkl.load(os.path.join(path, vals[0]))
    client_labels = hkl.load(os.path.join(path,vals[1]))
    train, test, client_orientation = load_data(client_data, client_labels, 0.2,
                                                9)

    train_data, train_labels, test_data, test_labels = preprocess.data_shortcuts.stack_train_test_orientation(train,
                                                                                                              test)
    return (train_data, train_labels), (test_data, test_labels), ()

dataset_classes_users_map ={
    #the map will contain a function to load the datset and the number of users
    "HHAR": (load_hhar, 51),
    "MotionSense": (load_motion_sense, 24),
    # "RealWorld": (load_realworld, 15)
    "UCI": (load_uci, 5),
    "SHL": (load_shl, 3),
    "WISDM": (load_wisdm, 51),
    "PAMAP": (load_pamap, 9)
}

dataset_training_classes = {
    "HHAR": [],
    "MotionSense": ['Downstairs', 'Upstairs', 'Sitting', 'Standing', 'Walking', 'Jogging'],
    "UCI": ['Walking', 'Upstair','Downstair', 'Sitting', 'Standing', 'Lying'],
    "SHL": ['Standing','Walking','Runing','Biking','Car','Bus','Train','Subway'],
    "WISDM": ["Walking", "Jogging", "Upstairs", "Sitting", "Standing", "Typing", "Teeth", "Soup", "Chips", "Pasta", "Drinking", "Sandwich", "Kicking", "Catch", "Dribbling", "Writing", "Clapping", "Folding" ],
    "PAMAP": ['lay',
'sit',
'stand',
'walk',
'run',
'bike',
'nordic-walk',
              'jumprope']
}




if __name__ == '__main__':
    load_datasets(["PAMAP"])
    pass