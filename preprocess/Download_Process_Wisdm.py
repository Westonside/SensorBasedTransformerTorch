import os
import re
import threading

from scipy.signal import resample
import numpy as np
# /static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip
import requests
import pandas as pd
from scipy.interpolate import interp1d

from preprocess.preprocess_utils import download_url, download_and_extract

link = "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"


# download_and_extract(["Wisdm"],[link], "../datasets" )
root_folder = "../datasets/extracted/wisdm-dataset/raw"




def prepare_data(data_file, modal_dict):
    data = pd.read_csv(data_file)
    # data = data.drop(columns=data.columns[0]) #drop their index
    data = data.to_numpy()
    modal_dict[os.path.basename(data_file)] = data




user_data = {}
user_labels = {}
for device in os.listdir(root_folder):
    if device.startswith('.'):
        continue
    # now that you have each device
    user_data[device] = {}
    user_labels[device] = {}
    modal_container = {}
    for modal in os.listdir(os.path.join(root_folder,device)):
        if modal.startswith('.'):
            continue
        # now we have the users
        for user in os.listdir(os.path.join(os.path.join(root_folder,device), modal)):
            if user.startswith('.'):
                continue
            if user_data[device].get(modal) is None:
                user_data[device][modal] = {}
                user_labels[device][modal] = {}
            user_num = re.search(r'_(\d+)_', user)
            if user_num:
                if user_data[device][modal].get(user_num.group(1)) is None:
                    user_data[device][modal][user_num.group(1)] = []
                    user_labels[device][modal][user_num.group(1)] = [] #ACC AND GYRO NOT ALIGNED
            else:
                exit(code=1)
            #now read the user data
            data = pd.read_csv(os.path.join(os.path.join(os.path.join(root_folder,device), modal),user))
            last_col = data.columns[-1]
            data[last_col] = data[last_col].str.replace(';','')
            data[last_col] = data[last_col].astype(float)
            data = data.to_numpy()
            user_data[device][modal][user_num.group(1)].append(data[:,-3:])
            user_labels[device][modal][user_num.group(1)].append(data[:,1])

for device in user_data:
    for modal in user_data[device]:
        for user in user_data[device][modal]:
            user_data[device][modal][user] = np.vstack(user_data[device][modal][user])
            user_labels[device][modal][user] = np.hstack(user_labels[device][modal][user])

resampled_data = {}
resampled_labels = {}
new_freq = 50
old_freq = 20
threads = []
lock = threading.Lock()

window_size = 0
for device in user_labels:
    # for each of the users compare the acc and the gyro


    users = user_labels[device][list(user_labels[device])[0]]
    for user in users: # for all the users
        classes_range = {}
        for modal in user_labels[device]: # here you will compare the acc and gyro to find the shortest segments
            pointer = 0
            segment_num = 0
            current_class = user_labels[device][modal][user][0]
            user_data_point = user_labels[device][modal][user]
            while(pointer < len(user_data[device][modal][user])):
                # print('testing')
                if(classes_range.get(segment_num) is None):
                    classes_range[segment_num] = []
                not_cur = np.where(user_data_point != current_class)[0]
                if (len([x for x in not_cur if x > pointer]) == 0):
                    print(pointer, len(user))
                    break;
                next_class = [x for x in not_cur if x > pointer][0]

                #     break

                indicies = range(pointer, next_class)
                # resampling time
                num_in_range = len(indicies)
                data_in_range = user_data[device][modal][user][indicies]  # the data in the range
                labels_in_range = user_labels[device][modal][user][indicies]
                new_samples = int(num_in_range * (new_freq / old_freq))
                not_current_count = np.count_nonzero(np.where(labels_in_range != current_class))
                percent_incorrect = not_current_count / num_in_range
                if not_current_count != 0:
                    print('not zero count!!')
                classes_range[segment_num].append(indicies)
                pointer = next_class
                current_class = user_labels[device][modal][user][next_class]
                segment_num +=1
        # the comparison time get the sum for each modal and then shorten the longest one
        smallest_col_holder = []
        for modal in user_labels[device]:
            print('testing')



print('testing')





def modal_resample(modal):
    for user_key in user_labels[modal]:
        user = user_labels[modal][user_key]
        user_container = []
        user_label_container = []
        pointer = 0
        current_class = user[0]
        while pointer != len(user)-1:
            not_cur = np.where(user != current_class)[0]
            if (len([x for x in not_cur if x > pointer]) == 0):
                print(pointer, len(user))
                break;
            next_class = [x for x in not_cur if x > pointer][0]

            #     break

            indicies = range(pointer, next_class)
            # resampling time
            num_in_range = len(indicies)
            data_in_range = user_data[modal][user_key][indicies] # the data in the range
            labels_in_range = user_labels[modal][user_key][indicies]
            new_samples = int(num_in_range * (new_freq / old_freq))
            not_current_count = np.count_nonzero(np.where(labels_in_range != current_class))
            percent_incorrect =  not_current_count / num_in_range
            if not_current_count != 0:
                print('not zero count!!')
            resampled_section = np.zeros((new_samples,3))
            for i in range(3):
                resampled_section[:,i] = resample(data_in_range[:,i], new_samples)
            if percent_incorrect < 0.2:
                user_container.append(resampled_section)
                user_label_container.append(np.full(new_samples, fill_value=current_class)) #TODO CHANGE THIS SO EACH MODALITY GETS THE SAME LABEL
            # if percent_correct > 0.8:
            current_class = user_labels[modal][user_key][next_class]
            pointer = next_class
        with lock:
            resampled_data[modal].append(np.vstack(user_container))
            resampled_labels[modal].append(np.hstack(user_label_container))


# def resampling(modal):
for modal in user_data:
    resampled_data[modal] = []
    resampled_labels[modal] = []
    t_modal = threading.Thread(target=modal_resample, args=(modal,))
    threads.append(t_modal)


for t in threads:
    t.start()
for t in threads:
    t.join()










segmented_acc = []
# now for each activity in the labels go 128, 256 and see how many segments you can make for each user
window_size = 128
stride = 64
segmented_data = {}
segmented_labels = {}
for device in user_data:
    for modal in user_data[device]: # get the modality key
        segmented_data[modal] = []
        segmented_labels[modal] = []
        for user in user_labels[modal].keys():
            user_container = []
            user_labels_container = []
            user_dat = user_labels[modal][user]
            current_class = user_dat[0]
            for i in range(0, user_dat.shape[0] - window_size, stride):
                indicies = range(i, i + window_size)

                segment = user_dat[indicies]
                actual_class = np.count_nonzero(segment == current_class) // len(segment)
                if actual_class > 0.8:
                    labels = np.full(128,fill_value=current_class)
                    user_labels_container.append(segment)
                    user_container.append(user_data[modal][user][indicies])
                elif actual_class < 0.4:
                    current_class = [val for val in np.unique(segment) if val != current_class][0]
            segmented_data[modal].append(user_container)
            segmented_labels[modal].append(user_labels_container)
    sampling_hz = 20.0
    target_hz = 50.0

    resampled_data = {}

    for modal in segmented_data.keys():
        for user_index in range(len(segmented_data[modal])):
            segmented_data[modal][user_index] = np.array(segmented_data[modal][user_index])

            segmented_labels[modal][user_index] = np.hstack(segmented_labels[modal][user_index])

        # num_samples =


test = 1





print('done')
