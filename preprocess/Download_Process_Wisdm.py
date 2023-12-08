import os
import re
import threading
import hickle as hkl
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
mapping = [
     0,
     7, # i am mapping jogging to running
     1, # upstairs
     3,
     4,
     13, #new key typing
     14, # brush teeth
     15, # soup
     16,
     17, # pasta
     18, #drinking
     19, #sandwich
     20, #kicking
     21, # catch
     22, # dribbling
     23, #writing
     24, # clapping
     25, # folding
]



def prepare_data(data_file, modal_dict):
    data = pd.read_csv(data_file)
    # data = data.drop(columns=data.columns[0]) #drop their index
    data = data.to_numpy()
    modal_dict[os.path.basename(data_file)] = data

def segment_data(seq, window_size, stride) -> list:
    segments = []
    for i in range(0,len(seq-window_size), stride):
        segments.append(seq[i:i+window_size])
    segments = [seg for seg in segments if seg.shape == (window_size, 6)]
    return segments



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





def modal_resample(device, modal):
    for user_key in user_labels[device][modal]:
        user = user_labels[device][modal][user_key]
        user_container = {}
        user_label_container = {}
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
            data_in_range = user_data[device][modal][user_key][indicies] # the data in the range
            labels_in_range = user_labels[device][modal][user_key][indicies]
            new_samples = int(num_in_range * (new_freq / old_freq))
            not_current_count = np.count_nonzero(np.where(labels_in_range != current_class))
            percent_incorrect =  not_current_count / num_in_range
            if not_current_count != 0:
                print('not zero count!!')
            resampled_section = np.zeros((new_samples,3))
            for i in range(3):
                resampled_section[:,i] = resample(data_in_range[:,i], new_samples)
            if percent_incorrect < 0.2:
                # add the values to the other dict
                if user_container.get(current_class) is None:
                    user_container[current_class] = []
                    user_label_container[current_class] = []
                user_container[current_class].append(resampled_section)
                user_label_container[current_class].append(np.full(new_samples, fill_value=current_class)) #TODO CHANGE THIS SO EACH MODALITY GETS THE SAME LABEL
            # if percent_correct > 0.8:
            current_class = user_labels[device][modal][user_key][next_class]
            pointer = next_class

        with lock: #here now I want to go through each activity
            resampled_data[device][modal].append(user_container)
            resampled_labels[device][modal].append(user_label_container)
            # resampled_data[device][modal].append(np.vstack(user_container))
            # resampled_labels[device][modal].append(np.hstack(user_label_container))


# def resampling(modal):
for device in user_data:
    resampled_data[device] = {}
    resampled_labels[device] = {}
    for modal in user_data[device]:
        resampled_data[device][modal] = []
        resampled_labels[device][modal] = []
        t_modal = threading.Thread(target=modal_resample, args=(device, modal,))
        threads.append(t_modal)


for t in threads:
    t.start()
for t in threads:
    t.join()





joined_device = {}
joined_device_labels = {}
# phone acc and gyro should go together
for device in resampled_data:
    # get all modals
    joined_device[device] = {}
    joined_device_labels[device] = {}
    acc_resampled = resampled_labels[device]['accel']
    gyro_resampled = resampled_labels[device]['gyro']
    for user_idx in range(len(acc_resampled)):
        print('testing')
        joined_device[device][user_idx] = []
        joined_device_labels[device][user_idx] = []
        user_acc = acc_resampled[user_idx]
        user_gyro = gyro_resampled[user_idx]
        # now compare each class between accel and gyro
        for class_val in user_acc:
            #we do not want the class if there is not a corresponding reading on the other modality
            if user_acc.get(class_val) is None or user_gyro.get(class_val) is None:
                continue
            accel_classes_instances = user_acc[class_val][0]
            gyro_classes_instances = user_gyro[class_val][0]
            smallest_device = min([len(accel_classes_instances), len(gyro_classes_instances)])

            short_data_acc = resampled_data[device]['accel'][user_idx][class_val][0][0:smallest_device]
            short_labels_acc = accel_classes_instances[0:smallest_device]

            short_data_gyro = resampled_data[device]['gyro'][user_idx][class_val][0][0:smallest_device]
            joined_device[device][user_idx].append(np.hstack((short_data_acc, short_data_gyro)))
            joined_device_labels[device][user_idx].append(short_labels_acc)



segmented_acc = []
# now for each activity in the labels go 128, 256 and see how many segments you can make for each user
window_size = 128
stride = 64
segmented_data = {}
segmented_labels = {}
# go through each device and segment them using window size and stride
for device in joined_device:
    segmented_data[device] ={}
    segmented_labels[device] ={}
    for user_idx in range(len(joined_device[device])): # for each of the users you will now do the segmentation
        segmented_data[device][user_idx] = []
        segmented_labels[device][user_idx] = []
        data_holder = []
        label_holder = []
        for task_id, activity in enumerate(joined_device[device][user_idx]):
            segment = segment_data(activity, window_size, stride)
            data_holder.append(segment)
            label_holder.append(np.full(len(segment), fill_value=mapping[task_id]))
        segmented_data[device][user_idx] = np.vstack(data_holder)
        segmented_labels[device][user_idx] = np.hstack(label_holder)





res_data = list([np.vstack([*x]) for x in zip(segmented_data["phone"].values(), segmented_data["watch"].values())])
res_labels =list([np.hstack([*x]) for x in zip(segmented_labels["phone"].values(), segmented_labels["watch"].values())])

#now concat the watch on the phone
total = {
    "data": res_data,
    "labels": res_labels
}
hkl.dump(total, "./processed_wisdm.hkl")


#get all devices




test = 1




print('done')
