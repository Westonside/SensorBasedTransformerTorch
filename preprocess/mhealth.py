import copy
import os
import numpy as np
import pandas as pd
import hickle as hkl
from preprocess.preprocess_utils import download_and_extract

user_label_mappings = {
    "0": 4,
    "1": 3,
    "2": 26,
    "3": 27,
    "4": 28,
    "5": 5,
    "6": 29,
    "7": 30,
    "8": 6,
    "9": 31,
    "10": 32,
    "11": 0,
    "12": 33,
    "13": 34,
    "14": 7,
    "15": 1,
    "16": 2,
    "17": 35
}

def segment_data(seq, window_size, stride) -> list:
    segments = []
    for i in range(0,len(seq-window_size), stride):
        segments.append(seq[i:i+window_size])
    segments = [seg for seg in segments if seg.shape == (window_size, 6)]
    return segments

def downsample_data(signal, current_freq, new_freq):
    scale = new_freq / current_freq
    n = round(len(signal) * scale)
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),
        np.linspace(0.0, 1.0, len(signal), endpoint=False),
        signal,
    )
    return resampled_signal

def load_user_file(file_path: str):
    data = pd.read_csv(file_path, delimiter=',', header=None)
    df = data.drop(data.columns[0], axis=1)
    df = df.drop(data.columns[3], axis=1).astype(float) # this will now leave it in terms of only accel and gyro
    df = df.reset_index(drop=True)
    val = df.to_numpy()
    resampled = np.dstack([downsample_data(val[:,x], 100,50) for x in range(val.shape[1])])[0]
    return resampled


link = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/45f952y38r-5.zip'
download_and_extract(['kuhar'], [link], '../datasets/')


user_data = {}
dir = '../datasets/extracted/kuhar/interp'

for activity in os.listdir(dir):
    cur = os.path.join(dir,activity)
    for user in os.listdir(cur):
        user_id = user[:user.find('_')] # get the user id
        if user_data.get(user_id) is None:
            user_data[user_id] = {}

        if user_data[user_id].get(activity) is None:
            user_data[user_id][activity] = []
        # now we are getting the user data
        data = load_user_file(file_path=os.path.join(cur,user))
        # now that we have the resampled data
        user_data[user_id][activity].append(data)

# now stack all activities and generate the labels
for user in user_data:
    for activity in user_data[user]:
        values = np.vstack(user_data[user][activity])
        # now you can segment the data
        segmented = segment_data(values, 128, 64)
        print('segmented')
        user_data[user][activity] = np.array(segmented)

# now translate the labels and create a label array


print('testing')
user_labels = []
stacked_user_data = []


for i, user in enumerate(user_data):
    user_holder_data = []
    user_holder_labels = []
    for activity in user_data[user]:
        data = user_data[user][activity]
        key = activity[:activity.find(".")]
        user_holder_labels.append(np.full(data.shape[0], fill_value=user_label_mappings[key]))
        user_holder_data.append(data)
    user_labels.append(user_holder_labels)
    stacked_user_data.append(user_holder_data)
print('testing')

# find the users with only 4 activities or less
# first find the largest
all = [len(x) for x in stacked_user_data]
median = int(np.median(all))-2 # get the median and you will try to get as many medians

selected = [(i,x ) for i, x in enumerate(all) if x <= median]
combo_holder = []
current = []
for i in selected:
    if sum(list(map(lambda x: x[1], current))) >= median:
       combo_holder.append(copy.deepcopy(current))
       current = []
    current.append(i) #
# now that we have the combinations combine those users
remove_list = []
for combination in combo_holder:
    data = np.vstack([np.vstack(stacked_user_data[x[0]]) for x in combination])
    labels = np.hstack([np.hstack(user_labels[x[0]]) for x in combination])
    # stacked_user_data = [x for i,x in enumerate(stacked_user_data) if i not in list(map(lambda x: x[0], combination))]
    # user_labels = [x for i,x in enumerate(user_labels) if i not in list(map(lambda x: x[0], combination))]

    stacked_user_data.append(data)
    user_labels.append(labels)
    for val in combination:
        remove_list.append(val)

stacked_user_data = [np.vstack(x) if isinstance(x,list) else x for i,x in enumerate(stacked_user_data) if i not in list(map(lambda x: x[0], remove_list))]
user_labels = [np.hstack(x) if isinstance(x,list) else x for i,x in enumerate(user_labels) if i not in list(map(lambda x: x[0], remove_list))]

print('testing')
res = {
    "data": stacked_user_data,
    "labels": user_labels
}
hkl.dump(res,'./processed_kuhar.hkl')