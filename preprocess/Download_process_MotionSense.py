import argparse
import os
import hickle as hkl
import numpy as np

#from preprocess import preprocess_utils
import preprocess_utils
from preprocess_utils import download_and_extract
#from preprocess.preprocess_utils import  download_and_extract
import pandas as pd

file_name = ["A_DeviceMotion_data"]
links = ["https://github.com/mmalekzadeh/motion-sense/blob/master/data/A_DeviceMotion_data.zip?raw=true"]

def match_label(activityFileName):
    if "dws" in activityFileName:
        return 0
    elif "ups" in activityFileName:
        return 1
    elif "sit" in activityFileName:
        return 2
    elif "std" in activityFileName:
        return 3
    elif "wlk" in activityFileName:
        return 4
    elif "jog" in activityFileName:
        return 5
    else:
        print("Not found!")
        return None

def segment_client_file(client_file, client_index, client_dict: dict, time_step, step: int):
    df = pd.read_csv(client_file,  sep=',')
    df = df.drop(columns=df.columns[0]) #drop their index
    acc = df[['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']].to_numpy()
    gyr = df[['rotationRate.x', 'rotationRate.y', 'rotationRate.z']].to_numpy()
    # mag = df[['attitude.roll', 'attitude.pitch', 'attitude.yaw']].to_numpy() #TODO add the mag in later

    # now hstack the acc and gyr
    acc_gyr = np.hstack((acc, gyr))
    # now segment the data
    #TODO MAKE IT SO THAT ALL SEQUENCES ARE THE SAME LENGTH BUT PADDING IS ADDED TO THE END SO THAT THE TRANSFORMER IGNORES IT
    segments = []
    for i in range(0, acc_gyr.shape[0] - time_step, step): # time step is 128 step is 64
       segments.append(acc_gyr[i:i + time_step, :])  # append the data to the segment array
    segments_res = np.asarray(segments)
    return segments_res.astype(np.float32)



def main():
    """
    The data format is as follows
    1. 24 subjects
    2. 6 activities
    [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z] zscore normalized

    :return:
    """
    modals = ['acc', 'gyr']
    args = preprocess_utils.get_parser()  # get the args
    # first create the directory for data
    download_and_extract(file_name, links, "../datasets/")
    extracted_directory = os.path.abspath("../datasets/extracted/A_DeviceMotion_data")

    # sort the directories in ascending order
    dirs = np.sort([d for d in os.listdir(extracted_directory) if os.path.isdir(os.path.join(extracted_directory, d))])

    num_subjects = len(os.listdir(os.path.join(extracted_directory,dirs[0]))) # get the number of subjects

    client_data = {iniArray: [] for iniArray in range(num_subjects)} # this will create an array for the 24 clients
    client_labels = {iniArray: [] for iniArray in range(num_subjects)} # create labels for the clients

    for activity_index, activity_file in enumerate(dirs):
        # for each activity file
        subject_file_names = sorted(os.listdir(extracted_directory + "/" + activity_file)) #sort the subjects
        for client_index, client_file in enumerate(subject_file_names): # this will the subject file name
            segmented = segment_client_file(os.path.join(extracted_directory, activity_file, client_file), client_index, client_data, 128, 64)
            client_data[client_index].append(segmented)
            # now append the label
            client_labels[client_index].append(np.full(segmented.shape[0], match_label(activity_file), dtype=int))
    print('done creating the labels and data')
        # for all of the clients
        # for client_index, client_file in enumerate(subject_file_names):

    all_user_data = np.vstack([np.vstack(client_data[cli_ind]) for cli_ind in client_data.keys()])
    all_user_labels = [np.hstack(client_labels[cli_ind]) for cli_ind in client_labels.keys()]
    print('joined')

    holder = []
    for tri_modal in range(0,len(modals)*3, 3):
        modal = all_user_data[:,:, tri_modal:tri_modal+3]
        # get the mean and std
        mean = np.mean(modal)
        std = np.std(modal)
        # now normalize the data
        modal = (modal - mean) / std # normalize the data with z score
        holder.append(modal)

    combined_normalized_data = np.dstack(holder)
    print('done normalizing')

    start_ind = 0
    end_ind = 0
    os.makedirs('../datasets/processed/MotionSense', exist_ok=True)
    for i in range(num_subjects): # for all subjects
        start_ind = end_ind
        end_ind = start_ind + sum([x.shape[0] for x in client_data[i]])
        a = combined_normalized_data[start_ind:end_ind]
        hkl.dump(a, '../datasets/processed/MotionSense/client_{}_data.hkl'.format(i))
        hkl.dump(all_user_labels[i], '../datasets/processed/MotionSense/client_{}_labels.hkl'.format(i))
        print('here')

if __name__ == '__main__':
    main()
