#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import os
import requests
import urllib.request
import zipfile
from scipy import stats
import scipy.signal
import hickle as hkl

# In[ ]:


np.random.seed(0)


# In[ ]:


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def create_segments_and_labels_PAMAP(df, time_steps, step, label_name="LabelsEncoded", n_features=6):
    # df = df.values
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        acc_x = df['acc_x'].values[i: i + time_steps]
        acc_y = df['acc_y'].values[i: i + time_steps]
        acc_z = df['acc_z'].values[i: i + time_steps]

        gyro_x = df['gyro_x'].values[i: i + time_steps]
        gyro_y = df['gyro_y'].values[i: i + time_steps]
        gyro_z = df['gyro_z'].values[i: i + time_steps]

        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0]
        reshaped = np.dstack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]).reshape(time_steps, n_features)
        segments.append(reshaped)
        labels.append(label)

    return np.asarray(segments), np.asarray(labels)


def standardize_data(deviceData):
    deviceDataAcc = deviceData[:, :, :3].astype(np.float32)
    deviceDataGyro = deviceData[:, :, 3:].astype(np.float32)
    accMean = np.mean(deviceDataAcc)
    accStd = np.std(deviceDataAcc)
    gyroMean = np.mean(deviceDataGyro)
    gyroStd = np.std(deviceDataGyro)
    deviceDataAcc = (deviceDataAcc - accMean) / accStd
    deviceDataGyro = (deviceDataGyro - gyroMean) / gyroStd
    deviceData = np.dstack((deviceDataAcc, deviceDataGyro))
    return deviceData


# In[ ]:


# Load data

list_of_files = ['PAMAP2_Dataset/Protocol/subject101.dat',
                 'PAMAP2_Dataset/Protocol/subject102.dat',
                 'PAMAP2_Dataset/Protocol/subject103.dat',
                 'PAMAP2_Dataset/Protocol/subject104.dat',
                 'PAMAP2_Dataset/Protocol/subject105.dat',
                 'PAMAP2_Dataset/Protocol/subject106.dat',
                 'PAMAP2_Dataset/Protocol/subject107.dat',
                 'PAMAP2_Dataset/Protocol/subject108.dat',
                 'PAMAP2_Dataset/Protocol/subject109.dat']

subjectID = [1, 2, 3, 4, 5, 6, 7, 8, 9]
keep_labels = {6: 'lay',
               3: 'sit',
               4: 'stand',
               5: 'walk',
               2: 'run',
               7: 'bike',
               8: 'nordic-walk',
               9:'jump-rope'}
activityIDdict = {
                    1: 'lying', #KEEP goes
                    2: 'sitting', #keep
                    3: 'standing', # keep
                    4: 'walking', # keep
                    5: 'running', # keep
                    6: 'cycling', # keep
                    7: 'Nordic_walking',  # keep
                    12: 'Up-Stairs'
                  }

# activityIDdict = {0: 'transient',
#                   1: 'lying', # goes to 6
#                   2: 'sitting', # goes to 3
#                   3: 'standing', # goes to 4
#                   4: 'walking', # goes to 5
#                   5: 'running', # goes 2
#                   6: 'cycling', 7
#                   7: 'Nordic_walking', 8
#                   9: 'watching_TV',
#                   10: 'computer_work',
#                   11: 'car driving',
#                   12: 'ascending_stairs', 1
#                   13: 'descending_stairs', 0
#                   16: 'vacuum_cleaning', -1
#                   17: 'ironing', -1
#                   18: 'folding_laundry',
#                   19: 'house_cleaning',
#                   20: 'playing_soccer',
#                   24: 'rope_jumping' 9
#                   }
colNames = ["timestamp", "activityID", "heartrate"]
colNames_reduced = ["timestamp", "activityID"]

IMUhand = ['handTemperature',
           'handAcc16_1', 'handAcc16_2', 'handAcc16_3',
           'handAcc6_1', 'handAcc6_2', 'handAcc6_3',
           'handGyro1', 'handGyro2', 'handGyro3',
           'handMagne1', 'handMagne2', 'handMagne3',
           'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

IMUchest = ['chestTemperature',
            'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3',
            'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3',
            'chestGyro1', 'chestGyro2', 'chestGyro3',
            'chestMagne1', 'chestMagne2', 'chestMagne3',
            'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

IMUankle = ['ankleTemperature',
            'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3',
            'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3',
            'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
            'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
            'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

only_pocket_setup = ['ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3',
                     'ankleGyro1', 'ankleGyro2', 'ankleGyro3']

columns = colNames + IMUhand + IMUchest + IMUankle  # all columns in one list

columns_reduced = colNames_reduced + only_pocket_setup

len(columns)
# len(columns_reduced)


# In[ ]:


fileName = ["pamap2+physical+activity+monitoring"]
links = ["https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"]

# In[ ]:


os.makedirs('dataset/download', exist_ok=True)
os.makedirs('dataset/extracted', exist_ok=True)

# In[ ]:


for i in range(len(fileName)):
    data_directory = os.path.abspath("dataset/download/" + str(fileName[i]) + ".zip")
    if not os.path.exists(data_directory):
        print("downloading " + str(fileName[i]))
        download_url(links[i], data_directory)
        print("download done")
        data_directory2 = os.path.abspath("dataset/extracted/" + str(fileName[i]) + ".zip")
        print("extracting data...")
        with zipfile.ZipFile(data_directory, 'r') as zip_ref:
            zip_ref.extractall(os.path.abspath("dataset/extracted/"))
        print("data extracted")
    else:
        print(str(fileName[i]) + " already downloaded")

# In[ ]:


if not os.path.exists("dataset/extracted/PAMAP2_Dataset"):
    print("extracting sub-zip...")
    with zipfile.ZipFile("dataset/extracted/PAMAP2_Dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(os.path.abspath("dataset/extracted/"))
    print("data extracted")
else:
    print("sub-zip already extracted.")

# In[ ]:


dataCollection = pd.DataFrame()
main_dir = "dataset/extracted/"
for file in list_of_files:
    procData = pd.read_table(main_dir + file, header=None, sep='\s+')
    procData.columns = columns
    # procData.columns = columns_reduced
    procData['subject_id'] = int(file[-5])
    # dataCollection = dataCollection.append(procData, ignore_index=True)
    dataCollection = pd.concat([dataCollection,procData], ignore_index=True)
dataCollection.reset_index(drop=True, inplace=True)
dataCollection.head()


# In[ ]:


def dataCleaning(dataCollection):
    # removal of orientation columns as they are not needed
    dataCollection = dataCollection.drop(dataCollection[
                                             dataCollection.activityID == 0].index)  # removal of any row of activity 0 as it is transient activity which it is not used
    # dataCollection = dataCollection.apply(pd.to_numeric, errors = 'coerse') #removal of non numeric data in cells
    dataCollection = dataCollection.interpolate()  # removal of any remaining NaN value cells by constructing new data points in known set of data points

    return dataCollection


# In[ ]:


dataCol = dataCleaning(dataCollection)

# In[ ]:


dataCol.reset_index(drop=True, inplace=True)
dataCol.head(10)

# In[ ]:


dataCol.isnull().sum()

# In[ ]:


for i in range(0, 4):
    dataCol["heartrate"].iloc[i] = 100
dataCol.isnull().sum()
# remove all but 8 activities

# In[ ]:


# dataCol['activityID'].value_counts().plot(kind="bar", figsize=(12, 6))

# In[ ]:

dataCol.rename(columns={
    'ankleAcc16_1': 'acc_x',
    'ankleAcc16_2': 'acc_y',
    'ankleAcc16_3': 'acc_z',

    'ankleGyro1': 'gyro_x',
    'ankleGyro2': 'gyro_y',
    'ankleGyro3': 'gyro_z',

    'activityID': 'LabelsEncoded'
}, inplace=True)
# here is where activities will be dropped


dataCol['activityString'] = dataCol['LabelsEncoded'].map(activityIDdict)
# dataCol.loc[dataCol['LabelsEncoded']==12, :] = 8
unique_user_ids = dataCol['subject_id'].unique()

# In[ ]:


all_data = []
all_labels = []
for user_id in unique_user_ids[:8]:
    # print(user_id)
    selected_data = dataCol.loc[dataCol['subject_id'] == user_id]
    x, y = create_segments_and_labels_PAMAP(selected_data, 256, 64)

    x = scipy.signal.decimate(x, q=2, n=None, ftype='iir', axis=1, zero_phase=True)
    # x_aligned = standardize_data(x)
    print(x.shape)
    mapping = [-1, 6, 3, 4, 5, 2, 7, 8, -1, -1, -1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9] # TODO WHAT IS THIS MAPPING DOING
    y_aligned = np.hstack(mapping[labelIndex] for labelIndex in y)
    all_data.append(x)
    all_labels.append(y_aligned)

print(len(all_data))

# In[ ]:

all_labels = np.asarray(all_labels)
all_data = np.asarray(all_data)
keep_indicies = [np.where(np.isin(user ,list(keep_labels.keys())))[0] for user in all_labels]
all_labels = [all_labels[x][keep_indicies[x]] for x in range(len(keep_indicies))]
all_data = [all_data[x][keep_indicies[x]] for x in range(len(keep_indicies))]

# In[ ]:

subjectIndex = []
for data in all_data:
    subjectIndex.append(data.shape[0])

# In[ ]:


allData = np.vstack((all_data))
standardizedData = standardize_data(allData)

# In[ ]:


subjectData = []
startIndex = 0
endIndex = 0
for index in subjectIndex:
    endIndex += index
    subjectData.append(standardizedData[startIndex:endIndex])
    startIndex = endIndex
subjectData = np.asarray(subjectData)

# In[ ]:


dataName = 'PAMAP'
os.makedirs('datasetClientsUnion/' + dataName, exist_ok=True)
hkl.dump(subjectData, 'datasetClientsUnion/' + dataName + '/clients_data.hkl')
hkl.dump(all_labels, 'datasetClientsUnion/' + dataName + '/clients_label.hkl')
# /static/public/364/smartphone+dataset+for+human+activity+recognition+har+in+ambient+assisted+living+aal.zip