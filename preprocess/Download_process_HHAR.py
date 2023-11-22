import os
import threading
import hickle as hkl
import torch
from scipy import signal
import numpy as np
import pandas as pd

import preprocess.preprocess_utils

fileName = ["Activity recognition exp"]
links = ["http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip"]
loadList = ['Phones_accelerometer.csv','Phones_gyroscope.csv','Watch_accelerometer.csv','Watch_gyroscope.csv']
classCounts = ['sit', 'stand', 'walk', 'stairsup', 'stairsdown', 'bike']
deviceCounts = ['nexus4', 'lgwatch','s3', 's3mini','gear','samsungold']
deviceSamplingRate = [200,200,150,100,100,50]
deviceWindowFrame = [512,512,384,256,256,128]
downSamplingRate = [4,4,3,2,2,1]
subDeviceCounts = ['nexus4_1', 'nexus4_2', 'lgwatch_1', 'lgwatch_2', 's3_1', 's3_2', 's3mini_1', 's3mini_2','gear_1', 'gear_2','samsungold_1', 'samsungold_2']
userCounts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']


def prepare_data(data_file, modal_dict):
    data = pd.read_csv(data_file)
    # data = data.drop(columns=data.columns[0]) #drop their index
    data = data.to_numpy()
    modal_dict[os.path.basename(data_file)] = data

def main():
    preprocess.preprocess_utils.download_and_extract(fileName, links, "../datasets")
    dataDir = "../datasets/extracted/Activity recognition exp"

    processed_modal = {}
    threads = []
    for i, modality in enumerate(loadList):
       t_modal = threading.Thread(target=prepare_data, args=(dataDir + '/' + modality, processed_modal))
       threads.append(t_modal)
       t_modal.start()

    for t in threads:
        t.join()

    unprocessedAccData = np.vstack([processed_modal['Phones_accelerometer.csv'], processed_modal['Watch_accelerometer.csv']])
    unprocessedGyroData = np.vstack([processed_modal['Phones_gyroscope.csv'], processed_modal['Watch_gyroscope.csv']])
    allProcessedData = {}
    allProcessedLabel = {}
    deviceIndex = {}
    clientCount = len(deviceCounts) * len(userCounts)
    deviceIndexes = {new_list: [] for new_list in range(len(deviceCounts))}
    indexOffset = 0
    # for every device get the clients that used it and
    for clientDeviceIndex, deviceName in enumerate(deviceCounts):
        print("Processsing device " + str(deviceName))
        for clientIDIndex, clientIDName in enumerate(userCounts):
            print("Processsing device:" + str(clientDeviceIndex) + " client " + str(clientIDIndex))

            processedClassData = []
            processedClassLabel = []
            dataIndex = (unprocessedAccData[:, 6] == clientIDName) & (unprocessedAccData[:, 7] == deviceName) # get the data it is the client and the device
            userDeviceDataAcc = unprocessedAccData[dataIndex]
            if (len(userDeviceDataAcc) == 0): # if they don't use that device
                print("No acc data found")
                print("Skipping device :" + str(deviceName) + " Client: " + str(clientIDName))
                indexOffset += 1
                continue
            userDeviceDataGyro = unprocessedGyroData[
                (unprocessedGyroData[:, 6] == clientIDName) & (unprocessedGyroData[:, 8] == deviceName)]
            if (len(userDeviceDataGyro) == 0):
                userDeviceDataGyro = unprocessedGyroData[np.where(dataIndex == True)[0]]

            for classIndex, className in enumerate(classCounts):
                if (len(userDeviceDataAcc) <= len(userDeviceDataGyro)):
                    classData = np.where(userDeviceDataAcc[:, 9] == className)[0]
                else:
                    classData = np.where(userDeviceDataGyro[:, 9] == className)[0]
                segmentedClass = consecutive(classData, deviceWindowFrame[int(clientDeviceIndex / 2)])
                for segmentedClassRange in (segmentedClass):
                    combinedData = np.dstack((segmentData(userDeviceDataAcc[segmentedClassRange][:, 3:6],
                                                          deviceWindowFrame[clientDeviceIndex],
                                                          deviceWindowFrame[clientDeviceIndex] / 2),
                                              segmentData(userDeviceDataGyro[segmentedClassRange][:, 3:6],
                                                          deviceWindowFrame[clientDeviceIndex],
                                                          deviceWindowFrame[clientDeviceIndex] / 2)))
                    processedClassData.append(combinedData)
                    processedClassLabel.append(np.full(combinedData.shape[0], classIndex, dtype=int))
            deviceCheckIndex = clientDeviceIndex % 2
            tempProcessedData = np.vstack((processedClassData))
            if (clientDeviceIndex < 5):
                tempProcessedData = downSampleLowPass(np.float32(tempProcessedData),
                                                      downSamplingRate[clientDeviceIndex])
            dataIndex = (len(userCounts) * clientDeviceIndex) + clientIDIndex - indexOffset
            print("Index is at " + str(dataIndex))
            allProcessedData[dataIndex] = tempProcessedData
            allProcessedLabel[dataIndex] = np.hstack((processedClassLabel))
            deviceIndex[dataIndex] = np.full(allProcessedLabel[dataIndex].shape[0], clientDeviceIndex)
            deviceIndexes[clientDeviceIndex].append(dataIndex)

    allProcessedData = list(allProcessedData.values()) #TODO FIX ME
    # allProcessedData = np.asarray(list(allProcessedData.items()))[:, 1] #TODO FIX ME
    allProcessedLabel = list(allProcessedLabel.values())
    # allProcessedLabel = np.asarray(list(allProcessedLabel.items()))[:, 1]
    # deviceIndex = np.asarray(list(deviceIndex.items()))[:, 1]
    deviceIndex = list(deviceIndex.values())
    deleteIndex = []
    for index, i in enumerate(allProcessedLabel):
        if (len(np.unique(i)) < len(classCounts)):
            print("Removing client " + str(index))
            print(np.unique(i))
            deleteIndex.append(index)
            for key, value in dict(deviceIndexes).items():
                if (value.count(index)):
                    value.remove(index)
    # allProcessedLabel = np.delete(allProcessedLabel, deleteIndex)
    # allProcessedData = np.delete(allProcessedData, deleteIndex)
    allProcessedLabel = [value for index, value in enumerate(allProcessedLabel) if index not in deleteIndex]
    allProcessedData = [value for index, value in enumerate(allProcessedData) if index not in deleteIndex]
    deviceIndex = [value for index, value in enumerate(deviceIndex) if index not in deleteIndex]
    # deviceIndex = np.delete(deviceIndex, deleteIndex)



    clientRange = [len(arrayLength) for arrayLength in allProcessedLabel]
    deviceSize = []
    for key, value in dict(deviceIndexes).items():
        deviceSize.append(len(value))

    # In[ ]:

    normalizedData = []

    # In[ ]:

    endIndex = 0
    for i in deviceSize:
        startIndex = endIndex
        endIndex += i
        #     print(startIndex)
        #     print(endIndex)
        if len(allProcessedData[startIndex:endIndex]) == 0:
            continue

        max_rows = max(arr.shape[0] for arr in allProcessedData[startIndex:endIndex])
        max_columns = max(arr.shape[1] for arr in allProcessedData[startIndex:endIndex])

        # Pad only the first dimension (rows) of the selected rows in allProcessedData
        padded_arrays =[np.pad(arr, ((0,max_rows-arr.shape[0]), (0,0), (0,0)), mode='constant', constant_values=np.nan) for arr in allProcessedData[startIndex:endIndex]]

        # Vertically stack the padded arrays
        deviceData = np.vstack(padded_arrays)

        deviceDataAcc = deviceData[:, :, :3].astype(np.float32)
        deviceDataGyro = deviceData[:, :, 3:].astype(np.float32)
        accMean = np.nanmean(deviceDataAcc)
        accStd = np.nanstd(deviceDataAcc)
        gyroMean = np.nanmean(deviceDataGyro)
        gyroStd = np.nanstd(deviceDataGyro)
        deviceDataAcc = (deviceDataAcc - accMean) / accStd
        deviceDataGyro = (deviceDataGyro - gyroMean) / gyroStd
        deviceData = np.dstack((deviceDataAcc, deviceDataGyro))


        normalizedData.append(deviceData)



        target_nan = np.isnan(deviceData)
        # now that we found the nan values
        # we replace them with -inf
        deviceData[target_nan] = -torch.inf
        # these values will be masked in the attention layer
        normalizedData = np.vstack(normalizedData)
        # In[ ]:

        startIndex = 0
        endIndex = 0
        dataName = 'HHAR'
        os.makedirs('../datasets/processed/' + dataName, exist_ok=True)
        # clientRange
        for i, dataRange in enumerate(clientRange):
            print(normalizedData[startIndex:endIndex].shape, allProcessedLabel[i].shape)
            startIndex = endIndex
            endIndex = startIndex + dataRange
            hkl.dump(normalizedData[startIndex:endIndex],
                     '../datasets/processed/' + dataName + '/UserData' + str(i) + '.hkl')
            hkl.dump(allProcessedLabel[i], '../datasets/processed/' + dataName + '/UserLabel' + str(i) + '.hkl')
        hkl.dump(deviceIndex, '../datasets/processed/' + dataName + '/deviceIndex.hkl')


    print('done processing')






def downSampleLowPass(motionData,factor):
    accX = signal.decimate(motionData[:,:,0],factor)
    accY = signal.decimate(motionData[:,:,1],factor)
    accZ = signal.decimate(motionData[:,:,2],factor)
    gyroX = signal.decimate(motionData[:,:,3],factor)
    gyroY = signal.decimate(motionData[:,:,4],factor)
    gyroZ = signal.decimate(motionData[:,:,5],factor)
    return np.dstack((accX,accY,accZ,gyroX,gyroY,gyroZ))
def segmentData(accData,time_step,step):
    step = int(step)
    segmentAccData = []
    for i in range(0, accData.shape[0] - time_step,step):

        segmentAccData.append(accData[i:i+time_step,:])


    return np.asarray(segmentAccData)
def segmentLabel(accData,time_step,step):
    segmentAccData = list()
    for i in range(0, accData.shape[0] - time_step,step):
        segmentAccData.append(processLabel(accData[i:i+time_step]))
    return np.asarray(segmentAccData)

def processLabel(labels):
    uniqueCount = np.unique(labels,return_counts=True)
    if(len(uniqueCount[0]) > 1):
        return uniqueCount[0][np.argmax(uniqueCount[1])]
    else:
        return uniqueCount[0][0]


def consecutive(data, treshHoldSplit,stepsize=1):
    splittedData = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    returnResults= [newArray for newArray in splittedData if len(newArray)>=treshHoldSplit]
    return returnResults




if __name__ == '__main__':
    main()