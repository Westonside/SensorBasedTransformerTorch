import hickle as hkl
import numpy as np
from sklearn.model_selection import StratifiedKFold


def loadDataset(dataSetName, clientCount, dataConfig, randomSeed, mainDir, StratifiedSplit=True):
    # loading datasets
    clientDataTrain = []
    clientLabelTrain = []
    clientDataTest = []
    clientLabelTest = []
    centralTrainData = []
    centralTrainLabel = []
    centralTestData = []
    centralTestLabel = []
    clientOrientationTrain = []
    clientOrientationTest = []
    orientationsNames = None

    if (dataSetName == "UCI"):

        centralTrainData = hkl.load(mainDir + 'datasetStandardized/' + str(dataSetName) + '/trainX.hkl')
        centralTestData = hkl.load(mainDir + 'datasetStandardized/' + str(dataSetName) + '/testX.hkl')
        centralTrainLabel = hkl.load(mainDir + 'datasetStandardized/' + str(dataSetName) + '/trainY.hkl')
        centralTestLabel = hkl.load(mainDir + 'datasetStandardized/' + str(dataSetName) + '/testY.hkl')


    elif (dataSetName == "SHL"):
        clientData = hkl.load(mainDir + 'datasetStandardized/' + str(dataSetName) + '/clientsData.hkl')
        clientLabel = hkl.load(mainDir + 'datasetStandardized/' + str(dataSetName) + '/clientsLabel.hkl')
        clientCount = clientData.shape[0]

        for i in range(0, clientCount):
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            skf.get_n_splits(clientData[i], clientLabel[i])
            trainIndex = []
            testIndex = []
            for enu_index, (train_index, test_index) in enumerate(skf.split(clientData[i], clientLabel[i])):
                #             let indices at index 4 be used for test
                if (enu_index != 2):
                    trainIndex.append(test_index)
                else:
                    testIndex = test_index
            trainIndex = np.hstack((trainIndex))
            clientDataTrain.append(clientData[i][trainIndex])
            clientLabelTrain.append(clientLabel[i][trainIndex])
            clientDataTest.append(clientData[i][testIndex])
            clientLabelTest.append(clientLabel[i][testIndex])
        clientDataTrain = np.asarray(clientDataTrain)
        clientDataTest = np.asarray(clientDataTest)

        clientLabelTrain = np.asarray(clientLabelTrain)
        clientLabelTest = np.asarray(clientLabelTest)

        centralTrainData = np.vstack((clientDataTrain))
        centralTrainLabel = np.hstack((clientLabelTrain))

        centralTestData = np.vstack((clientDataTest))
        centralTestLabel = np.hstack((clientLabelTest))

    elif (dataSetName == "RealWorld"):
        orientationsNames = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
        clientDataTrain = {new_list: [] for new_list in range(clientCount)}
        clientLabelTrain = {new_list: [] for new_list in range(clientCount)}
        clientDataTest = {new_list: [] for new_list in range(clientCount)}
        clientLabelTest = {new_list: [] for new_list in range(clientCount)}

        clientOrientationData = hkl.load(mainDir + 'datasetStandardized/' + str(dataSetName) + '/clientsData.hkl')
        clientOrientationLabel = hkl.load(mainDir + 'datasetStandardized/' + str(dataSetName) + '/clientsLabel.hkl')

        clientOrientationTest = {new_list: [] for new_list in range(clientCount)}
        clientOrientationTrain = {new_list: [] for new_list in range(clientCount)}

        orientationIndex = 0
        for clientData, clientLabel in zip(clientOrientationData, clientOrientationLabel):
            for i in range(0, clientCount):
                skf = StratifiedKFold(n_splits=5, shuffle=False)
                skf.get_n_splits(clientData[i], clientLabel[i])
                trainIndex = []
                testIndex = []
                for enu_index, (train_index, test_index) in enumerate(skf.split(clientData[i], clientLabel[i])):
                    #             let indices at index 2 be used for test
                    if (enu_index != 2):
                        trainIndex.append(test_index)
                    else:
                        testIndex = test_index

                trainIndex = np.hstack((trainIndex))
                clientDataTrain[i].append(clientData[i][trainIndex])
                clientLabelTrain[i].append(clientLabel[i][trainIndex])
                clientDataTest[i].append(clientData[i][testIndex])
                clientLabelTest[i].append(clientLabel[i][testIndex])

                clientOrientationTest[i].append(np.full((len(testIndex)), orientationIndex))
                clientOrientationTrain[i].append(np.full((len(trainIndex)), orientationIndex))

            orientationIndex += 1

        for i in range(0, clientCount):
            clientDataTrain[i] = np.vstack((clientDataTrain[i]))
            clientDataTest[i] = np.vstack((clientDataTest[i]))
            clientLabelTrain[i] = np.hstack((clientLabelTrain[i]))
            clientLabelTest[i] = np.hstack((clientLabelTest[i]))
            clientOrientationTest[i] = np.hstack((clientOrientationTest[i]))
            clientOrientationTrain[i] = np.hstack((clientOrientationTrain[i]))

        clientOrientationTrain = np.asarray(list(clientOrientationTrain.values()))
        clientOrientationTest = np.asarray(list(clientOrientationTest.values()))

        clientDataTrain = np.asarray(list(clientDataTrain.values()))
        clientDataTest = np.asarray(list(clientDataTest.values()))

        clientLabelTrain = np.asarray(list(clientLabelTrain.values()))
        clientLabelTest = np.asarray(list(clientLabelTest.values()))

        centralTrainData = np.vstack((clientDataTrain))
        centralTrainLabel = np.hstack((clientLabelTrain))

        centralTestData = np.vstack((clientDataTest))
        centralTestLabel = np.hstack((clientLabelTest))

    else:
        clientData = []
        clientLabel = []

        for i in range(0, clientCount):
            clientData.append(hkl.load(mainDir + 'datasetStandardized/' + dataSetName + '/UserData' + str(i) + '.hkl'))
            clientLabel.append(
                hkl.load(mainDir + 'datasetStandardized/' + dataSetName + '/UserLabel' + str(i) + '.hkl'))

        if (dataSetName == "HHAR"):
            orientations = hkl.load(mainDir + 'datasetStandardized/HHAR/deviceIndex.hkl')
            orientationsNames = ['nexus4', 'lgwatch', 's3', 's3mini', 'gear', 'samsungold']

        for i in range(0, clientCount):
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            skf.get_n_splits(clientData[i], clientLabel[i])
            partitionedData = list()
            partitionedLabel = list()
            dataIndex = []
            trainIndex = []
            testIndex = []
            for enu_index, (train_index, test_index) in enumerate(skf.split(clientData[i], clientLabel[i])):
                if (enu_index != 2):
                    trainIndex.append(test_index)
                else:
                    testIndex = test_index
            trainIndex = np.hstack((trainIndex))
            clientDataTrain.append(clientData[i][trainIndex])
            clientLabelTrain.append(clientLabel[i][trainIndex])
            clientDataTest.append(clientData[i][testIndex])
            clientLabelTest.append(clientLabel[i][testIndex])
            clientOrientationTrain.append(trainIndex)
            clientOrientationTest.append(testIndex)

        if (dataSetName == "HHAR"):
            for i in range(0, clientCount):
                clientOrientationTest[i] = orientations[i][clientOrientationTest[i]]
                clientOrientationTrain[i] = orientations[i][clientOrientationTrain[i]]

        centralTrainData = (np.vstack((clientDataTrain)))
        centralTrainLabel = (np.hstack((clientLabelTrain)))

        centralTestData = (np.vstack((clientDataTest)))
        centralTestLabel = (np.hstack((clientLabelTest)))

    dataReturn = dataHolder
    dataReturn.clientDataTrain = clientDataTrain
    dataReturn.clientLabelTrain = clientLabelTrain
    dataReturn.clientDataTest = clientDataTest
    dataReturn.clientLabelTest = clientLabelTest
    dataReturn.centralTrainData = centralTrainData
    dataReturn.centralTrainLabel = centralTrainLabel
    dataReturn.centralTestData = centralTestData
    dataReturn.centralTestLabel = centralTestLabel
    dataReturn.clientOrientationTrain = clientOrientationTrain
    dataReturn.clientOrientationTest = clientOrientationTest
    dataReturn.orientationsNames = orientationsNames
    return dataReturn


class dataHolder:
    clientDataTrain = []
    clientLabelTrain = []
    clientDataTest = []
    clientLabelTest = []
    centralTrainData = []
    centralTrainLabel = []
    centralTestData = []
    centralTestLabel = []
    clientOrientationTrain = []
    clientOrientationTest = []
    orientationsNames = None
    activityLabels = []
    clientCount = None