# This is a sample Python script.
import gc

import hickle
import numpy as np
import torch.nn.functional

from datasets import loadDataset
from model import TransformerModel, TransformerClassificationModel
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from training.pretext_tasks import match_configuration, Classification_Task
from utils import configuration_utils
from preprocess.dataset_loading import load_datasets
from utils.configuration_utils import match_config_key, modals


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

#TODO: REMOVE REALWORLD ADD IN WISDM AND HHAR EARLY STOPPING
#STORE ACCURACIES FOR EACH TASK IN SPARCL
# RANDOM SPLITTING ON USERS INSTEAD OF MEAN SPLITTING
# CNN AND TRANSFORMER
# FIRST BASELINE IS JUST CONTINUAL LEARNING
# ASK QUESTIONS ABOUT WHAT I AM DOING FOR FOR REPLAY
# 2 CLASSES PER TASK
# USE SOME OF THE AVALANCHE BASELINES
# CHOOSE TWO REGULARIZATION TWO REPLAY METHODS EWC AND ICARL 2 regularization and two replay
# I am taking one dataset for the self supervised training and another for the testing(downstream task)
# compare my transformer and hart transformer to see which is faster and has similar accuracy () do last

def main(configuration_file: str):
    # Use a breakpoint in the (code line below to debug your script.
    config = (configuration_utils.load_configuration(configuration_file))
    Motion_Sense = [2, 1, 3, 4, 0, 7]
    ACTIVITY_LABEL = ['Standing', 'Walking', 'Runing', 'Biking', 'Car', 'Bus', 'Train', 'Subway']
    activity_count= len(ACTIVITY_LABEL)


    local_epoch = 200
    batch_size = 64
    projection_dim = 192
    frame_length = 16
    time_step = 16
    data_set_name = "MotionSense"

    segment_size = 128
    num_input_channels = 6

    input_shape = (segment_size, num_input_channels)
    projection_half = projection_dim // 2
    projection_quarter = projection_dim // 4
    filter_attention = 4

    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    R = projection_half // filter_attention

    segmentTime = [x for x in range(0, segment_size - frame_length+ time_step, time_step)]
    # model = TransformerClassificationModel(input_shape,activity_count, modal_count=2)

    # define the optimizer here
    learningRate = 3e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    # print(model)
    training_tasks = []

    # dataset = load_datasets(["MotionSense"], path="./datasets/processed")
    #
    #
    # task = Classification_Task(dataset, save_path="./", previous_task_path=None, epochs=80, early_stop=False,modalities=['accelerometer'])
    # task.train()
    for configuration in config['configurations']:
        # for training_tasks in  training_tasks:
        print(configuration)
        dataset = load_datasets(match_config_key(configuration, "load_files"), path="./datasets/processed")
        task = match_configuration(configuration, 'type')(dataset, **configuration)
        task.train()


        gc.collect()
        torch.cuda.empty_cache()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    configuration_file = "./configurations/trained_clustering_ft_ext.json"

    main(configuration_file)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
