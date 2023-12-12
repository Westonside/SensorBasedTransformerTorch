# This is a sample Python script.
import gc

import hickle
import numpy as np
import torch.nn.functional


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

global_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(configuration_file: str):
    config = (configuration_utils.load_configuration(configuration_file))




    for configuration in config['configurations']:
        # for training_tasks in  training_tasks:
        print(configuration)
        dataset = load_datasets(match_config_key(configuration, "load_files"), path="./datasets/processed")
        task = match_configuration(configuration, 'type')(dataset, **configuration)
        print('starting training')
        task.train()

        with open("configurations_completed.txt", "a+") as f:
            f.write(str(configuration))

        gc.collect()
        torch.cuda.empty_cache()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    configuration_file = "./configurations/train_extractors.json"

    main(configuration_file)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
