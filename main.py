import gc

import hickle
import numpy as np
import torch.nn.functional


from training.pretext_tasks import match_configuration, Classification_Task
from utils import configuration_utils
from preprocess.dataset_loading import load_datasets
from utils.configuration_utils import match_config_key, modals

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
    configuration_file = "./configurations/trained_clustering_ft_ext.json"

    main(configuration_file)

