# This is a sample Python script.
import numpy as np
import torch.nn.functional

from datasets import loadDataset
from model import TransformerModel, TransformerClassificationModel
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from training.pretext_tasks import match_configuration, Classification_Task
from utils import configuration_utils
from preprocess.dataset_loading import load_datasets
from utils.configuration_utils import match_config_modal, modals


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


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

    #TODO: allow for actually getting the number of clients I am just hard coding for now
    client_count = 24
    # TODO: make a dataloader for the data set
    main_dir = "../HART/Lightweight-Transformer-Models-For-HAR-on-Mobile-Devices/"
    dataset_loader = loadDataset(data_set_name, client_count, None, None, main_dir)
    central_train_data = dataset_loader.centralTrainData
    central_train_label = dataset_loader.centralTrainLabel

    central_test_data = dataset_loader.centralTestData
    central_test_label = dataset_loader.centralTestLabel

    client_orientation_train_data = dataset_loader.clientDataTrain
    client_orientation_train_label = dataset_loader.clientLabelTrain
    orientation_names = dataset_loader.orientationsNames

    central_train_data, central_dev_data, central_train_label, central_dev_label = train_test_split(central_train_data, central_train_label, test_size=0.125, random_state=42)

    temp_weights = class_weight.compute_class_weight('balanced', classes=np.unique(central_train_label), y=central_train_label.ravel())
    class_weights = {j: temp_weights[j] for j in range(len(temp_weights))}


    # get the one hot of the labels
    # central_train_label = torch.nn.functional.one_hot(torch.from_numpy(central_train_label), num_classes=activity_count).numpy()
    # central_test_label = torch.nn.functional.one_hot(torch.from_numpy(central_test_label), num_classes=activity_count).numpy()
    # central_dev_label = torch.nn.functional.one_hot(torch.from_numpy(central_dev_label), num_classes=activity_count).numpy()




    # model = TransformerClassificationModel(input_shape,activity_count, modal_count=2)

    # define the optimizer here
    learningRate = 3e-4
    # optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    # print(model)
    training_tasks = []

    dataset = load_datasets(["MotionSense"], path="./datasets/processed")


    # task = Classification_Task(dataset, save_path="./", previous_task_path=None, epochs=80, early_stop=False)
    # task.train()

    for configuration in config['configurations']:
        # for training_tasks in  training_tasks:
        print(configuration)
        dataset = load_datasets(["MotionSense"], path="./datasets/processed")

        task = match_configuration(configuration, 'type')
        modal = match_config_modal(configuration, 'modalities')
        if modal is not None:
            modal_range = modals[modal]
            task = task(dataset, modal_range)
            task.train()

        # for epoch in range(1, local_epoch + 1): #TODO: implement early stopping
        #     # zero the gradients
        #     optimizer.zero_grad()
        #     train_epoch(model, epoch, central_train_data, central_train_label, optimizer, torch.nn.functional.cross_entropy)












# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    configuration_file = "./configurations/basic_configuration.json"
    main(configuration_file)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
