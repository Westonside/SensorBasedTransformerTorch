"""
    This function performs the first pretext task of training a transformer on each modality to perform transformation classification
    all datasets will be combined to perform the classification
"""
import inspect
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from UserDataLoader import UserDataLoader
from model import TransformerMultiTaskBinaryClassificationModel, TransformerClassificationModel
from utils.configuration_utils import modals
from utils.model_utils import train_epoch, MultiTaskLoss, SingleClassificationFormatter, BinaryClassificationFormatter, \
    MMS_loss
from utils.transformation_utils import transform_funcs_vectorized, transform_funcs_names
from fast_pytorch_kmeans import KMeans

def pretext_one():
    print('pretext one')


class Training_Task:
    # this will either load the model or create the model
    def __init__(self, dataset, modalities, save_path="./", previous_task_path=None, epochs=80, early_stop=False):
        self.model = None
        self.dataset = dataset
        self.save_path = save_path
        self.epochs = epochs
        self.model = None
        self.num_modal = len(modalities) if isinstance(modalities, list) else 1
        modals_names = " ".join(modalities) if isinstance(modalities, list) else modalities
        self.modal_range = modals[modals_names]
        self.dataset = dataset
        self.sequence_length = dataset.train.shape[1]
        self.models_path = "./models"
        if previous_task_path is None:
            self.create_model()
        else:
            self.load_model(previous_task_path)
        pass

    def create_model(self):
        pass

    def get_model(self):
        return self.model

    def load_model(self, configuration_path):
        self.model = torch.load(configuration_path)

    def get_save_file_name(self) -> str:
        pass

    def save_model(self):
        os.makedirs('./models', exist_ok=True) # make a new directory if it does not exist
        torch.save(self.model, os.path.join(self.models_path, os.path.join(self.save_path, self.get_save_file_name()))) # save the model

    def train_task_setup(self):
        self.dataset.keep_modalities(self.modal_range)



    def get_output_formatter(self):
        pass


    def get_training_data(self):
        pass

    def train(self):
        # move the device to gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.train_task_setup()  # will set up the training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        training, training_label = self.get_training_data()
        for epoch in range(1, self.epochs + 1):
            train_epoch(self.model, epoch, training, training_label, output_formatter=self.get_output_formatter(),
                        optimizer=optimizer, loss_fn=self.get_loss(), device=device)
            # self.save_model()
        self.save_model()


    def get_loss(self):
        pass


class Classification_Task(Training_Task):
    TASK_NAME = "classification_task"
    def __init__(self, dataset: UserDataLoader, modalities=["accelerometer"], save_path="./", previous_task_path=None, epochs=80, early_stop=False, **kwargs):
        super().__init__(dataset, save_path=save_path, modalities=modalities, previous_task_path=previous_task_path, epochs=epochs)
        self.dataset = dataset
        self.model = None
        self.create_model()

    def create_model(self):
        self.model = TransformerClassificationModel((self.sequence_length,len(self.modal_range)), 13, self.num_modal)

    def train_task_setup(self):
        super().train_task_setup()

    def get_training_data(self):
        return self.dataset.train, self.dataset.train_label

    def get_output_formatter(self):
        return SingleClassificationFormatter()


    def get_loss(self):
        return nn.CrossEntropyLoss()


class Transformation_Classification_Task(Training_Task):
    TASK_NAME = "transformation_classification_task"
    def __init__(self, dataset: UserDataLoader, epochs=80, modalities=["accelerometer"],  **kwargs):
        super().__init__(dataset, save_path=kwargs.get("save_path"), modalities=modalities, previous_task_path=kwargs.get("previous_model"), epochs=epochs)
        self.model = None
        self.transformations = transform_funcs_vectorized
        self.create_model()


    def train_task_setup(self):
        # this will set up the trainin
        super().train_task_setup()
        self.dataset.transform_sets(self.transformations)  # transform the data

    def create_model(self):
        #TODO: ALLLOW PASSING IN OTHER  transformations
        #TODO: remove the magic number
        self.model = TransformerMultiTaskBinaryClassificationModel((self.sequence_length,3), len(transform_funcs_vectorized))


    def get_training_data(self):
        return self.dataset.transform_train, self.dataset.transform_label


    def get_loss(self):
        return MultiTaskLoss(len(self.transformations))


    def get_output_formatter(self):
        return BinaryClassificationFormatter(len(self.transformations), transform_funcs_names)

def match_configuration(config, key):
    if config.get(key) is None:
        print(f"configuration does not have a {key}")
        return None
    config = config[key]
    config = config.lower()
    if PRETEXT_TASKS.get(config) is not None:
        return PRETEXT_TASKS.get(config)
    else:
        print("Not found!", key)
        return None



class Multi_Modal_Clustering_Task(Training_Task):
    TASK_NAME = "multi_modal_clustering_task"
    def __init__(self, dataset: UserDataLoader,  epochs=80, **kwargs):
        # use the silhouette score in kmeans
        super().__init__(dataset, save_path=kwargs.get("save_path"), previous_task_path=kwargs.get("previous_model"), epochs=epochs)
        self.dataset = dataset
    def create_model(self):
        self.model = None
        pass

    def train_task_setup(self):
        super().train_task_setup()

    def train(self):
        # you will first attempt to reconstruct the input
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        training, training_label = self.get_training_data()
        loss_op = MMS_loss()
        loss_op.to(device)
        for epoch in range(1, self.epochs + 1):
            # now you will train each batch
            queue_v = None
            use_the_queue = False
            centroid = None
            batch_size = 64


            permutation = torch.randperm(training.shape[0])

            for i in range(0, training.shape[0], batch_size):
                optimizer.zero_grad()
                # data = data.to(device)
                data = training[permutation[i:i + batch_size]]

                acc = data[:, :, 0:3].to(device)
                gyro = data[:, :, 3:6].to(device)
                with torch.set_grad_enabled(True):
                    acc_ft, gyro_ft, classified_acc, classified_gyro, recon_loss = self.model(acc,gyro)
                    recon_weight = 50
                    recon_loss = torch.mean(recon_loss) * recon_weight

                    acc_out = classified_acc
                    gyro_out = classified_gyro

                    fused_data = (acc_out + gyro_out) / 2 # joining the extracted features so that they can be clustered


                    sim_audio_acc = torch.matmul(acc_ft, gyro_ft.t()) #calculates the ismilarity between the gyro and the acc
                    sim_audio_gyro = torch.matmul(gyro_ft, acc_ft.t()) # calculates the similarity between the acc and the gyro

                    # calculate the loss
                    loss = loss_op(sim_audio_gyro) + loss_op(sim_audio_acc)
                    # kmeans time
                    queue_v,  out, use_the_queue = update_queue(queue_v, use_the_queue, fused_data)
                    kmeans = KMeans(n_clusters=256, mode='cosine')
                    labels = kmeans.fit_predict(out)
                    centroid = kmeans.centroids
                    # get the labels for the items in the batch
                    loss_val = cluster_contrastive(fused_data, centroid, labels[-batch_size:], batch_size)
                    loss += loss_val * 1 # clustering lambda
                    loss += recon_loss
                    loss.backward()
                    optimizer.step()

                    return loss.item(), queue_v, use_the_queue, centroid


    def train_one_epoch(self, model, opt, data, loss_fn, scheduler):
        pass



def cluster_contrastive(fushed,centroid,labels,bs):
    S = torch.matmul(fushed, centroid.t()) # get the similarity between the fused data and the centroids

    target = torch.zeros(bs, centroid.shape[0]).to(S.device) # create a target tensor

    target[range(target.shape[0]), labels] = 1 # set the target tensor to be 1 where the label is

    S = S - target * (0.001) # subtract the target from the similarity matrix

    I2C_loss = nn.functional.nll_loss(nn.functional.log_softmax(S, dim=1), labels) # calculate the loss

    # else:
    #     S = S.view(S.shape[0], S.shape[1], -1)
    #     nominator = S * target[:, :, None]
    #     nominator = nominator.sum(dim=1)
    #     nominator = th.logsumexp(nominator, dim=1)
    #     denominator = S.view(S.shape[0], -1)
    #     denominator = th.logsumexp(denominator, dim=1)
    #     I2C_loss = th.mean(denominator - nominator)

    return I2C_loss


def update_queue(queue,use_the_queue,fuse):
    bs = int(4096/2)
    fuse2 = fuse.detach()
    fuse2 = fuse2.view(-1, 32, fuse2.shape[-1])
    fuse2 = fuse2[:,:16,:]
    fuse2 = fuse2.reshape(-1, fuse2.shape[-1])
    out = fuse.detach()
    if queue is not None:  # no queue in first round
        if use_the_queue or not torch.all(queue[ -1, :] == 0):  # queue[2,3840,128] if never use the queue or the queue is not full
            use_the_queue = True
            # print('use queue')
            out = torch.cat((queue,fuse.detach()))  # queue [1920*128] w_t [128*3000] = 1920*3000 out [32*3000] 1952*3000

            #print('out size',out.shape)
        # fill the queue
        queue[ bs:] = queue[ :-bs].clone()  # move 0-6 to 1-7 place
        queue[:bs] = fuse2
    return queue,out,use_the_queue

PRETEXT_TASKS = {
    Transformation_Classification_Task.TASK_NAME: Transformation_Classification_Task,
    Multi_Modal_Clustering_Task.TASK_NAME: Multi_Modal_Clustering_Task,
    Classification_Task.TASK_NAME: Classification_Task
}
