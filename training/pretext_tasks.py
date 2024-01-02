"""
    This function performs the first pretext task of training a transformer on each modality to perform transformation classification
    all datasets will be combined to perform the classification
"""
import os

import torch
from torch import nn
import hickle as hkl

from model_impl.HART import HartClassificationModel
from preprocess.UserDataLoader import UserDataLoader
from model_impl.model import TransformerMultiTaskBinaryClassificationModel, TransformerClassificationModel, \
    MultiModalTransformer
from transfer_model import TransferModel, TransferModelClassification
from utils.configuration_utils import modals
from utils.model_utils import train_epoch, MultiTaskLoss, SingleClassificationFormatter, BinaryClassificationFormatter, \
    MMS_loss, EarlyStop, validation, extract_features
from preprocess.transformation_utils import transform_funcs_vectorized, transform_funcs_names
from fast_pytorch_kmeans import KMeans


def pretext_one():
    print('pretext one')


class Training_Task:
    # this will either load the model or create the model
    def __init__(self, dataset, modalities, save_dir: str, save_file: str, epochs=80, lr=0.03, batch_size=64,
                 early_stop_patience=None, verbose=False):
        self.verbose = verbose
        self.lr = lr
        self.model = None
        self.dataset = dataset
        self.dir = save_dir
        self.save_file = save_file
        self.epochs = epochs
        self.model = None
        self.batch_size = batch_size
        self.num_modal = len(modalities) if isinstance(modalities, list) else 1
        modals_names = " ".join(modalities) if isinstance(modalities, list) else modalities
        self.modal_range = modals[modals_names]
        self.dataset = dataset
        self.sequence_length = dataset.train.shape[1]
        self.models_path = "./models"
        self.create_model()
        if early_stop_patience is not None:
            self.get_early_stop(early_stop_patience)
        else:
            self.early_stopping = None
        # if previous_task_path is None:
        #     self.create_model()
        # else:
        #     self.load_model(previous_task_path)
        pass

    def create_model(self):
        pass

    def get_model(self):
        return self.model

    def load_model(self, configuration_path):
        pass

    def get_save_file_name(self) -> str:
        pass

    def save_model(self):
        os.makedirs(self.dir, exist_ok=True)  # make a new directory if it does not exist
        torch.save(self.model.extract_core(), os.path.join(self.dir, self.save_file))  # save the model

    def train_task_setup(self):
        self.dataset.keep_modalities(self.modal_range)

    def get_output_formatter(self):
        pass

    def get_training_data(self):
        pass

    def get_testing_data(self):
        pass

    def get_early_stop(self, patience: int):
        self.early_stopping = EarlyStop(patience)

    def train(self):
        # move the device to gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.train_task_setup()  # will set up the training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        training, training_label = self.get_training_data()
        for epoch in range(1, self.epochs + 1):
            train_epoch(self.model, epoch, training, training_label, batch_size=self.batch_size,
                        output_formatter=self.get_output_formatter(),
                        optimizer=optimizer, loss_fn=self.get_loss(), device=device)
            if self.early_stopping is not None:
                stop = self.early_stopping.check(
                    validation(self.model, epoch, *self.get_validation_data(), self.get_loss(),
                               self.get_output_formatter(), device))
                if stop:  # if you are to stop
                    break
            # self.save_model()
        self.save_model()

    def get_validation_data(self) -> tuple:
        return self.dataset.validation, self.dataset.validation_label

    def get_loss(self):
        pass


class Classification_HART_Task(Training_Task):
    TASK_NAME = "hart_classification_task"

    def __init__(self, dataset: UserDataLoader, epochs=80, batch_size=64, early_stop=False, lr=0.03, verbose=False, **kwargs):
        super().__init__(dataset, save_file=kwargs["save_file"], save_dir=kwargs["save_dir"],
                         modalities=["accelerometer", "gyroscope"], epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
        self.model = None
        self.create_model()

    def create_model(self):
        self.model = HartClassificationModel(len(self.dataset.classes))

    def train_task_setup(self):
        super().train_task_setup()

    def get_training_data(self):
        return self.dataset.train, self.dataset.train_label

    def get_output_formatter(self):
        return SingleClassificationFormatter(verbose=self.verbose)

    def get_loss(self):
        return nn.CrossEntropyLoss()


class Classification_Task(Training_Task):
    TASK_NAME = "classification_task"

    def __init__(self, dataset: UserDataLoader, modalities=["accelerometer"], epochs=80, batch_size=64, lr=0.03, verbose=False,
                 early_stop=False, **kwargs):
        super().__init__(dataset, save_file=kwargs["save_file"], save_dir=kwargs["save_dir"], modalities=modalities,
                         epochs=epochs, batch_size=batch_size, lr=lr)
        self.dataset = dataset
        self.model = None
        self.create_model()

    def create_model(self):
        self.model = TransformerClassificationModel((self.sequence_length, len(self.modal_range)),
                                                    len(self.dataset.classes), self.num_modal)

    def train_task_setup(self):
        super().train_task_setup()

    def get_training_data(self):
        return self.dataset.train, self.dataset.train_label

    def get_output_formatter(self):
        return SingleClassificationFormatter(verbose=self.verbose)

    def get_loss(self):
        return nn.CrossEntropyLoss()


class Transformation_Classification_Task(Training_Task):
    TASK_NAME = "transformation_classification_task"

    def __init__(self, dataset: UserDataLoader, epochs=80, batch_size=64, modalities=["accelerometer"], verbose=False, lr=0.03, **kwargs):
        super().__init__(dataset, save_file=kwargs["save_file"], save_dir=kwargs["save_dir"], modalities=modalities,
                         epochs=epochs, early_stop_patience=kwargs["early_stopping_patience"], batch_size=batch_size, lr=lr)
        self.model = None
        self.transformations = transform_funcs_vectorized
        self.create_model()

    def train_task_setup(self):
        # this will set up the trainin
        super().train_task_setup()
        self.dataset.transform_sets(self.transformations)  # transform the data

    def create_model(self):
        # TODO: ALLLOW PASSING IN OTHER  transformations
        # TODO: remove the magic number
        self.model = TransformerMultiTaskBinaryClassificationModel((self.sequence_length, 3),
                                                                   len(transform_funcs_vectorized))

    def get_validation_data(self) -> tuple:
        return self.dataset.transform_validation, self.dataset.transform_validation_label

    def get_training_data(self):
        return self.dataset.transform_train, self.dataset.transform_label

    def get_loss(self):
        return MultiTaskLoss(len(self.transformations))

    def get_output_formatter(self):
        return BinaryClassificationFormatter(len(self.transformations), transform_funcs_names, verbose=self.verbose)


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


class TransferLearningClassificationTask(Training_Task):
    TASK_NAME = "transfer_learning_classification_task"

    def __init__(self, dataset: UserDataLoader, pretrained_core: str, feature_extractor_paths: list, modalities=None, verbose=False, lr=0.03,
                 **kwargs):
        self.pretrained_path = pretrained_core
        self.pretrained_ft_extract = feature_extractor_paths
        super().__init__(dataset, save_file=kwargs["save_file"], save_dir=kwargs["save_dir"], modalities=modalities,
                         epochs=kwargs["epochs"], verbose=verbose, lr=lr)

    def create_model(self):
        model = TransferModel(self.pretrained_ft_extract)
        model_sd = model.state_dict()
        state_d = torch.load(self.pretrained_path)
        print(state_d)
        for name, param in state_d.items():
            if name in model_sd:
                if param.size() == model_sd[name].size():
                    print(f'just  right {name}')
                    model_sd[name].copy_(param)
                else:
                    print(f"Size mismatch for layer {name}. Skipping.")
            else:
                print(f"Layer {name} not found in the modified model. Skipping.")

        model.load_state_dict(state_d, strict=False)

        for p in model.parameters():
            p.requires_grad = False

        self.model = TransferModelClassification(model, 13)

    def get_training_data(self):
        return self.dataset.transform_train, self.dataset.transform_label

    def train_task_setup(self):
        # this will set up the trainin
        super().train_task_setup()
        # self.dataset.transform_sets(self.transformations)  # transform the data

    def get_training_data(self):
        return self.dataset.train, self.dataset.train_label

    def get_output_formatter(self):
        return SingleClassificationFormatter(verbose=self.verbose)


    def get_loss(self):
        return nn.CrossEntropyLoss()


class FeatureExtractionTask(Training_Task):
    TASK_NAME = "features_extraction_task"

    def __init__(self, dataset: UserDataLoader, model_type: str, feature_extractor_paths: list, trained_clustering_model: str, save_file: str,
                 save_dir: str, modalities=None, **kwargs):
        self.model_type = model_type
        self.clustering_model = trained_clustering_model
        self.feature_extractor_path = feature_extractor_paths

        super().__init__(dataset, save_file=save_file, save_dir=save_dir, modalities=modalities)
        dataset.combine_training_validation()  # combine the training and the validation data

    def create_model(self):
        if self.model_type == TransferModel.NAME:
            self.model = TransferModel(self.feature_extractor_path, self.clustering_model)

    def get_training_data(self):
        return self.dataset.train, self.dataset.train_label

    def get_testing_data(self):
        return self.dataset.test, self.dataset.test_label

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model
        model.to(device)
        model.eval()
        # now you will through the data and predict batch the input and labels
        data, labels = self.get_training_data()
        training_features = extract_features(model, data, device=device)
        test_data, test_labels = self.get_testing_data()
        test_features = extract_features(model, test_data, device=device)
        # now you will save the features
        data = {
            "train_data": (training_features, labels),
            "testing_data": (test_features, test_labels)
        }
        os.makedirs(self.dir, exist_ok=True)
        hkl.dump(data, os.path.join(self.dir, self.save_file), mode='w')


class Multi_Modal_Clustering_Task(Training_Task):
    TASK_NAME = "multi_modal_clustering_task"

    def __init__(self, dataset: UserDataLoader, modalities=None, epochs=80, batch_size=32, n_clusters=50, verbose=False, lr=0.01, **kwargs):
        # use the silhouette score in kmeans
        self.feature_extractor_path = kwargs["feature_extractor_paths"]
        self.n_clusters = n_clusters
        super().__init__(dataset, save_file=kwargs["save_file"], save_dir=kwargs["save_dir"], modalities=modalities,
                         batch_size=batch_size, epochs=epochs, lr=lr)

        self.dataset = dataset

    def create_model(self):
        self.model = MultiModalTransformer((128, 3), self.feature_extractor_path, 2, self.n_clusters)
        pass

    def train_task_setup(self):
        super().train_task_setup()

    def train(self):
        # you will first attempt to reconstruct the input
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        lr = self.lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = None
        training = self.dataset.train
        loss_op = MMS_loss()
        loss_op.to(device)
        for epoch in range(1, self.epochs + 1):
            # now you will train each batch
            print(f'epoch {epoch}')
            queue_v = None
            use_the_queue = False
            centroid = None
            batch_size = self.batch_size
            permutation = torch.randperm(training.shape[0])

            for i in range(0, training.shape[0], batch_size):
                optimizer.zero_grad()
                # data = data.to(device)
                data = training[permutation[i:i + batch_size]]

                acc = torch.from_numpy(data[:, :, 0:3]).float().to(device)  # get the first triaxial data
                gyro = torch.from_numpy(data[:, :, 3:6]).float().to(device)  # get the secodn triaxial data
                with torch.set_grad_enabled(True):
                    acc_ft, gyro_ft, classified_acc, classified_gyro, recon_loss = self.model(acc, gyro)
                    recon_weight = 50
                    recon_loss = torch.mean(recon_loss) * recon_weight

                    acc_out = classified_acc
                    gyro_out = classified_gyro

                    fused_data = (
                                         acc_out + gyro_out) / 2  # joining the extracted features so that they can be clustered
                    if fused_data.shape[0] < batch_size:
                        continue

                    sim_audio_acc = torch.matmul(acc_ft,
                                                 gyro_ft.t())  # calculates the ismilarity between the gyro and the acc
                    sim_audio_gyro = torch.matmul(gyro_ft,
                                                  acc_ft.t())  # calculates the similarity between the acc and the gyro

                    # calculate the loss
                    loss = loss_op(sim_audio_gyro) + loss_op(sim_audio_acc)
                    # kmeans time
                    queue_v, out, use_the_queue = update_queue(queue_v, use_the_queue, fused_data, batch_size)
                    kmeans = KMeans(n_clusters=self.n_clusters, mode='cosine')
                    labels = kmeans.fit_predict(out)  # this will be the assignments for the current inputs
                    centroid = kmeans.centroids
                    # get the labels for the items in the batch

                    # pass in the extracted features for the acc, the last batch centroid labels and the batch
                    loss_val = cluster_contrastive(acc_out, centroid, labels[-batch_size:], batch_size) \
                               + cluster_contrastive(gyro_out, centroid, labels[-batch_size:], batch_size)
                    loss_val = loss_val / 2

                    loss += loss_val * 1  # clustering lambda

                    loss += recon_loss
                    print(
                        f"epoch {epoch}: loss cluster contrastive {loss_val} and reconstruction + contrastive1: {recon_loss}")
                    
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    # return loss.item(), queue_v, use_the_queue, centroid
        # save the model
        torch.save(self.model.state_dict(), os.path.join(self.dir, self.save_file))

    def train_one_epoch(self, model, opt, data, loss_fn, scheduler):
        pass

"""
    @:param fushed  The cluster contrastive will take a modality that has gone through the gating unit
    @:param centroid the centroids that current kmeans algo has
    @:param labels the centroid assignments for the previous batch
    @:param bs this is the batch size
    
    
"""
# This takes the prev batch as labels this creates a one hot encode
def cluster_contrastive(fushed, centroid, labels, bs):
    S = torch.matmul(fushed, centroid.t())  # get a measure for the similarity between the modality and the centroids

    target = torch.zeros(bs, centroid.shape[0]).to(
        S.device)  # create a target tensor which is of shape batch x centroid shape each row is a sample in the batch and each col is a cnetroid

    target[range(target.shape[ # out of all centroids [0,..1, 0] for all targets
                     0]), labels] = 1  # set the target tensor to be 1 where the class labels are creating a one hot encoding

    S = S - target * (
        0.001)  # subtract the target from the similarity matrix to adjust the values this is regularization, the values that are most similar should be closest to 0

    I2C_loss = nn.functional.nll_loss(nn.functional.log_softmax(S, dim=1),
                                      labels)  # calculate the loss convert the similarities to probabilities
    # this is a form of contrastive loss so that the model is encouraged to assign high prob instances to the correct centroids this loss says how well assigning to centroids
    return I2C_loss


def update_queue(queue, use_the_queue, fuse, batch_size):
    # return queue,fuse,use_the_queue
    bs = batch_size
    fuse2 = fuse.detach()
    fuse2 = fuse2.view(-1, 32, fuse2.shape[
        -1])  # this breaks the fused array into a 3D array of inferred dimension x 32 x last dimension
    fuse2 = fuse2[:, :16, :]  # break into blocks of 16
    fuse2 = fuse2.reshape(-1, fuse2.shape[-1])  # brig back to 2D of dimensions inferred x last dim
    out = fuse.detach()  # when the dimension is inferred this means that it keeps the number of elements the same
    if queue is not None:  # no queue in first round
        if use_the_queue or not torch.all(
                queue[-1, :] == 0):  # queue[2,3840,128] if never use the queue or the queue is not full
            use_the_queue = True
            # print('use queue')
            out = torch.cat(
                (queue, fuse.detach()))  # queue [1920*128] w_t [128*3000] = 1920*3000 out [32*3000] 1952*3000

            # print('out size',out.shape)
        # fill the queue
        queue[bs:] = queue[:-bs].clone()  # move 0-6 to 1-7 place
        queue[:bs] = fuse2
    return queue, out, use_the_queue


PRETEXT_TASKS = {
    Transformation_Classification_Task.TASK_NAME: Transformation_Classification_Task,
    Multi_Modal_Clustering_Task.TASK_NAME: Multi_Modal_Clustering_Task,
    Classification_Task.TASK_NAME: Classification_Task,
    TransferLearningClassificationTask.TASK_NAME: TransferLearningClassificationTask,
    FeatureExtractionTask.TASK_NAME: FeatureExtractionTask,
    Classification_HART_Task.TASK_NAME: Classification_HART_Task
}
