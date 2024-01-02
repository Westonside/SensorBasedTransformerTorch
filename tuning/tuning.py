import wandb
import pprint
import numpy as np
import torch
import torch.optim as optim
from model_impl.model import MultiModalTransformer
from preprocess.dataset_loading import load_datasets
import torch.nn as nn
import os
from fast_pytorch_kmeans import KMeans
import hickle

wandb.login()

sweep_config = {
    'method': 'grid'
}
metric = {
    'name': 'loss',
    'goal': 'minimize'
}
sweep_config['metric'] = metric




parameters_dict = {
    # 'features': {
    #     'values': [64, 128, 256, 512, 1024, 2048]
    # },
    # 'num_clusters':{
    #     'values': [20,25,30,40,50]
    # },
    'batch_size': {
        'values': [32,64,128,256,512,1024,2048]
    },
    # 'lr':{
    #     'values':[0.001, .0009]
    # },

    # 'recon_size':{
    #     'values': [64, 128, 256, 512, 1024, 2048]
    # },
    # 'projection_size':{
    #     'values': [800, 1000, 1500, 2000, 3000]
    # }
}
# parameters_dict = {
#
# }

# parameters_dict = {
#     'learning_rate_scheduler':{
#         'values': ['none','linear', 'constant', 'exponential', 'step']
#     }
# }

sweep_config['parameters'] = parameters_dict
parameters_dict.update({
    'epochs': {
        'value': 120}
    })

combos = ((np.prod([len(parameters_dict[key]['values']) for key in parameters_dict if key != 'epochs'])))
# combos = ((np.prod([len(parameters_dict[key]['values']) for key in parameters_dict ])))


print("Tuning a total " + str(combos) + " combinations")

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="sweeping_multi_modal_clustering")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        # lr = config.lr

        lr = 0.0009



        data = load_datasets(['MotionSense', 'UCI', 'WISDM'], path='../datasets/processed/')
        cluster_size = 50


        # network = MultiModalTransformer((128,6), ["../models/accelerometer_extract.pt", "../models/gyroscope_extract.pt"], 2,cluster_size=config.num_clusters, embed_dim=config.features, reconstruction_size=config.recon_size, projection_dim=config.projection_size)
        network = MultiModalTransformer((128,6), ["../models/accelerometer_extract.pt", "../models/gyroscope_extract.pt"], 2, cluster_size=50)

        optimizer = optim.Adam(network.parameters(), lr)

        if lr == 0.1:
            lam = lambda epoch: lr * (0.5 ** (epoch // 15))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lam])
        else:
            scheduler = None

        network.to(device)
        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, data, optimizer, batch_size=config.batch_size, scheduler=scheduler, epoch=epoch)
            wandb.log({"loss": avg_loss, "epoch": epoch})

def train_epoch(model, data, optimizer, batch_size=64, scheduler=None, epoch=None):
    training = data.train
    permutation = torch.randperm(training.shape[0])
    cmu_loss = 0
    loss_op = MMS_loss()
    queue_v = None
    use_the_queue = False
    centroid = None
    for i in range(0, training.shape[0], batch_size):
        optimizer.zero_grad()
                # data = data.to(device)
        data = training[permutation[i:i + batch_size]]

        acc = torch.from_numpy(data[:, :, 0:3]).float().to(device) #get the first triaxial data
        gyro = torch.from_numpy(data[:, :, 3:6]).float().to(device) # get the secodn triaxial data
        with torch.set_grad_enabled(True):
            acc_ft, gyro_ft, classified_acc, classified_gyro, recon_loss = model(acc,gyro)
            print(acc_ft.shape, gyro_ft.shape, classified_acc.shape, classified_gyro.shape)
            recon_weight = 50
            recon_loss = torch.mean(recon_loss) * recon_weight
            wandb.log({"batch_loss_reconstruction": recon_loss.item()})
            acc_out = classified_acc
            gyro_out = classified_gyro

            fused_data = (acc_out + gyro_out) / 2 # joining the extracted features so that they can be clustered
            if fused_data.shape[0] < batch_size:
                continue
            #wandb: 	epochs: 1
            #wandb: 	features: 64

            #wandb: 	recon_size: 64
            #torch.Size([64, 64]) torch.Size([1024, 64]) torch.Size([64, 1024])
            print(acc_ft.shape, gyro_ft.t().shape, gyro_ft.shape)
            sim_audio_acc = torch.matmul(acc_ft, gyro_ft.t()) #calculates the ismilarity between the gyro and the acc
            sim_audio_gyro = torch.matmul(gyro_ft, acc_ft.t()) # calculates the similarity between the acc and the gyro

            # calculate the loss
            loss = loss_op(sim_audio_gyro) + loss_op(sim_audio_acc)
            # kmeans time
            queue_v,  out, use_the_queue = update_queue(queue_v, use_the_queue, fused_data, batch_size)
            kmeans = KMeans(n_clusters=model.cluster_size, mode='cosine')
            labels = kmeans.fit_predict(out)
            centroid = kmeans.centroids
            # get the labels for the items in the batch
            loss_val = cluster_contrastive(acc_out, centroid, labels[-batch_size:], batch_size) \
                               + cluster_contrastive(gyro_out, centroid, labels[-batch_size:], batch_size)
            loss += loss_val * 1 # clustering lambda
            wandb.log({"clustering contrastive loss": loss_val.item()})
            loss += recon_loss
            cmu_loss += loss
            print(f"epoch: {epoch} loss cluster contrastive {loss_val} and reconstruction + contrastive1: {recon_loss}")
            loss.backward()
            if scheduler is not None:
                scheduler.step()
            optimizer.step()
            wandb.log({"batch_loss":loss.item()})
        return cmu_loss / training.shape[0]

def cluster_contrastive(fushed,centroid,labels,bs):
    S = torch.matmul(fushed, centroid.t()) # get the similarity between the fused data and the centroids

    target = torch.zeros(bs, centroid.shape[0]).to(S.device) # create a target tensor

    target[range(target.shape[0]), labels] = 1 # set the target tensor to be 1 where the label is

    S = S - target * (0.001) # subtract the target from the similarity matrix

    I2C_loss = nn.functional.nll_loss(nn.functional.log_softmax(S, dim=1), labels) # calculate the loss
    return I2C_loss


def update_queue(queue,use_the_queue,fuse, batch_size):
    # return queue,fuse,use_the_queue
    bs = batch_size
    fuse2 = fuse.detach()
    fuse2 = fuse2.view(-1, 32, fuse2.shape[-1]) # this breaks the fused array into a 3D array of inferred dimension x 32 x last dimension
    fuse2 = fuse2[:,:16,:] # break into blocks of 16
    fuse2 = fuse2.reshape(-1, fuse2.shape[-1]) # brig back to 2D of dimensions inferred x last dim
    out = fuse.detach() # when the dimension is inferred this means that it keeps the number of elements the same
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


class MMS_loss(nn.Module):
    def __init__(self):
        super(MMS_loss, self).__init__()

    def forward(self, S, margin=0.001):
        deltas = margin * torch.eye(S.size(0)).to(S.device)
        S = S - deltas

        target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
        I2C_loss = nn.functional.nll_loss(nn.functional.log_softmax(S, dim=1), target)
        C2I_loss = nn.functional.nll_loss(nn.functional.log_softmax(S.t(), dim=1), target)
        loss = I2C_loss + C2I_loss
        return loss

wandb.agent(sweep_id, train, count=combos)

