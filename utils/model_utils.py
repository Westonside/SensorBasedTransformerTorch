import os

import numpy as np
import torch
from torch import nn



class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks: int):
        super(MultiTaskLoss, self).__init__()
        self.loss = nn.BCELoss()
        # self.losses = [nn.BCELoss() for _ in range(num_tasks)]
        self.class_accuracy = [0 for _ in range(num_tasks)]
        self.total = 0

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss = 0
        for transform_index, transform in enumerate(pred):
            loss += self.loss(transform, target[transform_index])
        return loss


class OutputFormatter:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def get_accuracy(self, pred: torch.Tensor, target: torch.Tensor, epoch, loss=None, valid=False):
       pass

    def print_accuracy(self, epoch: int, loss=None, valid=False):
        if self.verbose:
            self.print_accuracy(epoch, loss)


class BinaryClassificationFormatter(OutputFormatter):
    def __init__(self, tasks_count: int, task_names: list[str],  verbose=True):
        super().__init__(verbose)
        self.tasks_count = tasks_count
        self.class_accuracy = [(0, 0) for _ in range(tasks_count)]  # total correct, total
        self.task_names = task_names

    def get_accuracy(self, pred: torch.Tensor, target: torch.Tensor, epoch, loss=None, valid=False):
        predictions = (pred > 0.5).float()
        for i in range(self.tasks_count):
            correct, total = self.class_accuracy[i]
            correct += torch.count_nonzero(predictions[i] == target[i]).item()
            total += target[i].shape[0]
            self.class_accuracy[i] = (correct, total)

        self.print_accuracy(epoch, loss=loss, valid=valid)
    def print_accuracy(self, epoch: int, loss=None, valid=False):
        print('{} Epoch: {} | Acc: {} | total loss: {}'.format(
                "Train" if not validation else "Validation" ,epoch, self.format_accuracy(), loss if loss is not None else ""))

    def format_accuracy(self):
        output = ""
        for i in range(self.tasks_count):
            correct, total = self.class_accuracy[i]
            output += "{}: {:.6f} ".format(self.task_names[i], correct / total)
        return output


class SingleClassificationFormatter(OutputFormatter):
    def __init__(self, verbose=True):
        super().__init__(verbose)
        self.correct = 0
        self.total = 0

    def get_accuracy(self, pred: torch.Tensor, target: torch.Tensor, epoch, loss=None, valid=False):
        self.total += pred.shape[0]
        self.correct += torch.count_nonzero(torch.argmax(pred,dim=1)==torch.argmax(target,dim=1)).item()
        super().print_accuracy(epoch, loss, valid)

    def print_accuracy(self, epoch: int, loss=None, valid=False):
        print('{} Epoch: {} | Acc: {:.6f} | total loss: {}'.format(
            "Train" if not valid else "Validation",epoch, (self.correct / self.total), loss if loss is not None else ""))



def extract_features(model, data, device, batch_size=128):
    outputs = []
    for i in range(0, data.shape[0], batch_size):
        model_in = data[i:i + batch_size]
        acc = torch.from_numpy(model_in[:,:,:3]).float().to(device)
        gyro = torch.from_numpy(model_in[:,:,3:]).float().to(device)
        outputs.append(model(acc,gyro).detach().cpu())
    print('feature extraction has been completed')
    return np.vstack(data)



def train_epoch(model, epoch, train_data, train_label, optimizer, loss_fn, output_formatter, batch_size=128, device="cuda"):
    # print("Training")
    train_loss = 0.
    train_acc = 0.
    correct = 0
    total = 0
    model.train()

    permutation = torch.randperm(train_data.shape[0])
    for i in range(0, train_data.shape[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = train_data[indices], torch.from_numpy(train_label[indices]) if type(train_label) is not dict else torch.stack([torch.from_numpy(train_label[key][indices]) for key in train_label.keys()])
        # batch_x = torch.from_numpy(batch_x).float()
        batch_x = torch.from_numpy(batch_x).float().to(device)
        batch_y = (batch_y).float().to(device)
        # batch_y = (batch_y).float()
        outputs = model(batch_x)
        # pred = torch.argmax(outputs, dim=1)
        loss = loss_fn(outputs, batch_y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        output_formatter.get_accuracy(outputs, batch_y, epoch, loss, valid=False)



class EarlyStop():
    def __init__(self, patience: int, testing=False):
       self.count = 0
       self.patience = patience
       self.best_loss = float("inf")
       self.testing_counter = 0
       self.testing = testing

    def check(self, valid_loss: int) -> bool:
        if self.testing:
            self.testing_counter +=1
            if self.testing_counter >= 3:
                return True
        if self.best_loss > valid_loss:
            self.best_loss = valid_loss
            self.count = 0
        else:
            self.count += 1
        if self.count > self.patience:
            return True
        else:
            return False


def validation(model, epoch, validation_data, validation_labels, loss_fn, output_formatter, device):
    valid_loss = 0.
    valid_acc = 0.
    correct = 0
    total = 0
    model.eval()

    permutation = torch.randperm(validation_data.shape[0])
    batch_size = 64
    for i in range(0, validation_data.shape[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = validation_data[indices], torch.from_numpy(validation_labels[indices]) if type(
            validation_labels) is not dict else torch.stack(
            [torch.from_numpy(validation_labels[key][indices]) for key in validation_labels.keys()])
        # batch_x = torch.from_numpy(batch_x).float()
        batch_x = torch.from_numpy(batch_x).float().to(device)
        batch_y = (batch_y).float().to(device)
        # batch_y = (batch_y).float()
        outputs = model(batch_x)
        # pred = torch.argmax(outputs, dim=1)
        loss = loss_fn(outputs, batch_y)
        valid_loss += loss.item()
        output_formatter.get_accuracy(outputs, batch_y, epoch, loss, valid=True)

    return valid_loss




def temp_train_epoch(model, epoch, train_data, train_label, optimizer, loss_fn, output_formatter, batch_size=64, device="cuda"):
    # print("Training")
    train_loss = 0.
    train_acc = 0.
    correct = 0
    total = 0
    model.train()

    permutation = torch.randperm(train_data.shape[0])
    for i in range(0, train_data.shape[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = train_data[indices], torch.from_numpy(train_label[indices]) if type(
            train_label) is not dict else torch.stack(
            [torch.from_numpy(train_label[key][indices]) for key in train_label.keys()])
        # batch_x = torch.from_numpy(batch_x).float()
        batch_x = torch.from_numpy(batch_x).float().to(device)
        batch_y = (batch_y).float().to(device)
        # batch_y = (batch_y).float()
        outputs = model(batch_x[:,:,:3], batch_x[:,:,3:])
        # pred = torch.argmax(outputs, dim=1)
        loss = loss_fn(outputs, batch_y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        output_formatter.get_accuracy(outputs, batch_y, epoch, loss)


    # scores = np.vstack(scores)
    # y = arg_max_predict(scores)
    print('done')


def temp_validation(model, epoch, validation_data, validation_labels, loss_fn, output_formatter, device):
    valid_loss = 0.
    valid_acc = 0.
    correct = 0
    total = 0
    model.eval()

    permutation = torch.randperm(validation_data.shape[0])
    batch_size = 64
    for i in range(0, validation_data.shape[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = validation_data[indices], torch.from_numpy(validation_labels[indices]) if type(
            validation_labels) is not dict else torch.stack(
            [torch.from_numpy(validation_labels[key][indices]) for key in validation_labels.keys()])
        # batch_x = torch.from_numpy(batch_x).float()
        batch_x = torch.from_numpy(batch_x).float().to(device)
        batch_y = (batch_y).float().to(device)
        # batch_y = (batch_y).float()
        outputs = model(batch_x[:, :, :3], batch_x[:, :, 3:])
        loss = loss_fn(outputs, batch_y)
        valid_loss += loss.item()
        output_formatter.get_accuracy(outputs, batch_y, epoch, loss, validation=True)

    return valid_loss


def arg_max_predict(scores):
    y_final = np.zeros((scores.shape[0], scores.shape[1]))
    arg_y = np.argmax(scores, axis=1)
    for i in range(scores.shape[0]):
        y_final[i][arg_y[i]] = 1
    return y_final






def convert_multitask_dict(labels, indicies):
    # torch.argmax(torch.tensor([x[indicies] for x in labels.values()]), dim=0)
    values = torch.asarray([labels[key][indicies] for key in labels.keys()])
    options = torch.argmax(values, dim=0)
    outputs = torch.zeros(len(indicies), len(labels.keys()))  #this will be the ouptut
    # torch.asarray([labels[key][indicies] for key in labels.keys()])[:, 0]
    for i, value in enumerate(options):
        # print(values[:,i])
        if value == 0 and torch.max(values, dim=0) == 0:
            continue
        outputs[i][value] = 1

    return outputs





def save_model(model, folder, name): #TODO: Add model.extract_core to the model
    model = model.extract_core()
    model = freeze_model(model)  # freeze the model
    os.makedirs(folder, exist_ok=True)
    torch.jit.script(model)
    torch.jit.save(model, os.path.join(folder, name))


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model




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