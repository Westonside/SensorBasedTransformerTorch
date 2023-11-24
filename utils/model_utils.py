import torch
from torch import nn


class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks: int):
        super(MultiTaskLoss, self).__init__()
        # self.loss = nn.BCELoss()
        self.losses = [nn.BCEWithLogitsLoss() for _ in range(num_tasks)]
        self.class_accuracy = [0 for _ in range(num_tasks)]
        self.total = 0

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss = 0
        for i in range(pred.shape[1]):
            loss += self.losses[i](pred[:, i], target[:, i])

        return loss


class OutputFormatter:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def get_accuracy(self, pred: torch.Tensor, target: torch.Tensor, epoch, loss=None):
       pass

    def print_accuracy(self, epoch: int, loss=None):
        if self.verbose:
            self.print_accuracy(epoch, loss)


class BinaryClassificationFormatter(OutputFormatter):
    def __init__(self, tasks_count: int, task_names: list[str],  verbose=True):
        super().__init__(verbose)
        self.tasks_count = tasks_count
        self.class_accuracy = [(0, 0) for _ in range(tasks_count)]  # total correct, total
        self.task_names = task_names

    def get_accuracy(self, pred: torch.Tensor, target: torch.Tensor, epoch, loss=None):
        predictions = (pred > 0.5).float()
        for transform in range(self.tasks_count):
            correct, total = self.class_accuracy[transform]
            total += target.shape[0]
            correct += torch.count_nonzero(predictions[:, transform] == target[:, transform]).item()
            self.class_accuracy[transform] = (correct, total)
        if self.verbose:
            self.print_accuracy(epoch, loss=loss)

    def print_accuracy(self, epoch: int, batch=None, loss=None):
        print('Train Epoch: {} | Acc: {} | total loss: {}'.format(
                epoch, self.format_accuracy(), loss if loss is not None else ""))

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

    def get_accuracy(self, pred: torch.Tensor, target: torch.Tensor, epoch, loss=None):
        self.total += pred.shape[0]

        self.correct += torch.count_nonzero(torch.argmax(pred,dim=1)==torch.argmax(target,dim=1)).item()
        super().print_accuracy(epoch, loss)

    def print_accuracy(self, epoch: int, batch=None, loss=None):
        print('Train Epoch: {} | Acc: {:.6f} | total loss: {}'.format(
            epoch, self.correct / self.total), loss if loss is not None else "")


def train_epoch(model, epoch, train_data, train_label, optimizer, loss_fn, output_formatter, batch_size=32, device="cuda"):
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
        batch_x, batch_y = train_data[indices], torch.from_numpy(train_label[indices]) if type(train_label) is not dict else convert_multitask_dict(train_label, indices)
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
        output_formatter.get_accuracy(outputs, batch_y, epoch, loss)









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