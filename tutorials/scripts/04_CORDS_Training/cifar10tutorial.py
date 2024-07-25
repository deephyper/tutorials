import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as o
from torch.utils.data import DataLoader

import cords
from cords.utils.data.datasets.SL import gen_dataset
from cords.utils.data.dataloader.SL.adaptive import GradMatchDataLoader, RandomDataLoader, OLRandomDataLoader
from dotmap import DotMap

import deephyper as dh
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator, profile
from deephyper.search.hps import CBO
from deephyper.evaluator.callback import TqdmCallback

import pathlib
import os
import os.path as osp
import logging
import sys

import mpi4py
from mpi4py import MPI

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

search_log_dir = "search_log/"
pathlib.Path(search_log_dir).mkdir(parents=False, exist_ok=True)

search_log_dir = "search_log/"
pathlib.Path(search_log_dir).mkdir(parents=False, exist_ok=True)

def load_data():
    train_ds, valid_ds, test_ds, num_cls = gen_dataset('/lus/grand/projects/datascience/ianwixom/expcifar/', 'cifar10', None, isnumpy=False)
    
    return train_ds, valid_ds
        
def __get_logger(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    plain_formatter = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                      datefmt="%m/%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    s_handler = logging.StreamHandler(stream=sys.stdout)
    s_handler.setFormatter(plain_formatter)
    s_handler.setLevel(logging.INFO)
    logger.addHandler(s_handler)
    f_handler = logging.FileHandler(os.path.join(results_dir, "results.log"))
    f_handler.setFormatter(plain_formatter)
    f_handler.setLevel(logging.DEBUG)
    logger.addHandler(f_handler)
    logger.propagate = False
    return logger

results_dir = osp.abspath(osp.expanduser('results'))
logger = __get_logger(results_dir)

def RandomDL(dhargs):
    dss_args = dict(select_every = 10,
                    kappa = 0,
                    fraction = 0.125,
                    device = 'cuda')

    dss_args = DotMap(dss_args)
    return RandomDataLoader(dhargs['train_d'], dss_args, logger,
                               batch_size=dhargs['batch'], shuffle=True, pin_memory=True)

def SubsetDL(model, criterion, dhargs):
    dss_args = dict(model = model,
                loss = criterion,
                eta = dhargs['lr'],
                num_classes = 10,
                device = 'cpu',
                fraction = 0.125,
                kappa = 0,
                select_every = 10,
                linear_layer = False,
                selection_type = 'PerBatch',
                valid = False,
                v1 = True,
                lam = dhargs['reg_coeff'],
                eps = 0.1)

    dss_args = DotMap(dss_args)

    return GradMatchDataLoader(dhargs['train_d'], dhargs['valid_d'], dss_args,
                               logger, batch_size=dhargs['batch'],
                               shuffle=True, pin_memory=True)

def train(model, criterion, optimizer, scheduler, epochs, dl, valid_dl):
    acc_max = 0
    for i in range(epochs):
        model.train()
        for _, (features, labels, weights) in enumerate(dl):
            features, labels, weights = features.to(device), labels.to(device, non_blocking = True), weights.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = torch.dot(criterion(predictions, labels), weights / weights.sum())
            loss.backward()
            optimizer.step()

        acc = valid(model, criterion, optimizer, valid_dl)
        print(f"The accuracy of the model on epoch {i} is {acc*100:1f}%")

        if acc_max < acc:
            acc_max = acc

    return acc_max

def valid(model, optimizer, dl):
    model.eval()
    correct = 0
    with torch.no_grad():
        for _, (features, labels) in enumerate(dl):
            features, labels = features.to(device), labels.to(device, non_blocking = True)
            predictions = model(features)
            correct += (predictions.argmax(1) == labels).type(torch.float).sum().item()
    return correct / len(dl.dataset)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = f.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = f.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.embDim = 8 * self.in_planes * block.expansion

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = f.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = f.avg_pool2d(out, 4)
                e = out.view(out.size(0), -1)
        else:
            out = f.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = f.avg_pool2d(out, 4)
            e = out.view(out.size(0), -1)
        out = self.linear(e)
        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim
    
def run(config: dict):
    acc = 0
    batch = 20

    train_ds, valid_ds = load_data()

    train_dl = DataLoader(train_ds, batch_size = batch, shuffle = True, num_workers = 0, pin_memory = False)
    valid_dl = DataLoader(valid_ds, batch_size = batch, shuffle = True, num_workers = 0, pin_memory = False)

    dhargs = {'train_d': train_dl, 'valid_d': valid_dl, 'lr': config['lr'], 'batch': batch}
    block_struct = [2, 2, 2, 2]
    model = ResNet(BasicBlock, block_struct, 10).to(device)

    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = optdict[config["optimizers"]](model.parameters(), lr = config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["t_max"])

    subset = SubsetDL(model, criterion, optimizer, dhargs)
    acc = train(model, criterion, optimizer, scheduler, epochs, subset, valid_dl)

    return acc

if __name__ == "__main__":
    method_kwargs = {"callbacks": [TqdmCallback()]}

    prob = HpProblem()

    prob.add_hyperparameter((1e-05,5e-01, 'log-uniform'), "lr")
    prob.add_hyperparameter((0.1,0.95), "momentum")
    prob.add_hyperparameter((1e-5,1e-3, 'log-uniform'), "weightdecay")
    prob.add_hyperparameter((0.01, 10.0, 'log-uniform'), 'regularization')
    prob.add_hyperparameter((1, 50), "t_max")

    epochs = 50

    with Evaluator.create(
        run,
        method="mpicomm",
        method_kwargs=method_kwargs
    ) as evaluator:
        if evaluator is not None:
            print(f"Creation of the Evaluator done with {evaluator.num_workers} worker(s)")

            # Search creation
            search = CBO(prob,
                        evaluator)

            # Search execution
            print("Starting the search...")
            prelim_result = search.search(max_evals = 50)
            print("Search is done")

            prelim_result.to_csv(os.path.join(search_log_dir, f"results.csv"))
            i_max = prelim_result.objective.argmax()

            print(f"\nThe default configuration has an accuracy of {prelim_result['objective'].iloc[0]:.3f}. \n" \
                f"The best configuration found by DeepHyper has an accuracy {prelim_result['objective'].iloc[i_max]:.3f}, \n" \
                f"finished after {prelim_result['timestamp_gather'].iloc[i_max]-prelim_result['timestamp_submit'].iloc[i_max]:.2f} seconds of search.\n")
