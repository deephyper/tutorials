Hyperparameter Search with Subset Selection for Image Classification
====================================================================

**Author:** Ian Wixom

In this tutorial we will be covering the use of subset selection algorithms. Subset selection offers a more efficient way of training models while minimizing accuracy loss by selecting the most optimal subset out of a given batch. In particular, it is very useful for hyperparameter searches. This tutorial will be using CORDS, a library that offers subset selection. It is important to note that the library only uses PyTorch, and requires use of their dataloaders for each corresponding strategy. Now we will download the required package:

.. code-block:: console

   $ pip install cords
   
Note:
As of the writing of this tutorial, the suggested use is for CPU only for CORDS use. If there is needed use on GPUs such as ThetaGPU, one could edit the module files such that the torchtext module is removed. This module conflicts with PyTorch 1.12+ since CORDS requires 0.10.0 for torchtext, which is needed to use CUDA 11.3. We will use the following models to develop and train our model:

.. code-block:: python

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
  
The dataset will be CIFAR10, an image database of 10 categories of cats, dogs, planes, etc. We can utilize one of their generated datasets to get the training, validation and tests sets. For those who want to load their data manually, use:

.. code-block:: python

  import torchvision
  from torchvision import CIFAR10

**CORDS Prerequisites**

CORDS requires the use of a logger, so one could configure the logger as the following based from their tutorials:

.. code-block:: python

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
  
**CORDS DataLoaders**

In order to use the subset selection algorithms provided by CORDS, their specific DataLoader must be used as well. The code below provides to examples of DataLoaders for the Grad-Match and Random strategies respectively. The variables in the "dss_args" mean the following:

- ``eta``: learning rate
- ``kappa``: used for warm-starting. Determines the amount of epochs that use the full dataset.
- ``num_epochs``: total amount of epochs.
- ``select_every``: determines the frequency of re-evaluating the subsets.
- ``selection_type``: way that the subsets are determined.
- ``valid``: use validation data for subset evaluation.
- ``v1``: whether to use the new omp wrapper or not.
- ``lam``: regularization coefficient.
- ``eps``: the tolerance level of the algorithm's convergence.
 
More information concerning different selection strategies and other variables may be found at https://github.com/decile-team/cords. The Random strategy was included as another potential example if using a GPU.

.. code-block:: python

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
                                 
**Training and Validation**

Training with CORDS is similar to other PyTorch model training scripts, but one key change is the change to the loss function:

- 3 variables to call in a CORDS dataloader
- ``loss = torch.dot(criterion(predictions, labels), weights / weights.sum())``
 
Since weights are used to update the subset, it must be included in any loop or loss funcion calculation. Little to no change is needed for validation compared to previous tutorials.

.. code-block:: python

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
      
**Residual Network Model**

Residual networks are types of deep learning models that utilize residuals from previous layers to improve accuracy and performance. The code below was written by CORDS developer Krishnateja Killamsetty in turn from the "Deep Residual Learning for Image Recognition" paper. It has been slight

.. code-block:: python

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
          
**Defining the Run Function**

The run function within this tutorial is very similar to other DeepHyper tutorials. Within ``ResNet``, there are three inputs: block type, block structure and the number of classes. In this tutorial we do not include the Bottleneck structure seen in models like ResNet34. More information could be found at https://arxiv.org/pdf/1512.03385.pdf.

In order to use the weights from the CORDS dataloader, the reduction of the criterion must be set to ``none``.

.. code-block:: python
  
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
    
**Running the Search**

Now that we have defined our data, training and validation, model, run function and hyperparameter space, we can now conduct a search. Note that in the script file, the code was formatted for MPI usage. Altering all references of device from ``device = torch.device("cuda", rank)`` to ``device = torch.device("cpu")`` would be sufficient to change to CPU.

.. code-block:: python

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
          method="thread",
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
