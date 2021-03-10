# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:31:58 2020

@author: Maxime Gardoni, Hila Vardi, Niccolò Stefanini
"""
import torch
from torch import nn
from torch.nn import functional as F

#for plotting loss error and accuracy
import matplotlib.pyplot as plt

'''-----------------------  Doubling Training Set  -------------------------'''
#Double the training set by swapping between the pairs of images
#   the input and classes are swapped
#   the target is flipped: 0 becomes 1 and vice versa
def doubleTrainSet(input, classes, target):
    temp_in = input.clone()
    temp_in[:,0,:,:] = input[:,1,:,:]
    temp_in[:,1,:,:] = input[:,0,:,:]
    double_input = torch.cat((input,temp_in),0)

    temp_classes = classes.clone()
    temp_classes[:,0] = classes[:,1]
    temp_classes[:,1] = classes[:,0]
    double_classes = torch.cat((classes,temp_classes),0)

    temp_target = (target == target.new_zeros(target.size())).long()
    double_target = torch.cat((target,temp_target))
    
    return double_input, double_classes, double_target

'''------------------------  Statistic plotting  ---------------------------'''
''' for plotting loss and accuracy during training'''
class stat:
  def __init__(self, name, loss, test_accu):
    self.name = name
    self.loss = loss
    self.test_accu = test_accu



def plot_stat(statList):
  fig, axs = plt.subplots(2, 1, figsize=(15,15))
  axs[0].set_title('Training loss')
  axs[1].set_title('Test Accuracy')
  for elem in statList:
    axs[0].plot(elem.loss, label=elem.name)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')

    axs[1].plot(elem.test_accu, label=elem.name)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Test accuracy')

    axs[0].legend()
    axs[1].legend()

  plt.show()

#%%
'''------------------------  Transfer Learning  ----------------------------'''
'''Neural network for digit classification 
        with a logic function for comparison between the image pairs,
        possibility to use batch normalisation'''
class transferLearnNet(nn.Module):
  def __init__(self, batch_normalization = False):
    super(transferLearnNet, self).__init__()

    self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    self.bn2 = nn.BatchNorm2d(64)

    self.fc1 = nn.Linear(256, 200)
    self.fc2 = nn.Linear(200, 10)

    self.batch_normalization = batch_normalization

  def forward(self, x):
    y = self.conv1(x)
    if self.batch_normalization: y = self.bn1(y)
    y = F.max_pool2d(y, kernel_size=2, stride=2)
    y = F.relu(y)
    y = self.conv2(y)
    if self.batch_normalization: y = self.bn2(y)
    y = F.max_pool2d(y, kernel_size=2, stride=2)
    y = F.relu(y)
    y = F.relu(self.fc1(y.view(-1, 256)))
    y = self.fc2(y)
    return y

#logic function for comparison between pairs of digits
#   returns True if the first digit is lesser or equal to the second digit
def compare_digits(classes1, classes2):
    _, class_img_1= classes1.max(1)
    _, class_img_2= classes2.max(1)
    predicted_classes = class_img_1 <= class_img_2
    return predicted_classes

#train the neural net for digit classification
#   possibility to choose optimiser 
def train_model_transferL(model, train_input, train_classes, train_target, 
                          mini_batch_size, lr, optim_choice = "adam",
                          criterion = nn.CrossEntropyLoss()):

  for e in range(25):
    model.train(True)
    #shuffling between epoch
    randIndex= torch.randperm(train_input.size(0))
    train_input= train_input[randIndex]
    train_target=train_target[randIndex]
    train_classes=train_classes[randIndex]

    for b in range(0, train_input.size(0), mini_batch_size):
      output = model(train_input.narrow(0, b, mini_batch_size))
      loss = criterion(output,train_classes.narrow(0,b,mini_batch_size).long())
      
      model.zero_grad()
      loss.backward()
      
      if optim_choice== "grad_desc":  # manually do the optimization
        with torch.no_grad():
          for p in model.parameters():
            p -= lr * p.grad
      elif optim_choice=="SGD": #optimization with SGD
        optimizer = torch.optim.SGD(model.parameters(), lr = lr)
        optimizer.step()
      elif optim_choice=="adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.step()
      else:
        print("error, optimizer name not recognised")

#Compute the fraction of missclasified data out of the whole data set
def compute_err_perc_transferL(model, input, target, mini_batch_size, 
                               outputIDToCheck="recogn"):
  nb_errors = 0
  model.train(False)
  for b in range(0, input.size(0), mini_batch_size):
    if outputIDToCheck== "recogn":
      input2=input[:, 0:1,:,:].narrow(0, b, mini_batch_size)
      output = model(input2)
      _, predicted_classes = output.max(1)
    elif outputIDToCheck=="compar":
      input1 = model(input[:, 0:1,:,:].narrow(0, b, mini_batch_size))
      input2 = model(input[:, 1:2,:,:].narrow(0, b, mini_batch_size))
      predicted_classes = compare_digits(input1, input2)
    
    for k in range(mini_batch_size):
      if not torch.eq(target[b + k], predicted_classes[k]).all():
        nb_errors = nb_errors + 1  

  return nb_errors/input.shape[0]



'''----------------------  Siamese Neural Network  -------------------------'''
""" a Siamese CNN net, will build a feature space, predict digits, and compare them.
 Performs a binary classification at the end, output on a single node, 
 between 0 and 1 (sigmoid)
 

 Possibility to use an Auxiliary losses (recognition part), batch normalisation and dropout.

 """
class SiamNetDigit(nn.Module):
  def __init__(self, aux_loss = False, batch_normalization=False, dropout=False):
    super(SiamNetDigit, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
    self.bn1= nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    self.bn2= nn.BatchNorm2d(64)
    self.fc1 = nn.Linear(256, 200)
    self.fc2 = nn.Linear(200, 10)
    
    self.bn3 = nn.BatchNorm1d(10)

    self.compar1= nn.Linear(20, 20)
    self.drop1 = nn.Dropout(p = 0.3)

    self.compar2= nn.Linear(20, 20)
    self.compar3= nn.Linear(20, 1)
    self.batch_normalization=batch_normalization
    self.dropout=dropout
    self.aux_loss= aux_loss

  def forward(self, x):
    # recognition part
    recogn=torch.empty(x.shape[0], 10, 2)
    recogn_norma=torch.empty(x.shape[0], 10, 2)

    for i in range (2): #for the 2 images: weight sharing
      y = self.conv1(x[:, i:i+1, :, :])
      if self.batch_normalization: y=self.bn1(y)
      y = F.max_pool2d(y, kernel_size=2, stride=2)
      y = F.relu(y)
      y = self.conv2(y)
      if self.batch_normalization: y=self.bn2(y)
      y = F.max_pool2d(y, kernel_size=2, stride=2)
      y = F.relu(y)
      y = F.relu(self.fc1(y.view(-1, 256)))
      y = self.fc2(y)
      recogn[:,:, i]=y
      recogn_norma[:,:, i]=self.bn3(y)
    recogn_concat= torch.cat((recogn_norma[:,:, 0], recogn_norma[:,:, 1]), -1)

    
    # comparison part
    y=F.relu( self.compar1(recogn_concat))
    if self.dropout:y=self.drop1(y)
    y=F.relu(self.compar2(y))
    if self.dropout:y=self.drop1(y)
    y=self.compar3(y)
    
    compar= torch.sigmoid(y).squeeze()
    return recogn, compar

'''------------  Siamese Neural Network - No Weight Sharing  ---------------'''
''' a siamese network as above, 
    but without weight sharing for the digit recognition'''
class SiamNetDigitNoWeightShare(nn.Module):
  def __init__(self, aux_loss = False, batch_normalization=False, dropout=False):
    super(SiamNetDigitNoWeightShare, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
    self.bn1= nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    self.bn2= nn.BatchNorm2d(64)
    self.fc1 = nn.Linear(256, 200)
    self.fc2 = nn.Linear(200, 10)

    self.bn3 = nn.BatchNorm1d(10)

    #different weights for the second inputs
    self.conv12 = nn.Conv2d(1, 32, kernel_size=3)
    self.bn12= nn.BatchNorm2d(32)
    self.conv22 = nn.Conv2d(32, 64, kernel_size=3)
    self.bn22= nn.BatchNorm2d(64)
    self.fc12 = nn.Linear(256, 200)
    self.fc22 = nn.Linear(200, 10)
    
    self.bn32 = nn.BatchNorm1d(10)

    self.compar1= nn.Linear(20, 20)
    self.drop1 = nn.Dropout(p = 0.3)

    self.compar2= nn.Linear(20, 20)
    self.compar3= nn.Linear(20, 1)
    self.batch_normalization=batch_normalization
    self.dropout=dropout
    self.aux_loss= aux_loss

  def forward(self, x):
    # recognition part
    recogn=torch.empty(x.shape[0], 10, 2)
    recogn_norma=torch.empty(x.shape[0], 10, 2)

    #for the first image in the pair
    y = self.conv1(x[:, 0:1, :, :])
    if self.batch_normalization: y=self.bn1(y)
    y = F.max_pool2d(y, kernel_size=2, stride=2)
    y = F.relu(y)
    y = self.conv2(y)
    if self.batch_normalization: y=self.bn2(y)
    y = F.max_pool2d(y, kernel_size=2, stride=2)
    y = F.relu(y)
    y = F.relu(self.fc1(y.view(-1, 256)))
    y = self.fc2(y)

    recogn[:,:, 0]=y
    recogn_norma[:,:, 0]=self.bn3(y)

    #for the second image in the pair
    y = self.conv12(x[:, 1:2, :, :])
    if self.batch_normalization: y=self.bn12(y)
    y = F.max_pool2d(y, kernel_size=2, stride=2)
    y = F.relu(y)
    y = self.conv22(y)
    if self.batch_normalization: y=self.bn22(y)
    y = F.max_pool2d(y, kernel_size=2, stride=2)
    y = F.relu(y)
    y = F.relu(self.fc12(y.view(-1, 256)))
    y = self.fc22(y)
    recogn[:,:, 1]=y
    recogn_norma[:,:, 1]=self.bn32(y)

    recogn_concat= torch.cat((recogn_norma[:,:, 0], recogn_norma[:,:, 1]), -1)

    # comparison part
    y=F.relu( self.compar1(recogn_concat))
    if self.dropout:y=self.drop1(y)
    y=F.relu(self.compar2(y))
    if self.dropout:y=self.drop1(y)
    y=self.compar3(y)
    
    compar= torch.sigmoid(y).squeeze()
    return recogn, compar

'''-----------------  Siamese Neural Network - Training  -------------------
   a train function for siamese net which support the use of mutiple criterion

'''
def train_siam_model(model, train_input, train_classes, train_target, 
                     mini_batch_size, lr, optim_choice, train_stat):
  criterion_recogn= nn.CrossEntropyLoss()
  criterion_compar= nn.BCELoss()
  criterion_val= nn.BCELoss()
  nb_epoch= 25
  #for plotting losses:
  loss_hist=torch.zeros(nb_epoch)
  accu_hist=torch.zeros(nb_epoch)

  if optim_choice=="SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
      
  if optim_choice=="adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.02) 
  
  for e in range(nb_epoch):
    model.train(True)

    #shuffling between epoch
    randIndex= torch.randperm(train_input.size(0))
    train_input= train_input[randIndex]
    train_target=train_target[randIndex]
    train_classes=train_classes[randIndex]
  
    for b in range(0, train_input.size(0), mini_batch_size):
      input = train_input.narrow(0, b, mini_batch_size)
      recogn, compar = model(input)
      if model.aux_loss:# weighted sum of 2 loss fct
        loss_recogn = criterion_recogn(recogn, 
                                       train_classes.narrow(0, b, mini_batch_size).long())
        loss_compar = criterion_val(compar[:], 
                                    train_target.narrow(0, b, mini_batch_size).float())
        loss = 0.5*loss_recogn + loss_compar 
      else:
        loss = criterion_compar(compar, train_target.narrow(0, b, mini_batch_size).float())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
    #### statistic: plot losses and accuracy during training
    if train_stat: # avoid computing statistic on the whole dataset
                   #when we don't need them
      model.train(False)
      _, compar = model(train_input)
      
      loss_stat = criterion_compar(compar, train_target.float())
      test_acc_stat = 1-compute_err_perc_siam(model, test_input, test_target, 
                                              mini_batch_size, "compar")
      loss_hist[e]=loss_stat
      accu_hist[e]=test_acc_stat
    name= model.__class__.__name__ + " Aux loss: " + str(model.aux_loss) \
                                + ", Batch norm: " + str(model.batch_normalization) \
                                + ", Dropout: " + str(model.dropout)
  if train_stat:
    return stat(name, loss_hist, accu_hist) #for futur plotting

"""Compute the fraction of missclasified data out of the whole data set
   For siamese debugging and exploration: can check the recognition part (auxiliary)
   or the final comparison part
"""  
def compute_err_perc_siam(model, input, target, mini_batch_size, 
                          outputIDToCheck="recogn"):
  nb_errors = 0
  for b in range(0, input.size(0), mini_batch_size):
    input2=input.narrow(0, b, mini_batch_size)
    output = model(input2)
    if type(output)== tuple:
      if outputIDToCheck== "compar":
        output= output[1]
      elif outputIDToCheck== "recogn":
        output= output[0]
    else:
      print("warning not a tuple")
    
    if outputIDToCheck== "compar" :
      predicted_classes = (output>0.5)
       
    else:
      _, predicted_classes = output.max(1)
    
    for k in range(mini_batch_size):
      if not torch.eq(target[b + k], predicted_classes[k]).all():
        nb_errors = nb_errors + 1
  return nb_errors/input.shape[0]
