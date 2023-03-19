
from this import d
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.utils import save_image
import argparse
import shutil
from datetime import timedelta
import json

from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision



def top1_accuracy(model, data_loader):
    '''
    model: model to test
    data_loader: data loader of the test set
    Binary classification performance check for top1 accuracy
    The output of this function is more than the accuracy, it returns top1 accuracy, precision, recall and F1 score
    '''
    model.eval()
    correct = 0
    total = 0
    pred_list = {'ASD':[], 'TD':[]}
    with torch.no_grad():
      
        for data in data_loader:
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = model(images).cuda()
            _, predicted = torch.max(outputs.data, 1)
            #----------

            #----------
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            labels_list = labels.tolist()
            predicted_list = predicted.tolist()
            #----------------------------------
            #changing tensor cpu to numpy
            label = labels.detach().cpu().numpy()
            predict = predicted.detach().cpu().numpy()
            #----------------------------------------
            precision_result = precision(label, predict)
            recall_result = recall(label, predict)

            F1 = (precision_result * recall_result * 2 / (precision_result + recall_result)).mean()

            if len(labels_list)== len(predicted_list):
                for i in range(len(labels_list)):
                    if labels_list[i] != predicted_list[i]:
                        if labels_list[i] == 0:
                            pred_list['ASD'].append(i)# number of index of mis predicted label
                        else:
                            pred_list['TD'].append(i)


    return round(correct / total,6)*100, pred_list , round(precision_result,6)*100,round(recall_result, 6)*100, round(F1, 6)*100

#-------------

def top1_accuracy_level(model, data_loader):
    '''
    If we want to create a split by the ASD score. this is the fuction to get the top1 accuracy
    '''
    model.eval()
    correct = 0
    total = 0
    pred_list = {'0':[], '1':[],'2':[],'3':[],'4':[],'5':[],'6':[]}
    with torch.no_grad():
      
        for data in data_loader:
            images, labels = data[0].cuda(), data[1].cuda()
            outputs = model(images).cuda()
            _, predicted = torch.max(outputs.data, 1)
            #----------

            #----------
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            labels_list = labels.tolist()
            predicted_list = predicted.tolist()
            #----------------------------------
            #changing tensor cpu to numpy
            label = labels.detach().cpu().numpy()
            predict = predicted.detach().cpu().numpy()
            #----------------------------------------

            precision_result = 0
            recall_result = 0
            F1 = 0


            if len(labels_list)== len(predicted_list):
                for i in range(len(labels_list)):
                    if labels_list[i] != predicted_list[i]:
                        if labels_list[i] == 0:
                            pred_list['0'].append(i)# number of index of mis predicted label
                        elif labels_list[i] == 1:
                            pred_list['1'].append(i)
                        elif labels_list[i] == 2:
                            pred_list['2'].append(i)
                        elif labels_list[i] == 3:
                            pred_list['3'].append(i)
                        elif labels_list[i] == 4:
                            pred_list['4'].append(i)
                        elif labels_list[i] == 5:
                            pred_list['5'].append(i)
                        elif labels_list[i] == 6:
                            pred_list['6'].append(i)

    return round(correct / total,6)*100, pred_list , round(precision_result,6)*100,round(recall_result, 6)*100, round(F1, 6)*100