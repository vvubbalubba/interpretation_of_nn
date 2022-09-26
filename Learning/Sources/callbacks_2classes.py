import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
import cv2
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import io
from pytorch_lightning.callbacks import Callback
from PIL import Image
import itertools 

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 3 * 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    image = Image.open(buf)
    image = image.convert("RGB")
    transform = transforms.ToTensor()
    image = transform(image)
    
    return image

def get_true_classes(input_classes):
    true_classes = []
    for i in range(len(input_classes)):
        input_classes[i] = input_classes[i].to('cpu')
        for item in input_classes[i]:
            true_classes.append(item.item()) 
    out = np.array(true_classes, dtype=np.int64)
    return out

def get_predicted_classes(input_classes):
    pred_classes = []
    for i in range(len(input_classes)):
        input_classes[i] = input_classes[i].to('cpu')
        for item in input_classes[i]:
            pred_classes.append(item.detach().numpy()) 
    pred_classes = np.array(pred_classes, dtype=object).astype(float)
    out = np.argmax(pred_classes, axis = 1)
    return out

def get_classes_probs(input_classes):
    pred_classes = []
    for i in range(len(input_classes)):
        input_classes[i] = input_classes[i].to('cpu')
        for item in input_classes[i]:
            pred_classes.append(item.detach().numpy()) 
    pred_classes = np.array(pred_classes, dtype=object).astype(float)
    return pred_classes

def log_confusion_matrix(epoch, true, pred, true_val, pred_val, class_names, writer):
    
    # Calculate the confusion matrix using sklearn.metrics
    cm = confusion_matrix(true, pred)
    cm_val = confusion_matrix(true_val, pred_val)
    
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)
    
    figure_val = plot_confusion_matrix(cm_val, class_names=class_names)
    cm_image_val = plot_to_image(figure_val)
    
    # Log the confusion matrix as an image summary.
    writer.add_image('Confusion matrix train', cm_image, epoch)
    writer.add_image('Confusion matrix val', cm_image_val, epoch)
    
def log_running_loss(running_loss, val_loss, epoch, writer):
    writer.add_scalars('Loss', {'Train': running_loss,
                               'Val': val_loss}, epoch)
    
def log_accuracy(true, pred, true_val, pred_val, epoch, writer):
    writer.add_scalars('Accuracy', {'Train': accuracy_score(true, pred),
                                   'Val': accuracy_score(true_val, pred_val)}, epoch)
    
def log_rocauc(true, pred, true_val, pred_val, epoch, writer):
    writer.add_scalars('ROC_AUC', {'Train': roc_auc_score(true, pred), 
                                   'Val': roc_auc_score(true_val, pred_val)}, epoch)
    
def log_precision(true, pred, true_val, pred_val, epoch, writer):
    train_prec = precision_score(true, pred)
    val_prec = precision_score(true_val, pred_val)
    writer.add_scalars('Precision', {'Train': train_prec, 'Val': val_prec}, epoch)
    
def log_recall(true, pred, true_val, pred_val, epoch, writer):
    train_rec = recall_score(true, pred)
    val_rec = recall_score(true_val, pred_val)
    writer.add_scalars('Recall', {'Train': train_rec, 'Val': val_rec}, epoch)

class callback(Callback):
    
    def __init__(self): 
        pass
    
    def on_epoch_begin(self, epoch):
        self.epoch = epoch
        
    def on_epoch_end(self, t, p, tv, pv, class_names, running_loss, val_loss, writer):
        true = get_true_classes(t)
        pred = get_predicted_classes(p)
        probs = get_classes_probs(p)
        true_val = get_true_classes(tv)
        pred_val = get_predicted_classes(pv)
        probs_val = get_classes_probs(pv)
        log_confusion_matrix(self.epoch, true, pred, true_val, pred_val, class_names, writer)
        log_running_loss(running_loss, val_loss, self.epoch, writer)
        log_accuracy(true, pred, true_val, pred_val, self.epoch, writer)
        log_rocauc(true, probs[:,1], true_val, probs_val[:,1], self.epoch, writer)
        log_precision(true, pred, true_val, pred_val, self.epoch, writer)
        log_recall(true, pred, true_val, pred_val, self.epoch, writer)