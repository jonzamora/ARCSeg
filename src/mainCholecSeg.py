'''
Image Segmentation using SegNet
'''

import argparse
import os
import shutil

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

import utils
from model.unet import UNet
from model.segnet import SegNet
from datasets.cholecSegDataLoader import cholecSegDataset

import matplotlib.pyplot as plt
import mplcursors
import mpld3
from mpld3 import plugins

parser = argparse.ArgumentParser(description='PyTorch SegNet Training')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
            help='number of data loading workers (default: 4)') # 4 * num_gpu
parser.add_argument('--epochs', default=20, type=int, metavar='N',
            help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
parser.add_argument('--batchSize', default=2, type=int,
            help='Mini-batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
            metavar='LR', help='initial learning rate')
parser.add_argument('--wd', '--weight_decay', default=0.00005, type=float,
            help='initial learning rate')
parser.add_argument('--bnMomentum', default=0.1, type=float,
            help='Batch Norm Momentum (default: 0.1)')
parser.add_argument('--imageSize', default=256, type=int,
            help='height/width of the input image to the network')
parser.add_argument('--resizedImageSize', default=224, type=int,
            help='height/width of the resized image to the network')
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N',
            help='print frequency (default:1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
            help='evaluate model on validation set')
parser.add_argument('--save-dir', dest='save_dir',
            help='The directory used to save the trained models',
            default='save_temp', type=str)
parser.add_argument('--saveTest', default='False', type=str,
            help='Saves the validation/test images if True')

best_prec1 = np.inf

'''
check GPU availability
'''
use_gpu = torch.cuda.is_available()
curr_device = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(curr_device)


print("CUDA AVAILABLE:", use_gpu)
print("CURRENT DEVICE:", curr_device, torch.cuda.device(curr_device))
print("DEVICE NAME:", device_name)

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    if args.saveTest == 'True':
        args.saveTest = True
    elif args.saveTest == 'False':
        args.saveTest = False

    # Check if the save directory exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cudnn.benchmark = True

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.resizedImageSize, args.resizedImageSize), interpolation=Image.NEAREST),
            #transforms.TenCrop(args.resizedImageSize),
            #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            #transforms.Lambda(lambda normalized: torch.stack([transforms.Normalize([0.295, 0.204, 0.197], [0.221, 0.188, 0.182])(crop) for crop in normalized]))
            #transforms.RandomResizedCrop(224, interpolation=Image.NEAREST),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.RandomCrop(args.resizedImageSize),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize((args.resizedImageSize, args.resizedImageSize), interpolation=Image.NEAREST),
            transforms.ToTensor()
            #transforms.Normalize([0.295, 0.204, 0.197], [0.221, 0.188, 0.182])
        ]),
    }

    # Data Loading

    data_dir = '/home/jonzamora/Desktop/arclab/ARCSeg/src/datasets/cholec'
    json_path = '/home/jonzamora/Desktop/arclab/ARCSeg/src/datasets/classes/cholecSegClasses.json'

    image_datasets = {x: cholecSegDataset(os.path.join(data_dir, x), data_transforms[x],
                        json_path) for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=args.batchSize,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  pin_memory=True)
                  for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    # Get the dictionary for the id and RGB value pairs for the dataset
    classes = image_datasets['train'].classes
    print("\nCLASSES:", classes, "\n")
    key = utils.disentangleKey(classes)
    print("KEY", key, "\n")
    num_classes = len(key)
    print("NUM CLASSES:", num_classes, "\n")
    
    # Initialize the model
    model = UNet(num_classes)
    #model = SegNet(0.1, num_classes)

    # # Optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         #args.start_epoch = checkpoint['epoch']
    #         pretrained_dict = checkpoint['state_dict']
    #         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}
    #         model.state_dict().update(pretrained_dict)
    #         model.load_state_dict(model.state_dict())
    #         print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))
    #
    #     # # Freeze the encoder weights
    #     # for param in model.encoder.parameters():
    #     #     param.requires_grad = False
    #
    #     optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
    # else:
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)

    # Load the saved model
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    print(model)

    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss() # used by UNet
    #criterion = nn.BCEWithLogitsLoss() # used by SegNet

    # Use a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if use_gpu:
        model.cuda()
        criterion.cuda()

    # Initialize an evaluation Object
    evaluator = utils.Evaluate(key, use_gpu)

    train_losses = []
    val_losses = []

    total_iou = []
    total_precision = []
    total_recall = []
    total_f1 = []

    #iou_classwise = []
    #precision_classwise = []
    #recall_classwise = []
    #f1_classwise = []

    for epoch in range(args.epochs):
        #adjust_learning_rate(optimizer, epoch)
        print('>>>>>>>>>>>>>>>>>>>>>>>Training<<<<<<<<<<<<<<<<<<<<<<<')
        # One Epoch
        train(dataloaders['train'], model, criterion, optimizer, scheduler, epoch, key, train_losses)

        print('>>>>>>>>>>>>>>>>>>>>>>>Testing<<<<<<<<<<<<<<<<<<<<<<<')
        validate(dataloaders['test'], model, criterion, epoch, key, evaluator, val_losses)

        # Calculate the metrics
        print('>>>>>>>>>>>>>>>>>> Evaluating the Metrics <<<<<<<<<<<<<<<<<')
        IoU = evaluator.getIoU()

        print('Mean IoU: {}, Class-wise IoU: {}'.format(torch.mean(IoU), IoU))
        total_iou.append(torch.mean(IoU))
        #iou_classwise.append(IoU.cpu().detach().numpy())

        PRF1 = evaluator.getPRF1()
        precision, recall, F1 = PRF1[0], PRF1[1], PRF1[2]

        print('Mean Precision: {}, Class-wise Precision: {}'.format(torch.mean(precision), precision))
        total_precision.append(torch.mean(precision))
        #precision_classwise.append(precision.cpu().detach().numpy())

        print('Mean Recall: {}, Class-wise Recall: {}'.format(torch.mean(recall), recall))
        total_recall.append(torch.mean(recall))
        #recall_classwise.append(recall.cpu().detach().numpy())

        print('Mean F1: {}, Class-wise F1: {}'.format(torch.mean(F1), F1))
        total_f1.append(torch.mean(F1))
        #f1_classwise.append(F1.cpu().detach().numpy())

        evaluator.reset()

    # loss curves
    plt.plot(range(1, args.epochs+1), train_losses, color='blue')
    plt.plot(range(1, args.epochs+1), val_losses, color='black')
    plt.legend(["Train Loss", "Val Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    figure_name = "cholec_segnet_" + str(args.epochs) + "-epochs_" + str(args.lr) + "-lr_" + str(args.batchSize) + "-batchSize.pdf"
    plt.savefig("loss_curves/" + figure_name)

    # mean accuracy curves
    plt.clf()
    plt.plot(range(1, args.epochs+1), total_iou, color='blue')
    plt.plot(range(1, args.epochs+1), total_precision, color='red')
    plt.plot(range(1, args.epochs+1), total_recall, color='magenta')
    plt.plot(range(1, args.epochs+1), total_f1, color='black')
    plt.legend(["Mean IoU", "Mean Precision", "Mean Recall", "Mean F1"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    figure_name = "cholec_segnet_" + str(args.epochs) + "-epochs_" + str(args.lr) + "-lr_" + str(args.batchSize) + "-batchSize.pdf"
    plt.savefig("accuracy_curves/" + figure_name)

    # classwise accuracy curves
    '''
    iou_classwise = np.array(iou_classwise)
    precision_classwise = np.array(precision_classwise)
    recall_classwise = np.array(recall_classwise)
    f1_classwise = np.array(f1_classwise)

    iou_classwise = np.dstack(iou_classwise)
    precision_classwise = np.dstack(precision_classwise)
    recall_classwise = np.dstack(recall_classwise)
    f1_classwise = np.dstack(f1_classwise)
    
    iou_fig = plt.figure()
    iou_axes = plt.axes()
    # generate classwise plots
    for c in iou_classwise[0]:
        iou_axes.plot(range(1,6), c)
    
    plt.title("Classwise IoU")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(range(1,14))
    
    #mplcursors.cursor(hover=True)
    #mplcursors.cursor().connect(
    #"add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
    class_labels = range(1,14)
    interactive_legend = plugins.InteractiveLegendPlugin(iou_axes, class_labels)
    plugins.connect(iou_fig, interactive_legend)

    mpld3.display()
    #plt.show()
    mpld3.save_html(iou_fig, "iou.html", template_type='simple', no_extras=True)
    '''
    
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch+1)))


def train(train_loader, model, criterion, optimizer, scheduler, epoch, key, losses):
    '''
    Run one training epoch
    '''

    # Switch to train mode
    model.train()

    train_loop = tqdm(enumerate(train_loader), total=len(train_loader))

    total_train_loss = 0

    for i, (img, gt) in train_loop:

        # For TenCrop Data Augmentation
        img = img.view(-1, 3, args.resizedImageSize, args.resizedImageSize)
        img = utils.normalize(img, torch.Tensor([0.336, 0.213, 0.182]), torch.Tensor([0.278, 0.219, 0.185])) # (image batch, per-channel mean, per-channel standard deviation)
        gt = gt.view(-1, 3, args.resizedImageSize, args.resizedImageSize)

        # Process the network inputs and outputs
        gt_temp = gt * 255
        label = utils.generateLabel4CE(gt_temp, key)
        oneHotGT = utils.generateOneHot(gt_temp, key)

        img, label = Variable(img), Variable(label)

        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        # Compute output
        seg = model(img)
        # Printing out shapes for SegNet debugging
        #print("SEG SHAPE:", seg.shape) #torch.Size([16, 13, 224, 224])
        #print("LABEL SHAPE:", label.shape) #torch.Size([16, 224, 224])
        
        loss = model.dice_loss(seg, label)
        total_train_loss += loss.mean().item()

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #scheduler.step(loss.mean().item())
        scheduler.step(loss.mean().detach())
        #print('[%d/%d][%d/%d] Loss: %.4f'%(epoch, args.epochs-1, i, len(train_loader)-1, total_loss.mean().item()))
        train_loop.set_description(f"Epoch [{epoch + 1}/{args.epochs}]")
        train_loop.set_postfix(avg_loss = total_train_loss / (i + 1), epoch=epoch+1)
        #train_loop.set_postfix(loss = loss.mean().item(), epoch=epoch+1)
        
        utils.displaySamples(img, seg, gt, use_gpu, key, False, epoch, i, args.save_dir)
        
    
    losses.append(total_train_loss / len(train_loop))


def validate(val_loader, model, criterion, epoch, key, evaluator, losses):
    '''
    Run evaluation
    '''

    # Switch to evaluate mode
    model.eval()

    total_val_loss = 0

    val_loop = tqdm(enumerate(val_loader), total=len(val_loader))

    for i, (img, gt) in val_loop:

        # Process the network inputs and outputs
        img = utils.normalize(img, torch.Tensor([0.336, 0.213, 0.182]), torch.Tensor([0.278, 0.219, 0.185]))
        gt_temp = gt * 255
        label = utils.generateLabel4CE(gt_temp, key)
        oneHotGT = utils.generateOneHot(gt_temp, key)

        img, label = Variable(img), Variable(label)

        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        # Compute output
        seg = model(img)
        loss = model.dice_loss(seg, label)

        total_val_loss += loss.mean().item()

        #print('[%d/%d][%d/%d] Loss: %.4f'%(epoch, args.epochs-1, i, len(val_loader)-1, loss.mean().item()))
        val_loop.set_description(f"Epoch [{epoch + 1}/{args.epochs}]")
        val_loop.set_postfix(avg_loss = total_val_loss / (i + 1), epoch=epoch+1)
        #val_loop.set_postfix(loss = loss.mean().item(), epoch=epoch+1)

        utils.displaySamples(img, seg, gt, use_gpu, key, args.saveTest, epoch, i, args.save_dir)
        evaluator.addBatch(seg, oneHotGT)
        
    
    losses.append(total_val_loss / len(val_loop))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    '''
        Save the training model
    '''
    torch.save(state, filename)

if __name__ == '__main__':
    main()
