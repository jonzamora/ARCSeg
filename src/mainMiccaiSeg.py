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
from datasets.miccaiSegDataLoader import miccaiSegDataset

parser = argparse.ArgumentParser(description='PyTorch SegNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
            help='number of data loading workers (default: 4)') # 4 * num_gpu
parser.add_argument('--epochs', default=20, type=int, metavar='N',
            help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
parser.add_argument('--batchSize', default=2, type=int,
            help='Mini-batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
            metavar='LR', help='initial learning rate')
parser.add_argument('--wd', '--weight_dacay', default=0.0005, type=float,
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
use_gpu = torch.cuda.is_available()

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
            transforms.Resize((args.imageSize, args.imageSize), interpolation=Image.NEAREST),
            transforms.TenCrop(args.resizedImageSize),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            #transforms.Lambda(lambda normalized: torch.stack([transforms.Normalize([0.295, 0.204, 0.197], [0.221, 0.188, 0.182])(crop) for crop in normalized]))
            #transforms.RandomResizedCrop(224, interpolation=Image.NEAREST),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((args.imageSize, args.imageSize), interpolation=Image.NEAREST),
            transforms.ToTensor(),
            #transforms.Normalize([0.295, 0.204, 0.197], [0.221, 0.188, 0.182])
        ]),
    }

    # Data Loading
    data_dir = '/home/jonzamora/Desktop/segNets/datasets/miccaiSegOrgans'
    # json path for class definitions
    json_path = '/home/jonzamora/Desktop/segNets/datasets/miccaiSegOrganClasses.json'

    image_datasets = {x: miccaiSegDataset(os.path.join(data_dir, x), data_transforms[x],
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
    key = utils.disentangleKey(classes)
    num_classes = len(key)

    # Initialize the model
    model = UNet(num_classes)

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
    criterion = nn.CrossEntropyLoss()

    # Use a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if use_gpu:
        model.cuda()
        criterion.cuda()

    # Initialize an evaluation Object
    evaluator = utils.Evaluate(key, use_gpu)

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)
        print('>>>>>>>>>>>>>>>>>>>>>>>Training<<<<<<<<<<<<<<<<<<<<<<<')
        # One Epoch
        train(dataloaders['train'], model, criterion, optimizer, scheduler, epoch, key)

        # COMMENT OUT FOR TRAINING LOOP
        # Evaulate on validation set

        print('>>>>>>>>>>>>>>>>>>>>>>>Testing<<<<<<<<<<<<<<<<<<<<<<<')
        validate(dataloaders['test'], model, criterion, epoch, key, evaluator)


        # Calculate the metrics
        print('>>>>>>>>>>>>>>>>>> Evaluating the Metrics <<<<<<<<<<<<<<<<<')
        IoU = evaluator.getIoU()
        print('Mean IoU: {}, Class-wise IoU: {}'.format(torch.mean(IoU), IoU))
        PRF1 = evaluator.getPRF1()
        precision, recall, F1 = PRF1[0], PRF1[1], PRF1[2]
        print('Mean Precision: {}, Class-wise Precision: {}'.format(torch.mean(precision), precision))
        print('Mean Recall: {}, Class-wise Recall: {}'.format(torch.mean(recall), recall))
        print('Mean F1: {}, Class-wise F1: {}'.format(torch.mean(F1), F1))
        evaluator.reset()
    '''
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))
    '''

def train(train_loader, model, criterion, optimizer, scheduler, epoch, key):
    '''
        Run one training epoch
    '''

    # Switch to train mode
    model.train()
    
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

    for i, (img, gt) in train_loop:

        # For TenCrop Data Augmentation
        img = img.view(-1,3,args.resizedImageSize,args.resizedImageSize)
        img = utils.normalize(img, torch.Tensor([0.295, 0.204, 0.197]), torch.Tensor([0.221, 0.188, 0.182]))
        gt = gt.view(-1,3,args.resizedImageSize,args.resizedImageSize)

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
        loss = model.dice_loss(seg, label)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        train_loop.set_description(f"Epoch [{epoch + 1}/{args.epochs}] [{i}/{len(train_loader)-1}]")
        train_loop.set_postfix(loss = loss.mean().detach())

        #scheduler.step(loss.mean().item())
        scheduler.step(loss.mean().detach())
        print('[%d/%d][%d/%d] Loss: %.4f'%(epoch, args.epochs-1, i, len(train_loader)-1, loss.mean().item()))

        
        utils.displaySamples(img, seg, gt, use_gpu, key, False, epoch, i, args.save_dir)

def validate(val_loader, model, criterion, epoch, key, evaluator):
    '''
        Run evaluation
    '''

    # Switch to evaluate mode
    model.eval()

    for i, (img, gt) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):

        # Process the network inputs and outputs
        img = utils.normalize(img, torch.Tensor([0.295, 0.204, 0.197]), torch.Tensor([0.221, 0.188, 0.182]))
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

        print('[%d/%d][%d/%d] Loss: %.4f'%(epoch, args.epochs-1, i, len(val_loader)-1, loss.mean().detach()))

        utils.displaySamples(img, seg, gt, use_gpu, key, args.saveTest, epoch, i, args.save_dir)
        evaluator.addBatch(seg, oneHotGT)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    '''
        Save the training model
    '''
    torch.save(state, filename)

if __name__ == '__main__':
    main()
