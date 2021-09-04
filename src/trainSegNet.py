'''
Semantic Segmentation on Surgical Images
'''

# torch imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
import segmentation_models_pytorch as smp

# general imports
import argparse
import os

# utility imports
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot
from tqdm import tqdm
import random
import utils
from utils import dice_loss

# model imports
from model.unet import UNet
from model.segnet import SegNet
from model.resnet_unet import ResNetUNet
from data.dataloaders.SegNetDataLoaderV2 import SegNetDataset # V2 is for loading entire dataset into memory, considering cholec data can easily fit into memory


parser = argparse.ArgumentParser(description='Semantic Segmentation Training Parameters')

# DATA PROCESSING
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)') # 4 * num_gpu
parser.add_argument('--data_dir', type=str, help='data directory with train, test, and trainval image folders')
parser.add_argument('--json_path', type=str, help='path to json file containing class information for segmentation')
parser.add_argument('--dataset', type=str, help='dataset title (options: synapse / cholec / miccaiSegOrgans / miccaiSegRefined)')

# MODEL PARAMETERS
parser.add_argument('--model', default='segnet', type=str, help='model architecture for segmentation (default: segnet)')
parser.add_argument('--batchnorm_momentum', default=0.1, type=float, help='batchnorm momentum for segnet')

# TRAINING PARAMETERS
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N', help='print frequency (default:1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--trainBatchSize', default=32, type=int, help='Training Mini-batch size (default: 32)')
parser.add_argument('--valBatchSize', default=27, type=int, help='ValidationMini-batch size (default: 27)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--optimizer', type=str, help='optimizer for the semantic segmentation network')
parser.add_argument('--wd', default=0.001, type=float, help='weight decay factor for optimizer')
parser.add_argument('--dice_loss_factor', default=0.5, type=float, help='loss weight factor for dice loss')
parser.add_argument('--lr_steps', default=2, type=int, help='number of steps to take with StepLR')
parser.add_argument('--step_gamma', default=0.1, type=float, help='gamma decay factor when stepping the Learning Rate')
parser.add_argument('--resnetModel', default=18, type=float, help='resnet model number')
parser.add_argument('--differential_lr', default=False, type=bool, help='use differential learning rate for pretrained encoder layers')

# IMAGE PARAMETERS
parser.add_argument('--resizedHeight', default=256, type=int, help='height of the input image to the network')
parser.add_argument('--resizedWidth', default=256, type=int, help='width of the resized image to the network')
parser.add_argument('--cropSize', default=256, type=int, help='height/width of the resized crop to the network')
parser.add_argument('--display_samples', default="False", type=str, help='Display samples during training / validation')
parser.add_argument('--save_samples', default="True", type=str, help='Save samples during final validation epoch')


# EVALUATION PARAMETERS
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

# SAVE PARAMETERS
parser.add_argument('--save_dir', dest='save_dir', help='The directory used to save the trained models', default='save_temp', type=str)
parser.add_argument('--seg_save_dir', dest='seg_save_dir', help='The directory used to save the segmentation results in the final epochs', type=str)
parser.add_argument('--saveSegs', default="True", type=str, help='Saves the validation/test images if True')

# GPU Check
use_gpu = torch.cuda.is_available()
curr_device = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(curr_device)
device = torch.device('cuda' if use_gpu else 'cpu')

print("CUDA AVAILABLE:", use_gpu, flush=True)
print("CURRENT DEVICE:", curr_device, torch.cuda.device(curr_device), flush=True)
print("DEVICE NAME:", device_name, flush=True)

# Reproducibility
seed = 6210 # [6210, 2021, 3005]
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    # setup and display args on debug.log
    global args
    args = parser.parse_args()
    print(f"args: {args}")

    # logger setup and display args on train.log
    log_path = os.path.join(args.save_dir, "train.log")
    logger = utils.get_logger("model", log_path)
    logger.info(f"args: {args}")

    image_size = [args.resizedHeight, args.resizedWidth]

    # Data Augmentations and Loading
    rotate, horizontal_flip, vertical_flip = True, True, True
    logger.info(f"Data Augmentations: rotate={rotate}, horizontal_flip={horizontal_flip}, vertical_flip={vertical_flip}")

    image_datasets = {x: SegNetDataset(os.path.join(args.data_dir, x), args.cropSize, args.json_path, x, args.dataset, 
                      image_size, rotate=rotate, horizontal_flip=horizontal_flip, vertical_flip=vertical_flip) for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=args.trainBatchSize if x == 'train' else args.valBatchSize,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  pin_memory=True,
                                                  drop_last=True)
                   for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    logger.info(f"# of Training Images: {dataset_sizes['train']}")
    logger.info(f"# of Validation Images: {dataset_sizes['test']}")

    # Get the dictionary for the id and RGB value pairs for the dataset
    classes = image_datasets['train'].classes
    print("\nCLASSES:", classes, "\n", flush=True)
    key = utils.disentangleKey(classes)
    print("KEY", key, "\n", flush=True)
    num_classes = len(key)
    print("NUM CLASSES:", num_classes, "\n", flush=True)
    
    # Model Initialization
    if args.model == 'segnet':
        model = SegNet(args.batchnorm_momentum, num_classes)
        model = model.to(device)
    elif args.model == 'unet':
        model = UNet(n_channels=3, n_classes=num_classes, bilinear=True)
        model = model.to(device)
    elif "resnet18" in args.model:
        model = ResNetUNet(n_class=num_classes, resnet_model=18)
        model = model.to(device)
    elif args.model == 'smp_UNet++':
        model = smp.UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        model = model.to(device)
    elif args.model == 'smp_unet18':
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        model = model.to(device)
    elif args.model == "smp_DeepLabV3+":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        model = model.to(device)
    elif args.model == "smp_MANet":
        model = smp.MAnet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        model = model.to(device)
    else:
        return "Model not available!"
    
    if args.cropSize != -1:
        print(summary(model, input_size=(3, args.cropSize, args.cropSize)), flush=True)
    else:
        print(summary(model, input_size=(3, args.resizedHeight, args.resizedWidth)), flush=True)

    # distributions of pixels per class
    # # of classes in each image

    # Computed using "../notebooks/Calculate Mean and Std of Dataset.ipynb"
    # NOTE: If you add any data to your training set, recompute the Mean and STD using the above jupyter notebook
    if args.dataset == "synapse":
        image_mean = [0.425, 0.304, 0.325] # mean [R, G, B]
        image_std = [0.239, 0.196, 0.202] # standard deviation [R, G, B]
    elif args.dataset == "cholec":
        image_mean = [0.337, 0.212, 0.182]
        image_std = [0.278, 0.218, 0.185]
    else:
        return "Dataset not available!"


    # Optionally resume training from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            pretrained_dict = checkpoint['state_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model.state_dict()}

            new_weights = {}
            
            print("Pretrained Weight Dict:\n")

            for n, p in pretrained_dict.items():
                print(n, p.shape)

            for n, p in pretrained_dict.items():
                if n.split(".")[0] != "conv_last": # set all parameters to pretrained weights except conv_last
                    new_weights[n] = p.data
            
            for n, p in model.named_parameters():
                if n.split(".")[0] == "conv_last": # conv_last trained from scratch
                    new_weights[n] = p.data
                        
            model.state_dict().update(new_weights)
            model.load_state_dict(new_weights, strict=False)
            logger.info(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # Freeze all weights except those in the output layer
    #for n, p in model.named_parameters():
    #    if n.split(".")[0] != "base_model":
    #        p.requires_grad = False
    
    # Optimization Setup
    if args.dataset == "synapse":
        criterion = nn.CrossEntropyLoss(ignore_index=21)
    else:
        criterion = nn.CrossEntropyLoss()

    if args.differential_lr == False:
        if args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.wd)
            logger.info(f"{args.optimizer} Optimizer LR = {args.lr} with WD = {args.wd}")
        elif args.optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.wd)
            logger.info(f"{args.optimizer} Optimizer LR = {args.lr} with WD = {args.wd}")
        elif args.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9)
            logger.info(f"{args.optimizer} Optimizer LR = {args.lr} with WD = {args.wd} and Momentum = 0.9")
    else:
        reduced_lr = args.lr * 0.1
        if args.optimizer == "Adam":
            optimizer = optim.Adam([
                        {'params': model.base_model.parameters(), 'lr': args.lr}], lr=reduced_lr, weight_decay=args.wd)
        elif args.optimizer == "SGD":
            optimizer = optim.SGD([{'params': model.base_model.parameters(), 'lr': args.lr}], lr=reduced_lr, weight_decay=args.wd, momentum=0.9)
        
        logger.info(f"{args.optimizer} Optimizer LR = {args.lr} for Base Model ({args.model}) Parameters and LR = {reduced_lr} for UNet Decoder Parameters with WD = {args.wd}")

    if args.lr_steps > 0:
        step_size = int(args.epochs // (args.lr_steps + 1))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args.step_gamma) # Step the Learning Rate by the gamma factor during training
        logger.info(f"StepLR initialized with step size = {step_size} and gamma = 0.1")
    else:
        raise ValueError("args.lr_steps must be > 0")
    
    if args.dice_loss_factor == -1:
        logger.info("Training with CE Loss Only")
    elif args.dice_loss_factor >= 0.0 and args.dice_loss_factor <= 1.0:
        logger.info(f"dice loss factor: {args.dice_loss_factor}")
    else:
        raise ValueError("args.dice_loss_factor must be a float value from 0.0 to 1.0")

    if use_gpu:
        criterion.cuda()

    # Initialize an evaluation Object
    evaluator = utils.Evaluate(key, use_gpu)

    train_losses = []
    val_losses = []

    total_iou = []
    total_precision = []
    total_recall = []
    total_f1 = []

    logger.info(f"Training Starting")
    best_f1 = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss = train(dataloaders['train'], model, criterion, optimizer, scheduler, epoch, key, train_losses, image_mean, image_std, logger, args)

        val_loss = validate(dataloaders['test'], model, criterion, epoch, key, evaluator, val_losses, image_mean, image_std, logger, args)

        scheduler.step()

        logger.info(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss}, Val Loss={val_loss}, LR={optimizer.param_groups[0]['lr']}")

        # Calculate the metrics
        print(f'\n>>>>>>>>>>>>>>>>>> Evaluation Metrics {epoch+1}/{args.epochs} <<<<<<<<<<<<<<<<<', flush=True)
        IoU = evaluator.getIoU()

        print(f"Mean IoU = {torch.mean(IoU)}", flush=True)
        print(f"Class-Wise IoU = {IoU}", flush=True)
        total_iou.append(torch.mean(IoU))

        PRF1 = evaluator.getPRF1()
        precision, recall, F1 = PRF1[0], PRF1[1], PRF1[2]

        print(f"Mean Precision = {torch.mean(precision)}", flush=True)
       #print(f"Class-Wise Precision = {precision}", flush=True)
        total_precision.append(torch.mean(precision))

        print(f"Mean Recall = {torch.mean(recall)}", flush=True)
        #print(f"Class-Wise Recall = {recall}", flush=True)
        total_recall.append(torch.mean(recall))

        print(f"Mean F1 = {torch.mean(F1)}", flush=True)
        print(f"Class-Wise F1 = {F1}", flush=True)
        total_f1.append(torch.mean(F1))

        if torch.mean(F1) > best_f1:
            best_f1 = torch.mean(F1)
            best_epoch = epoch + 1

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(args.save_dir, f"{args.model}_{args.dataset}_bs{args.trainBatchSize}lr{args.lr}e{args.epochs}_checkpoint"))
            
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Mean IoU={torch.mean(IoU)}, Mean Precision={torch.mean(precision)}, Mean Recall={torch.mean(recall)}, Mean F1={torch.mean(F1)} (Best) (Saved)")
        else:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Mean IoU={torch.mean(IoU)}, Mean Precision={torch.mean(precision)}, Mean Recall={torch.mean(recall)}, Mean F1={torch.mean(F1)}")

        evaluator.reset()

    logger.info(f"(Training Complete): Best Mean F1={best_f1}, Best Epoch={best_epoch}")

    # loss curves
    plt.plot(range(1, args.epochs+1), train_losses, color='blue')
    plt.plot(range(1, args.epochs+1), val_losses, color='black')
    plt.legend(["Train Loss", "Val Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves for {args.model} on {args.dataset} (bs{args.trainBatchSize}/lr{args.lr}/e{args.epochs})")
    figure_name = f"loss_{args.model}_{args.dataset}_bs{args.trainBatchSize}lr{args.lr}e{args.epochs}.png"
    plt.savefig(f"{args.save_dir}/{figure_name}")

    logger.info(f"Loss Curve saved to {args.save_dir}/{figure_name}")

    # mean accuracy curves
    plt.clf()
    plt.plot(range(1, args.epochs+1), total_iou, color='blue')
    plt.plot(range(1, args.epochs+1), total_precision, color='red')
    plt.plot(range(1, args.epochs+1), total_recall, color='magenta')
    plt.plot(range(1, args.epochs+1), total_f1, color='black')
    plt.legend(["Mean IoU", "Mean Precision", "Mean Recall", "Mean F1"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curves for {args.model} on {args.dataset} (bs{args.trainBatchSize}/lr{args.lr}/e{args.epochs})")
    figure_name = f"acc_{args.model}_{args.dataset}_bs{args.trainBatchSize}lr{args.lr}e{args.epochs}.png"
    plt.savefig(f"{args.save_dir}/{figure_name}")

    logger.info(f"Accuracy Curve saved to {args.save_dir}/{figure_name}")


def train(train_loader, model, criterion, optimizer, scheduler, epoch, key, losses, img_mean, img_std, logger, args):
    '''
    Run one training epoch
    '''

    # Switch to train mode
    model.train()

    train_loop = tqdm(enumerate(train_loader), total=len(train_loader))

    total_train_loss = 0

    for i, (img, gt, label) in train_loop:

        # For TenCrop Data Augmentation
        if args.cropSize != -1:
            img = img.view(-1, 3, args.cropSize, args.cropSize)
            gt = gt.view(-1, 3, args.cropSize, args.cropSize)
        else:
            img = img.view(-1, 3, args.resizedHeight, args.resizedWidth)
            gt = gt.view(-1, 3, args.resizedHeight, args.resizedWidth)
        
        img = utils.normalize(img, torch.Tensor(img_mean), torch.Tensor(img_std))

        # Process the network inputs and outputs

        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        # Compute output
        seg = model(img)

        if args.dataset == "synapse":
            if args.dice_loss_factor != -1:
                loss = (args.dice_loss_factor * dice_loss(seg, label, ignore_index=21)) +  ((1 - args.dice_loss_factor) * criterion(seg, label))
            else:
                loss = criterion(seg, label)
        else:
            loss = (0.5 * dice_loss(seg, label)) +  (0.5 * criterion(seg, label))
        total_train_loss += loss.mean().item()

        # PyTorch recommended over optimizer.zero_grad()
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
        for param in model.parameters():
            param.grad = None

        loss.backward()
        optimizer.step()

        train_loop.set_description(f"Epoch [{epoch + 1}/{args.epochs}]")
        train_loop.set_postfix(avg_loss = total_train_loss / (i + 1))

        seg = torch.argmax(seg, dim=1)
        
        if args.display_samples == "True":
            utils.displaySamples(img, seg, gt, use_gpu, key, False, epoch, i)
        
    losses.append(total_train_loss / len(train_loop))

    return total_train_loss/len(train_loop)

@torch.no_grad() # disables gradient calculations
def validate(val_loader, model, criterion, epoch, key, evaluator, losses, img_mean, img_std, logger, args):
    '''
    Run evaluation
    '''

    # Switch to evaluate mode
    model.eval()

    total_val_loss = 0

    val_loop = tqdm(enumerate(val_loader), total=len(val_loader))

    for i, (img, gt, label) in val_loop:
        # Process the network inputs and outputs
        img = utils.normalize(img, torch.Tensor(img_mean), torch.Tensor(img_std))
        oneHotGT = one_hot(label, len(key)).permute(0, 3, 1, 2)

        if use_gpu:
            img = img.cuda()
            label = label.cuda()
            oneHotGT = oneHotGT.cuda()

        # Compute output
        seg = model(img)

        if args.dataset == "synapse":
            if args.dice_loss_factor != -1:
                loss = (args.dice_loss_factor * dice_loss(seg, label, ignore_index=21)) +  ((1 - args.dice_loss_factor) * criterion(seg, label))
            else:
                loss = criterion(seg, label)
        else:
            loss = (0.5 * dice_loss(seg, label)) +  (0.5 * criterion(seg, label))

        total_val_loss += loss.mean().item()

        val_loop.set_description(f"Epoch [{epoch + 1}/{args.epochs}]")
        val_loop.set_postfix(avg_loss = total_val_loss / (i + 1))

        evaluator.addBatch(seg, oneHotGT, args)

        seg = torch.argmax(seg, dim=1)

        if args.display_samples == "True":
            utils.displaySamples(img, seg, gt, use_gpu, key, saveSegs=args.saveSegs, epoch=epoch, imageNum=i, save_dir=args.seg_save_dir, total_epochs=args.epochs)
        elif args.display_samples == "False" and args.save_samples == "True" and (epoch+1) == args.epochs:
            utils.displaySamples(img, seg, gt, use_gpu, key, saveSegs=args.saveSegs, epoch=epoch, imageNum=i, save_dir=args.seg_save_dir, total_epochs=args.epochs)

        
    losses.append(total_val_loss / len(val_loop))

    return total_val_loss/len(val_loop)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    '''
    Save the training model
    '''
    torch.save(state, filename)


if __name__ == '__main__':
    main()