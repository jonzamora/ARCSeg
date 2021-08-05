'''
Semantic Segmentation on Surgical Images
'''

# torch imports
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable

# general imports
import argparse
import os

# utility imports
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.transforms import FiveCrop
from tqdm import tqdm
import random
import utils

# model imports
from model.unet import UNet
from model.segnet import SegNet
from data.dataloaders.SegNetDataLoader import SegNetDataset


parser = argparse.ArgumentParser(description='SegNet Training Parameters')

# DATA PROCESSING
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)') # 4 * num_gpu
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
parser.add_argument('--batchSize', default=2, type=int, help='Mini-batch size (default: 2)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--wd', '--weight_decay', default=0.0001, type=float, help='weight decay factor for optimizer')

# IMAGE PARAMETERS
parser.add_argument('--resizedHeight', default=256, type=int, help='height of the input image to the network')
parser.add_argument('--resizedWidth', default=256, type=int, help='width of the resized image to the network')

# EVALUATION PARAMETERS
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

# SAVE PARAMETERS
parser.add_argument('--save_dir', dest='save_dir', help='The directory used to save the trained models', default='save_temp', type=str)
parser.add_argument('--saveTest', default='False', type=str, help='Saves the validation/test images if True')


best_prec1 = np.inf

# GPU Check
use_gpu = torch.cuda.is_available()
curr_device = torch.cuda.current_device()
device_name = torch.cuda.get_device_name(curr_device)

print("CUDA AVAILABLE:", use_gpu, flush=True)
print("CURRENT DEVICE:", curr_device, torch.cuda.device(curr_device), flush=True)
print("DEVICE NAME:", device_name, flush=True)

# Reproducibility
seed = 6210
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    # Determine whether to save Input | Gen | GT images or not
    if args.saveTest == 'True':
        args.saveTest = True
    elif args.saveTest == 'False':
        args.saveTest = False


    log_path = os.path.join(args.save_dir, "train.log")
    logger = utils.get_logger("model", log_path)
    logger.info(f"args: {args}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.resizedHeight, args.resizedWidth), interpolation=Image.NEAREST),
            #transforms.FiveCrop((int(args.resizedHeight // 4), int(args.resizedWidth // 4))),
            #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize((args.resizedHeight, args.resizedWidth), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ]),
    }

    # Data Loading
    image_datasets = {x: SegNetDataset(os.path.join(args.data_dir, x), data_transforms[x], args.json_path, x, args.dataset) for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=args.batchSize,
                                                  shuffle=True,
                                                  num_workers=args.workers,
                                                  pin_memory=True)
                  for x in ['train', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    # Get the dictionary for the id and RGB value pairs for the dataset
    classes = image_datasets['train'].classes
    print("\nCLASSES:", classes, "\n", flush=True)
    key = utils.disentangleKey(classes)
    print("KEY", key, "\n", flush=True)
    num_classes = len(key)
    print("NUM CLASSES:", num_classes, "\n", flush=True)
    
    # Initialize the model
    if args.model == 'segnet':
        model = SegNet(args.batchnorm_momentum, num_classes)
    elif args.model == 'unet':
        model = UNet(n_channels=3, n_classes=num_classes, bilinear=False)
    else:
        return "Model not available!"

    # Computed using "../notebooks/Calculate Mean and Std of Dataset.ipynb"
    if args.dataset == "synapse":
        image_mean = [0.423, 0.303, 0.325] # mean [R, G, B]
        image_std = [0.235, 0.190, 0.196] # standard deviation [R, G, B]
    elif args.dataset == "cholec":
        image_mean = [0.337, 0.212, 0.182]
        image_std = [0.278, 0.218, 0.185]
    else:
        return "Dataset not available!"


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

    # Load the saved model
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume), flush=True)

    print(model, flush=True)
    
    # Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)    

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

    logger.info(f"Training Starting")
    for epoch in range(args.epochs):
        train_loss = train(dataloaders['train'], model, criterion, optimizer, scheduler, epoch, key, train_losses, image_mean, image_std, logger)

        val_loss = validate(dataloaders['test'], model, criterion, epoch, key, evaluator, val_losses, image_mean, image_std, logger)

        logger.info(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss}, Val Loss={val_loss}")

        # Calculate the metrics
        print(f'\n>>>>>>>>>>>>>>>>>> Evaluation Metrics {epoch+1}/{args.epochs} <<<<<<<<<<<<<<<<<', flush=True)
        IoU = evaluator.getIoU()

        print(f"Mean IoU = {torch.mean(IoU)}", flush=True)
        #print(f"Class-Wise IoU = {IoU}", flush=True)
        total_iou.append(torch.mean(IoU))
        #iou_classwise.append(IoU.cpu().detach().numpy())

        PRF1 = evaluator.getPRF1()
        precision, recall, F1 = PRF1[0], PRF1[1], PRF1[2]

        print(f"Mean Precision = {torch.mean(precision)}", flush=True)
        #print(f"Class-Wise Precision = {precision}", flush=True)
        total_precision.append(torch.mean(precision))
        #precision_classwise.append(precision.cpu().detach().numpy())

        print(f"Mean Recall = {torch.mean(recall)}", flush=True)
        #print(f"Class-Wise Recall = {recall}", flush=True)
        total_recall.append(torch.mean(recall))
        #recall_classwise.append(recall.cpu().detach().numpy())

        print(f"Mean F1 = {torch.mean(F1)}", flush=True)
        #print(f"Class-Wise F1 = {F1}", flush=True)
        total_f1.append(torch.mean(F1))
        #f1_classwise.append(F1.cpu().detach().numpy())

        logger.info(f"Epoch {epoch+1}/{args.epochs}: Mean IoU={torch.mean(IoU)}, Mean Precision={torch.mean(precision)}, Mean Recall={torch.mean(recall)}, Mean F1={torch.mean(F1)}")

        evaluator.reset()

    logger.info("Training Complete")

    # loss curves
    plt.plot(range(1, args.epochs+1), train_losses, color='blue')
    plt.plot(range(1, args.epochs+1), val_losses, color='black')
    plt.legend(["Train Loss", "Val Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves for {args.model} on {args.dataset} (bs{args.batchSize}/lr{args.lr}/e{args.epochs}")
    figure_name = f"loss_{args.model}_{args.dataset}_bs{args.batchSize}lr{args.lr}e{args.epochs}"
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
    plt.title(f"Accuracy Curves for {args.model} on {args.dataset} (bs{args.batchSize}/lr{args.lr}/e{args.epochs}")
    figure_name = f"acc_{args.model}_{args.dataset}_bs{args.batchSize}lr{args.lr}e{args.epochs}"
    plt.savefig(f"{args.save_dir}/{figure_name}")

    logger.info(f"Accuracy Curve saved to {args.save_dir}/{figure_name}")


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
    }, filename=os.path.join(args.save_dir, f"{args.model}_{args.dataset}_bs{args.batchSize}lr{args.lr}e{args.epochs}_checkpoint"))


def train(train_loader, model, criterion, optimizer, scheduler, epoch, key, losses, img_mean, img_std, logger):
    '''
    Run one training epoch
    '''

    # Switch to train mode
    model.train()

    train_loop = tqdm(enumerate(train_loader), total=len(train_loader))

    total_train_loss = 0

    for i, (img, gt) in train_loop:

        # For TenCrop Data Augmentation
        img = img.view(-1, 3, args.resizedHeight, args.resizedWidth)
        img = utils.normalize(img, torch.Tensor(img_mean), torch.Tensor(img_std)) # Synapse
        gt = gt.view(-1, 3, args.resizedHeight, args.resizedWidth)

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
        total_train_loss += loss.mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(loss.mean().detach())
        train_loop.set_description(f"Epoch [{epoch + 1}/{args.epochs}]")
        train_loop.set_postfix(avg_loss = total_train_loss / (i + 1))
        
        utils.displaySamples(img, seg, gt, use_gpu, key, False, epoch, i, args.save_dir)
        
    losses.append(total_train_loss / len(train_loop))
    return total_train_loss/len(train_loop)


def validate(val_loader, model, criterion, epoch, key, evaluator, losses, img_mean, img_std, logger):
    '''
    Run evaluation
    '''

    # Switch to evaluate mode
    model.eval()

    total_val_loss = 0

    val_loop = tqdm(enumerate(val_loader), total=len(val_loader))

    for i, (img, gt) in val_loop:

        # Process the network inputs and outputs
        img = utils.normalize(img, torch.Tensor(img_mean), torch.Tensor(img_std))
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

        val_loop.set_description(f"Epoch [{epoch + 1}/{args.epochs}]")
        val_loop.set_postfix(avg_loss = total_val_loss / (i + 1))

        utils.displaySamples(img, seg, gt, use_gpu, key, args.saveTest, epoch, i, args.save_dir)
        evaluator.addBatch(seg, oneHotGT)
        
    losses.append(total_val_loss / len(val_loop))
    return total_val_loss/len(val_loop)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    '''
    Save the training model
    '''
    torch.save(state, filename)


if __name__ == '__main__':
    main()