'''
Utilites for data visualization and manipulation.
'''

import torch
import numpy as np
import cv2
import os
import logging
from torch.nn.functional import one_hot


########################### Evaluation Utilities ##############################

class Evaluate():
    '''
        Returns the mean IoU over the entire test set

        Code apapted from:
        https://github.com/Eromera/erfnet_pytorch/blob/master/eval/iouEval.py
    '''

    def __init__(self, key, use_gpu):
        self.num_classes = len(key)
        self.key = key
        self.use_gpu = use_gpu
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def addBatch(self, seg, gt, args):
        '''
            Add a batch of generated segmentation tensors and the respective
            groundtruth tensors.
            Dimensions should be:
            Seg: batch_size * num_classes * H * W
            GT: batch_size * num_classes * H * W
            GT should be one-hot encoded and Seg should be the softmax output.
            Seg would be converted to oneHot inside this method.
        '''

        # Convert Seg to one-hot encoding

        if args.dataset == "synapse":
            seg = seg[:,0:21:,:,]
            gt = gt[:,0:21:,:,]

        seg = torch.argmax(seg, dim=1)
        seg = one_hot(seg, self.num_classes - 1).permute(0, 3, 1, 2)
        
        #seg = convertToOneHot(seg, self.use_gpu).byte()
        seg = seg.float()
        gt = gt.float()

        if not self.use_gpu:
            seg = seg.cuda()
            gt = gt.cuda()

        tpmult = seg * gt    #times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fpmult = seg * (1-gt) #times prediction says its that class and gt says its not
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-seg) * (gt) #times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return iou    #returns "iou per class"

    def getPRF1(self):
        precision = self.tp / (self.tp + self.fp + 1e-15)
        recall = self.tp / (self.tp + self.fn + 1e-15)
        f1 = (2 * precision * recall) / (precision + recall + 1e-15)

        return precision, recall, f1

def convertToOneHot(batch, use_gpu):
    '''
        Converts the network output from softmax to one-hot encoding.
    '''

    if use_gpu:
        batch = batch.cpu()

    batch = batch.data.numpy()

    oneHot = []

    # Iterate over all images in a batch
    for i in range(len(batch)):
        vec = batch[i]
        idxs = vec

        single = np.zeros([1, batch.shape[1], batch.shape[2]])
        # Iterate over all the key-value pairs in the class Key dict
        for k in range(batch.shape[1]):
            mask = idxs == k
            mask = np.expand_dims(mask, axis=0)
            single = np.concatenate((single, mask), axis=0)

        single = np.expand_dims(single[1:,:,:], axis=0)
        
        oneHot.append(single)

    oneHot = np.concatenate(oneHot)
    oneHot = torch.from_numpy(oneHot.astype(np.uint8))
    return oneHot

############################# Regular Utilities ###############################
def get_logger(name, log_path=None):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    if log_path:
        handler = logging.FileHandler(log_path, 'w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


def displaySamples(img, generated, gt, use_gpu, key, saveSegs, epoch, imageNum, save_dir=None, total_epochs=None):
    ''' Display the original, generated, and the groundtruth image.
        If a batch is used, it displays only the first image in the batch.

        Args:
            input image, output image, groundtruth segmentation,
            use_gpu, class-wise key, save or not?, epoch, image number,
            save directory
    '''

    if use_gpu:
        img = img.cpu()
        generated = generated.cpu()

    gt = gt.numpy()
    gt = np.transpose(np.squeeze(gt[0,:,:,:]), (1,2,0)) # [256, 256, 3]
    gt = gt.astype(np.uint8)
    #print(f"gt shape: {gt.shape}, gt dtype: {gt.dtype}")
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB) / 255

    generated = generated.data.numpy()
    generated = reverseOneHot(generated, key)
    generated = np.squeeze(generated[0]).astype(np.uint8)
    generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB) / 255

    img = img.data.numpy()
    img = np.transpose(np.squeeze(img[0]), (1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    stacked = np.concatenate((img, generated, gt), axis = 1)

    if saveSegs == "True" and (epoch+1) == total_epochs:
        file_name = 'epoch_%d_img_%d.png' %(epoch, imageNum)
        save_path = os.path.join(save_dir, file_name)
        print(f"saving {save_path}")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        cv2.imwrite(save_path, stacked*255)

    cv2.namedWindow('Input | Gen | GT', cv2.WINDOW_NORMAL)
    cv2.imshow('Input | Gen | GT', stacked)

    cv2.waitKey(1)

def disentangleKey(key):
    '''
        Disentangles the key for class and labels obtained from the
        JSON file
        Returns a python dictionary of the form:
            {Class Id: RGB Color Code as numpy array}
    '''
    dKey = {}
    for i in range(len(key)):
        class_id = int(key[i]['id'])
        c = key[i]['color']
        c = c.split(',')
        c0 = int(c[0][1:])
        c1 = int(c[1])
        c2 = int(c[2][:-1])
        color_array = np.asarray([c0,c1,c2])
        dKey[class_id] = color_array

    return dKey

def generateLabel4CE(gt, key):
    '''
        Generates the label for Cross Entropy Loss from a batch of groundtruth
        segmentation images.

        Given the ground truth mask for the surgical image, we perform two iterations:

        The outer iteration iterates over all the images in the provided GT batch.

        The inner iteration iterates over the images pixel classes and assigns the pixels
        to their numbered classes (e.g. 0 - 12 are the numbered classes, and the RGB pixels
        are re-assigned to fit the 0 - 12 format for one-hot encoding)
    '''

    batch = gt.numpy()
    # Iterate over all images in a batch
    label = []
    for i in range(len(batch)):
        img = batch[i,:,:,:]
        img = np.transpose(img, (1,2,0))
        catMask = np.zeros((img.shape[0], img.shape[1]))

        # Iterate over all the key-value pairs in the class Key dict
        for k in range(len(key)):
            rgb = key[k]
            mask = np.where(np.all(img == rgb, axis = 2))
            catMask[mask] = k
        
        catMaskTensor = torch.from_numpy(catMask).unsqueeze(0)

        label.append(catMaskTensor)

    label = torch.cat(label, 0)
    return label.long()

def reverseOneHot(batch, key):
    '''
        Generates the segmented image from the output of a segmentation network.
        Takes a batch of numpy oneHot encoded tensors and returns a batch of
        numpy images in RGB (not BGR).
    '''

    generated = []

    # Iterate over all images in a batch
    for i in range(len(batch)):
        vec = batch[i]
        idxs = vec

        segSingle = np.zeros([idxs.shape[0], idxs.shape[1], 3])

        # Iterate over all the key-value pairs in the class Key dict
        for k in range(len(key)):
            rgb = key[k]
            mask = idxs == k
            segSingle[mask] = rgb

        segMask = np.expand_dims(segSingle, axis=0)
        
        generated.append(segMask)
    
    generated = np.concatenate(generated)

    return generated

def generateOneHot(gt, key):
    '''
        Generates the one-hot encoded tensor for a batch of images based on
        their class.
    '''

    batch = gt.numpy()
    # Iterate over all images in a batch
    for i in range(len(batch)):
        img = batch[i,:,:,:]
        img = np.transpose(img, (1,2,0))
        catMask = np.zeros((img.shape[0], img.shape[1]))

        # Iterate over all the key-value pairs in the class Key dict
        for k in range(len(key)):
            catMask = catMask * 0
            rgb = key[k]
            mask = np.where(np.all(img == rgb, axis = -1))
            catMask[mask] = 1

            catMaskTensor = torch.from_numpy(catMask).unsqueeze(0)
            if 'oneHot' in locals():
                oneHot = torch.cat((oneHot, catMaskTensor), 0)
            else:
                oneHot = catMaskTensor

    label = oneHot.view(len(batch),len(key),img.shape[0],img.shape[1])
    return label

def normalize(batch, mean, std):
    '''
        Normalizes a batch of images, provided the per-channel mean and
        standard deviation.
    '''

    mean.unsqueeze_(1).unsqueeze_(1)
    std.unsqueeze_(1).unsqueeze_(1)
    for i in range(len(batch)):
        img = batch[i,:,:,:]
        img = img.sub(mean).div(std).unsqueeze(0)

        if 'concat' in locals():
            concat = torch.cat((concat, img), 0)
        else:
            concat = img

    return concat

'''
Loss Functions
'''

def dice_loss(output, target, weights=None, ignore_index=None):
        '''
            output : NxCxHxW Variable
            target :  NxHxW LongTensor
            weights : C FloatTensor
            ignore_index : int index to ignore from loss
        '''
        eps = 0.0001

        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)