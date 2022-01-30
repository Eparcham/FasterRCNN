## ebrahim parcham eval and train faster rcnn
## import Lib
import os
import torch
import torchvision
import torch.nn as nn
import copy
import time
import torchvision.models as models
import torchsummary
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
print(torch.__version__)
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_properties('cuda'))

class costum_dataset(torch.utils.data.Dataset):
    def __init__(self,phase):
        self.dir_data = "./dataset"
        self.phase = phase
        self.list_img = os.listdir(os.path.join(self.dir_data, 'images'))
        self.targets = pd.read_csv(os.path.join(self.dir_data, 'data/{}_labels.csv'.format(self.phase)))

    def __len__(self):
        return len(self.dir_data)
    def __getitem__(self, item):
        image_path = os.path.join(self.dir_data, 'images', self.list_img[item])
        img = Image.open(image_path).convert("RGB")
        img = F.to_tensor(img)

        box_list = self.targets[self.targets['filename'] == self.list_img[item]]
        box_list = box_list[['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = torch.tensor(box_list, dtype=torch.float32)

        labels = torch.ones((len(box_list),),dtype=torch.int64)

        target = {}
        target['boxes']  = boxes
        target['labels'] = labels

        return img, target


## select divice for cuda or cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dir_data = "./dataset"
phase = "train"

list_img = os.listdir(os.path.join(dir_data,'images'))
# print(list_img)
## read target
targets = pd.read_csv(os.path.join(dir_data,'data/{}_labels.csv'.format(phase)))

## read one image
idx = 15
image_path = os.path.join(dir_data,'images',list_img[idx])
img = Image.open(image_path).convert("RGB")
plt.imshow(img)
plt.show()

box_list = targets[targets['filename']==list_img[idx]]
print(box_list)
print(box_list[['xmin' , 'ymin','xmax','ymax']])

##
train_dataset = costum_dataset("train")
test_dataset  = costum_dataset("test")

## give one image
one_sample = train_dataset[5]

## har bach ra ba asse tuple baradesh
def new_concat(batch):
  return tuple(zip(*batch))

train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=2,
                            shuffle=True,
                            collate_fn=new_concat)  ## use ouer model to read image

test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=new_concat)

one_use_train_loader_img, one_use_train_loader_targ = next(iter(train_loader))

## model :
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(1024, 2)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

## train

import math
def train_one_epoch(model, optimizer, train_dataloader):
    model.train()
    total_loss = 0
    for images, targets in train_dataloader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    return total_loss/len(train_dataloader)


num_epochs = 5

for epoch in range(num_epochs):
    loss = train_one_epoch(model, optimizer, train_loader)
    print('epoch [{}]:  \t lr: {}  \t loss: {}  '.format(epoch, lr_scheduler.get_last_lr(), loss))
    lr_scheduler.step()


## eval for test

import matplotlib.patches as patches

def evaluate(model, test_dataloader):
    model.eval()
    with torch.no_grad():
        cnt = 0
        for images , targets in test_dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            out = model(images)
            scores = out[0]['scores'].cpu().numpy()
            inds = scores > 0.7
            bxs = out[0]['boxes'].cpu().numpy()
            bxs = bxs[inds]
            gt = targets[0]['boxes'].cpu().numpy()
            # gt = gt[0]
            img = images[0].permute(1, 2, 0).cpu().numpy()
            #----------------------------------------------------------
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for j in range(len(gt)):
                rect1 = patches.Rectangle((int(gt[j][0]),int(gt[j][1])),abs(gt[j][0]-gt[j][2]),
                                abs(gt[j][1]-gt[j][3]),linewidth=3,edgecolor='g',facecolor='none')
                ax.add_patch(rect1)
            for i in range(len(bxs)):
                rect = patches.Rectangle((int(bxs[i][0]),int(bxs[i][1])),abs(bxs[i][0]-bxs[i][2]),
                                         abs(bxs[i][1]-bxs[i][3]),linewidth=3,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            fig.savefig("./output_images/{}.png".format(cnt), dpi=90, bbox_inches='tight')
            cnt = cnt + 1

## test
evaluate(model, test_loader)

## change backbone network

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280

anchor_gen = torchvision.models.detection.rpn.AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                                 aspect_ratios=((0.5, 1.0, 2.0), ))

# torchvision.models.detection.roi_heads.

light_model = torchvision.models.detection.FasterRCNN(backbone=backbone, num_classes=2,
                                                      rpn_anchor_generator=anchor_gen, )
light_model.train()
light_model.to(device)
num_epochs = 50

for epoch in range(num_epochs):
    loss = train_one_epoch(light_model, optimizer, train_loader)
    print('epoch [{}]:  \t lr: {}  \t loss: {}  '.format(epoch, lr_scheduler.get_last_lr(), loss))
    lr_scheduler.step()