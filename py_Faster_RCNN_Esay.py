## ebrahim parcham eval and train faster rcnn
## import Lib
import torch
import torchvision
import torch.nn as nn
import copy
import time
import torchvision.models as models
import torchsummary
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
print(torch.__version__)
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_properties('cuda'))

## select divice for cuda or cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## load model:
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
print(model)
model.to(device)

## load one image
img_pil = Image.open('test.jpg').convert('RGB')
# img_pil.show()

## eval image
## convert image to tensor
tensor = torchvision.transforms.functional.to_tensor(img_pil)
## create list image
List_img = [tensor.to(device)]
model.eval()
with torch.no_grad():
    predict = model(List_img)

# predict = predict.to(torch.device('cpu'))

boxes  = predict[0]['boxes']
labels = predict[0]['labels']
scores = predict[0]['scores']

boxes  = boxes.cpu()
labels = labels.cpu()
scores = scores.cpu()

print(predict)

## plat box
np_arr = tensor.permute(1,2,0).numpy()
plt.imshow(np_arr)
ax = plt.gca()

for box, label, score in zip(boxes, labels, scores):
  if score > 0.6:
    rect = Rectangle((box[0], box[1]),
                     (box[2] - box[0]),
                     (box[3] - box[1]),
                     fill=False,
                     edgecolor=(1, 0, 0),
                     linewidth=2)
    ax.add_patch(rect)

plt.show()

## Train
targets =[]
target = {}
target['boxes']  = boxes.to(device)
target['labels'] = labels.to(device)
targets.append(target)

##
model.train()  ## go to train mode
print(model(List_img,targets))
