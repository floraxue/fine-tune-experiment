
# imports
import argparse
import os
import torch
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

use_gpu = torch.cuda.is_available()

# load data
data_transform = transforms.Compose([
                 transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ])
data_dir = '../fine-tune-data/cats_vs_dogs_test1/'
# image_dataset = datasets.ImageFolder(data_dir, data_transform)
# dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4,
#                                          shuffle=True, num_workers=4)

images_dir = os.listdir(data_dir)
print(images_dir)

class_names = ['cat', 'dog']

# calculate results
def print_prediction_results(model):
    for img_dir in images_dir:
        img = Image.open(os.path.join(data_dir, img_dir))
        inputs = data_transform(img)
        inputs.unsqueeze_(0)
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            print(class_names[preds[j]] + ' for image name ' + img_dir)

# parse model name to load
parser = argparse.ArgumentParser()
parser.add_argument(
    '-n', '--name',
    help='model file name')
args = parser.parse_args()

# load model
if not args.name:
    print("Need model file name.")
    exit(0)
model_name = os.path.join("../fine-tune-data/", args.name)
model = torch.load(model_name)

# show results
print_prediction_results(model)
