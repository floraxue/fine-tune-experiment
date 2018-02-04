
# imports
import argparse
import os
import time
import csv
import torch
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

use_gpu = torch.cuda.is_available()

# parse model name to load
parser = argparse.ArgumentParser()
parser.add_argument(
    '-n', '--name',
    help='model file name')
parser.add_argument(
    '-d', '--data',
    help='data directory')
parser.add_argument(
    '--custom',
    help='test image file name as in mytest folder')
args = parser.parse_args()

if not args.name:
    print("Need model file name.")
    exit(0)
if not args.data:
    data_dir = '../fine-tune-data/cats_vs_dogs_test1/'
else:
    data_dir = os.path.join('../fine-tune-data/', args.data)
use_custom_fname = False
if args.custom:
    use_custom_fname = True
model_name = os.path.join("../fine-tune-data/", args.name)

# load data
data_transform = transforms.Compose([
                 transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ])
# image_dataset = datasets.ImageFolder(data_dir, data_transform)
# dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4,
#                                          shuffle=True, num_workers=4)

images_dir = os.listdir(data_dir)
print(images_dir)

class_names = ['cat', 'dog']

# load model
model = torch.load(model_name)

### calculating results
# get image clean number
def get_clean_name(img_dir):
    return int(img_dir.split('.')[0])

def get_clean_name_for_mytest(img_dir):
    '''
    Ideally, the test results would be: first 0-999 = 0, 1000-1999 = 1
    '''
    tokens = img_dir.split('.')
    num = int(tokens[1])
    num -= 9000  # eliminate the 9xxx
    num += 1     # start index at 1
    if tokens[0] == "dog":
        num += 1000  # "dog" starts from 1001
    return num

def write_to_csv(result):
    fname = '../fine-tune-data/cats_vs_dogs_submission_' + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + '.csv'
    file = open(fname, 'w', newline='')
    with file:
        fields = ['id', 'label']
        writer = csv.DictWriter(file, fieldnames=fields)    
        writer.writeheader()
        for i in range(1, len(images_dir) + 1):
            writer.writerow({'id' : i, 'label': result[i]})
    file.close()

# run on model and print results
def print_prediction_results(model):
    result = {}
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
            # print(class_names[preds[j]] + ' for image name ' + img_dir)
            if use_custom_fname:
                img_no = get_clean_name_for_mytest(img_dir)
            else:
                img_no = get_clean_name(img_dir)
            print(img_no)
            result[img_no] = preds[j]
    write_to_csv(result)
    # print(result)

# show results
print_prediction_results(model)
