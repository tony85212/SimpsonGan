import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
import os

transform = transforms.Compose([transforms.Resize(256),
                                #transforms.RandomHorizontalFlip(),
                                #transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                              ])
#helper function
def load_image(path, n = 1000):

    dataset = datasets.ImageFolder(path, transform=transform)
    if n >= len(dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers= 2)
    else:
        subset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), n, replace=False))
        dataloader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers= 2)

    return dataloader

def load_landmarks():

    l = []
    filename = "landmarks_simpson.txt"
    with open(filename) as f:
        content = f.readlines()
        for e in content:
            e = e.strip("\n")
            l.append(int(e))
    return l

def load_custom(path):

    image = Image.open(path)
    img = transform(image)
    img.unsqueeze_(0)
    return img

def to_np(t):

    return np.transpose(t[0].cpu().detach().numpy(), (1, 2, 0))

def save_image(image, path, text, epoch = 'original'):

    img = to_np(image)
    #Rescale to 0-255 and convert to uint8
    img = (((img - img.min()) * 255) / (img.max() - img.min())).astype(np.uint8)
    im = Image.fromarray(img)
    im.save(path + text + '_' + str(epoch) + ".png")

def make_directory(name):

    if not os.path.exists(os.path.dirname(name)):
        try:
            os.makedirs(os.path.dirname(name))
            print("Making " + name + " directory...")
        except:
            print("Can not gerate " + name + " directory...")
