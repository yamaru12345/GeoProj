import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import skimage
import skimage.io as io
from torchvision import transforms
import numpy as np
import scipy.io as scio
from collections import OrderedDict

from modelNetM import EncoderNet, DecoderNet, ClassNet, EPELoss

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model_en = EncoderNet([1,1,1,1,2])
model_de = DecoderNet([1,1,1,1,2])
model_class = ClassNet()

#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    model_en = nn.DataParallel(model_en)
#    model_de = nn.DataParallel(model_de)
#    model_class = nn.DataParallel(model_class)

if torch.cuda.is_available():
    model_en = model_en.cuda()
    model_de = model_de.cuda()
    model_class = model_class.cuda()

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

model_en.load_state_dict(fix_model_state_dict(torch.load('model_en.pkl')))
model_de.load_state_dict(fix_model_state_dict(torch.load('model_de.pkl')))
model_class.load_state_dict(fix_model_state_dict(torch.load('model_class.pkl')))

model_en.eval()
model_de.eval()
model_class.eval()  

testImgPath = '/content/images'
saveFlowPath = '/content/flow_cla'

correct = 0
for img in os.listdir(testImgPath):
    imgPath = os.path.join(testImgPath, img)
    disimgs = io.imread(imgPath)
    disimgs = transform(disimgs)

    use_GPU = torch.cuda.is_available()
    if use_GPU:
        disimgs = disimgs.cuda()

    disimgs = disimgs.view(1,3,256,256)
    disimgs = Variable(disimgs)

    middle = model_en(disimgs)
    flow_output = model_de(middle)
    clas = model_class(middle)

    _, predicted = torch.max(clas.data, 1)
    if predicted.cpu().numpy()[0] == index:
        correct += 1

    u = flow_output.data.cpu().numpy()[0][0]
    v = flow_output.data.cpu().numpy()[0][1]

    idx = split(img, jpg)[0]
    saveMatPath =  os.path.join(saveFlowPath, f'mat_{idx}.mat')
    scio.savemat(saveMatPath, {'u': u,'v': v}) 

