import os
from matplotlib import pyplot, image
import numpy as np
import torch
from math import log10
import ssim
from PIL import Image
from torchvision.transforms import ToTensor

path = "C:\\CodingSSD\\FYP-General\\SID_Dataset\\Downscaled_JPEG\\All\\Combined\\test"

L1_total = 0
PSNR_total = 0
SSIM_total = 0
matching_total = 0

imgToTensor = ToTensor()
criterionL1 = torch.nn.L1Loss()
criterionMSE = torch.nn.MSELoss()
test_size = len(os.listdir(path))
for i, filename in enumerate(os.listdir(path)):
    if(i%100==0):
        print("%i/%i"%(i, test_size))
    fullPath = os.path.join(path, filename)
    img = Image.open(fullPath).convert('RGB')
    # #img = image.imread(fullPath)
    # height = len(img)
    # width = len(img[0])
    # middle = int(width/2)
    w, h = img.size
    w2 = int(w / 2)
    shortExpImg = img.crop((0, 0, w2, h))
    longExpImg = img.crop((w2, 0, w, h))
    shortExpImg = torch.unsqueeze(imgToTensor(shortExpImg), 0)
    longExpImg = torch.unsqueeze(imgToTensor(longExpImg), 0)

    L1_total += criterionL1(shortExpImg, longExpImg).item()
    PSNR_total += 10*log10(1/criterionMSE(shortExpImg, longExpImg).item())
    #SSIM_total += ssim.ssim(shortExpImg, longExpImg).item()

print("L1: %.4f, PSNR %.4f, SSIM: %.4f, matching score: %.4f" % (L1_total/test_size, PSNR_total/test_size, SSIM_total/test_size, matching_total/test_size))