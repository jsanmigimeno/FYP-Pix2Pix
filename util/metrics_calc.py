import os
from matplotlib import pyplot, image
import numpy as np
import torch
from math import log10
import ssim
from PIL import Image
from torchvision.transforms import ToTensor
import matching_tools.utils as matching_utils
from PIL import Image
import torchvision.transforms as transforms

def get_Matching(A, B):
    # TODO: Use opt to set path to checkpoint
    checkpoint_path = './util/matching_tools/HardNet++.pth'
    # Convert to Grayscale
    real_B = matching_utils.rgb2gray(np.transpose(A.cpu().numpy()[0], (1, 2, 0)))
    fake_B = matching_utils.rgb2gray(np.transpose(B.cpu().numpy()[0], (1, 2, 0)))
    indexes = matching_utils.get_keypoints_coordinates(real_B)
    desc_real_B = matching_utils.compute_desc(real_B, indexes, checkpoint_path=checkpoint_path)
    desc_fake_B = matching_utils.compute_desc(fake_B, indexes, checkpoint_path=checkpoint_path)

    # match descriptors
    matches = matching_utils.match(desc_real_B, desc_fake_B)
    matches_np = matching_utils.convert_opencv_matches_to_numpy(matches)
    true_matches = np.where(matches_np[:, 0] == matches_np[:, 1], 1., 0.)
    matching_score = np.sum(true_matches) / len(true_matches)
    return matching_score

path = "C:\\CodingSSD\\FYP-General\\SID_Dataset\\Downscaled_JPEG\\All\\Combined\\test"

load_size = 256
resizeTr = transforms.Resize([load_size, load_size], Image.BICUBIC)

L1_total = 0
PSNR_total = 0
SSIM_total = 0
matching_total = 0

imgToTensor = ToTensor()
criterionL1 = torch.nn.L1Loss()
criterionMSE = torch.nn.MSELoss()
test_size = len(os.listdir(path))
for i, filename in enumerate(os.listdir(path)):
    if(i%1==0):
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
    shortExpImg = resizeTr(shortExpImg)
    longExpImg = resizeTr(longExpImg)

    shortExpImg = torch.unsqueeze(imgToTensor(shortExpImg), 0)
    longExpImg = torch.unsqueeze(imgToTensor(longExpImg), 0)

    L1_total += criterionL1(shortExpImg, longExpImg).item()
    PSNR_total += 10*log10(1/criterionMSE(shortExpImg, longExpImg).item())
    SSIM_total += ssim.ssim(shortExpImg, longExpImg).item()
    matching_total += get_Matching(shortExpImg, longExpImg)

print("L1: %.4f, PSNR %.4f, SSIM: %.4f, matching score: %.4f" % (L1_total/test_size, PSNR_total/test_size, SSIM_total/test_size, matching_total/test_size))