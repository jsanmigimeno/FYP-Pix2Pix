import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )

        for param in self.features.parameters():
            param.requires_grad = False

    def input_norm(self, x, eps=1e-6):
        x_flatten = x.view(x.shape[0], -1)
        x_mu = torch.mean(x_flatten, dim=-1, keepdim=True)
        x_std = torch.std(x_flatten, dim=-1, keepdim=True)
        x_norm = (x_flatten - x_mu) / (x_std + eps)
        return x_norm.view(x.shape)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return F.normalize(x, dim=-1, p=2)

class HardNetDescriptor(object):
    def __init__(self, checkpoint_path, patch_size=32, gpu_ids=[]):
        self.patch_size = patch_size

        self._model = HardNet()
        self.gpu_ids = gpu_ids

        # CPU/GPU
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self._model.to(gpu_ids[0])
            self._model = torch.nn.DataParallel(self._model, gpu_ids)  # multi-GPUs
        print(self._model.device)
        self._load_model(checkpoint_path)  # load pretrained model

    def _load_model(self, checkpoint_path):
        assert os.path.isfile(checkpoint_path), \
            "Invalid file: {}".format(checkpoint_path)
        
        # CPU/GPU
        if len(self.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            checkpoint = torch.load(checkpoint_path,
                map_location=torch.device('cuda'))
        else:
            checkpoint = torch.load(checkpoint_path,
                map_location=torch.device('cpu'))
        self._model.load_state_dict(checkpoint['state_dict'], strict=True)

    def compute(self, image, kpts, mask=None):
        """Compute the descriptors from given features.
           We assume image (H, W) numpy array in np.unit8
        """
        # extract patches and convert to torch tensor
        #patches = extract_patches_from_opencv_keypoints(image, kpts)
        patches = extract_patches_from_coords(image, kpts)

        #patches = torch.unsqueeze(torch.from_numpy(patches), dim=1)
        patches = torch.unsqueeze(patches, dim=1)
        patches = patches.float() / 255.

        # inference model with patches
        descs = self._model(patches)
        return kpts, descs


def match(descs1, descs2):
     """Compute brute force matches between two sets of descriptors.
     """
     assert isinstance(descs1, np.ndarray), type(descs1)
     assert isinstance(descs2, np.ndarray), type(descs2)
     assert len(descs1.shape) == 2, descs1.shape
     assert len(descs2.shape) == 2, descs2.shape
     matcher = cv2.BFMatcher(cv2.NORM_L2)
     matches = matcher.match(descs1, descs2)
     return matches


def convert_opencv_matches_to_numpy(matches):
    """Returns a np.ndarray array with points indices correspondences
       with the shape of Nx2 which each N feature is a vector containing
       the keypoints id [id_ref, id_dst].
    """
    assert isinstance(matches, list), type(matches)
    correspondences = []
    for match in matches:
        assert isinstance(match, cv2.DMatch), type(match)
        correspondences.append([match.queryIdx, match.trainIdx])
    return np.asarray(correspondences)


def get_keypoints_coordinates(img, patch_size=32, use_detector=False):

    if use_detector:
        # TODO: Include a Keypoint detector to extract feature coordinates
        print('Implement a Keypoint Detector')
        coordinates = []
    else:
        # Define a grid where to extract patches
        dim_y, dim_x = img.shape[0] // patch_size, img.shape[1] // patch_size
        coordinates = []
        for y in range(dim_y):
            for x in range(dim_x):
                point = [int((y+0.5)*patch_size), int((x+0.5)*patch_size)]
                coordinates.append(point)

    return np.asarray(coordinates)

def rgb2gray(rgb):
    return rgb[...,:3] @ torch.tensor([0.2989, 0.5870, 0.1140]).to(rgb.device)

def convert_numpy_features_to_opencv_keypoints(features):
    """Returns a list of OpenCV keypoints from a np.ndarray array of features
       in the shape of Nx3 which each N features is a vector containing [x, y, radius].
    """
    assert isinstance(features, np.ndarray), type(features)
    # TODO: add proper shape checking
    #assert len(features.shape) == 2 and features.shape[1] == 3, features.shape
    kpts = []
    for feat in features:
        kpts.append(cv2.KeyPoint(x=feat[1], y=feat[0], _size=1))
    return kpts

def extract_patches_from_opencv_keypoints(image, kpts, patch_size=32):
    """Extract image patches from OpenCV keypoints (cv2.KeyPoint)
    """
    _image = image.clone().detach().requires_grad_(False).numpy()
    patches = []
    N = patch_size  # alias
    for kp in kpts:
        x, y, s, a = kp.pt[0], kp.pt[1], kp.size, kp.angle
        cos = np.cos(a * np.pi / 180.0)
        sin = np.sin(a * np.pi / 180.0)

        H = np.matrix([
            [+s * cos, -s * sin, (-s * cos + s * sin) * N / 2.0 + x],
            [+s * sin, +s * cos, (-s * sin - s * cos) * N / 2.0 + y]
        ])

        res = cv2.warpAffine(_image, H, (N, N),
            flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)
        patches.append(res)
    return np.array(patches)

def extract_patches_from_coords(image, kpts, patch_size=32):
    N = patch_size
    N_half = N // 2

    dim_y, dim_x = image.shape[0] // patch_size, image.shape[1] // patch_size
    patches = torch.Tensor(dim_y*dim_x, N, N)
    for i, kp in enumerate(kpts):
        patches[i] = image[kp[0]-N_half:kp[0]+N_half, kp[1]-N_half:kp[1]+N_half]

    return patches

def compute_desc(img, points, checkpoint_path, gpu_ids=[]):

    #kpts = convert_numpy_features_to_opencv_keypoints(points)
    kpts = points 

    descriptor = HardNetDescriptor(checkpoint_path, gpu_ids)
    _, descs = descriptor.compute(img, kpts, None)
    return descs