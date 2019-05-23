import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from .pytorch_sift import SIFTNet

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

        self._load_model(checkpoint_path)  # load pretrained model

    def _load_model(self, checkpoint_path):
        assert os.path.isfile(checkpoint_path), \
            "Invalid file: {}".format(checkpoint_path)
        
        if checkpoint_path != '':
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

class SIFTDescriptor():
    def __init__(self, patch_size=32, gpu_ids=[]):
        self.patch_size = patch_size
        self.gpu_ids = gpu_ids

        self.SIFT = SIFTNet(patch_size=patch_size)

        # CPU/GPU
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.SIFT.to(gpu_ids[0])

    def compute(self, image, kpts, mask=None):
        patches = extract_patches_from_coords(image, kpts)
        patches = torch.unsqueeze(patches, dim=1)
        patches = patches.float() / 255.
        # inference model with patches
        descs = self.SIFT(patches)
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


def get_keypoints_coordinates(imgA, imgB, patch_size=32, use_detector=False, num_points=None, detector=None, nonEmptyOnly=False):
    if use_detector:
        if num_points is None:
            raise Exception('Number of patches to extract was not specified!')

        img_local = imgB.clone().cpu().numpy()
        if img_local.min() < 0.0:
            img_local = img_local-img_local.min()
        img_local = np.asarray(255 * (img_local / img_local.max()), np.uint8)
        if not detector:
            # FAST as the default detector
            detector = cv2.FastFeatureDetector_create()
        openCV_kps = detector.detect(img_local, None)
        openCV_kps.sort(key=lambda x: x.response, reverse=True)
        coordinates = [[int(kp.pt[1]), int(kp.pt[0])] for kp in openCV_kps[:num_points]]
        mask = np.array([True]*len(coordinates))
    else:
        coordinates = []
        keepPatch = []
        variance = []

        if nonEmptyOnly or num_points is not None:
            img_local = imgA.clone().cpu().numpy()
            if img_local.min() < 0.0:
                img_local = img_local-img_local.min()
                
        # Define a grid where to extract patches
        dim_y, dim_x = imgB.shape[0] // patch_size, imgB.shape[1] // patch_size
        for y in range(dim_y):
            for x in range(dim_x):
                point = [int((y+0.5)*patch_size), int((x+0.5)*patch_size)]
                coordinates.append(point)
                
                # Get patch variance
                if num_points is not None:
                    variance.append(np.var(img_local[int(y*patch_size):int((y+1)*patch_size), int(x*patch_size):int((x+1)*patch_size)]))
                # Filter out empty patches
                elif nonEmptyOnly:
                    if np.max(img_local[int(y*patch_size):int((y+1)*patch_size), int(x*patch_size):int((x+1)*patch_size)]) > 0:
                        keepPatch.append(True)
                    else:
                        keepPatch.append(False)
                else:
                    keepPatch.append(True)

        

        if num_points is not None:
            variance = np.array(variance)
            sortIdx = np.flip(np.argsort(variance))
            selectIdx = sortIdx[0:num_points]
            mask = np.zeros(len(variance), dtype='bool')
            mask[selectIdx] = True
        else:
            mask = np.array(keepPatch)

    return np.asarray(coordinates), mask

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

    dim_y, dim_x = image.shape[0], image.shape[1]
    nPathes = len(kpts)
    patches = torch.Tensor(nPathes, N, N).to(image.device)

    for i, kp in enumerate(kpts):
        y1 = kp[0]-N_half
        y2 = kp[0]+N_half
        if y1 < 0:
            yOffset = -y1
        elif y2 > dim_y:
            yOffset = dim_y - y2
        else:
            yOffset = 0
    
        x1 = kp[1]-N_half
        x2 = kp[1]+N_half
        if x1 < 0:
            xOffset = -x1
        elif x2 > dim_x:
            xOffset = dim_x - x2
        else:
            xOffset = 0
        
        patches[i] = image[y1+yOffset:y2+yOffset, x1+xOffset:x2+xOffset]
    return patches

def compute_desc(img, points, descType, checkpoint_path, gpu_ids=[]):

    #kpts = convert_numpy_features_to_opencv_keypoints(points)
    kpts = points 

    if descType == 'HardNet':
        descriptor = HardNetDescriptor(checkpoint_path, gpu_ids=gpu_ids)
    elif descType == 'SIFT':
        descriptor = SIFTDescriptor(gpu_ids=gpu_ids)
    else:
        raise Exception('Descriptor %s not implemented' % descType)
    _, descs = descriptor.compute(img, kpts, None)
    return descs