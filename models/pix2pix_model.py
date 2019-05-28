import torch
from .base_model import BaseModel
from . import networks
from math import log10
from util import ssim
import numpy as np
from util.matching_tools import utils as matching_utils
import sys

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='unaligned')
        parser.add_argument('--desc_weights_path', type=str, default='./util/matching_tools/HardNet++.pth', help='relative path to HardNet descriptor weights')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--lambda_desc', type=float, default=0.0, help='weight for descriptor loss')
        parser.add_argument('--lambda_GAN', type=float, default=1, help='wheight for gan loss for debugging purposes')
        parser.add_argument('--siamese_descriptor', action='store_true', help='use siamese network for descriptor loss')
        parser.add_argument('--per_channel_descriptor', action='store_true', help='compute descriptor for each RGB channel')
        parser.add_argument('--use_detector', action='store_true', help='use detector when extracting patches')
        parser.add_argument('--descriptor', type=str, default='HardNet', help='descriptor to be used for loss computation: HardNet|SIFT')
        parser.add_argument('--non_empty_patches_only', action='store_true', help='only select non empty patches for descriptor')
        parser.add_argument('--num_points', type=int, default=None, help='number of points to detect for patch extraction by the detector. This option overrides non_empty_patches_only')
        parser.add_argument('--log_trans', action='store_true', help='perform logarithmic transfor to the dataset')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.opt.lambda_desc == 0:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        else:
            self.loss_names = ['G_GAN', 'G_L1', 'G_Desc', 'G_Matching', 'D_real', 'D_fake']
        
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain or opt.phase == 'val':
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain or opt.phase == 'val':  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        if self.isTrain or opt.phase == 'val':
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def fake_forward(self):
        self.fake_B = self.real_A

    def forward_real(self):
        """Run forward pass on ground truth image"""
        self.fake_real_B = self.netG(self.real_B)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        if torch.isnan(self.loss_D):
            data = {
                'fake_AB'       : fake_AB.clone().cpu(),
                'pred_fake'     : pred_fake.clone().cpu(),
                'loss_D_fake'   : self.loss_D_fake.clone().cpu(),
                'real_AB'       : real_AB.clone().cpu(),
                'pred_real'     : pred_real.clone().cpu(),
                'loss_D_real'   : self.loss_D_real.clone().cpu(),
            }
            raise Exception('DiscriminatorNaN', data)
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * (self.opt.lambda_GAN)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # Third, descriptor loss
        if self.opt.lambda_desc != 0:
            if not self.opt.siamese_descriptor:
                descriptorLoss, self.loss_G_Matching, *_ = self.get_Descriptor_loss_and_matching(getMatching=True, useDetector=self.opt.use_detector, descType=self.opt.descriptor, num_points=self.opt.num_points)
            else:
                self.forward_real()
                descriptorLoss, self.loss_G_Matching, *_ = self.get_Descriptor_loss_and_matching(getMatching=True, useFakeRealB=True, useDetector=self.opt.use_detector, descType=self.opt.descriptor, num_points=self.opt.num_points)
            self.loss_G_Desc = descriptorLoss*self.opt.lambda_desc 
            self.loss_G = self.loss_G + self.loss_G_Desc
        # Catch exploding gradients
        if torch.isnan(self.loss_G):
            data = {
                'fake_AB'       : fake_AB.clone().cpu(),
                'pred_fake'     : pred_fake.clone().cpu(),
                'loss_G_GAN'    : self.loss_G_GAN.clone().cpu(),
                'loss_G_L1'     : self.loss_G_L1.clone().cpu(),
                'loss_G_Matching' : self.loss_G_Matching.clone().cpu(),
                'descriptorLoss' : self.loss_G_Desc.clone().cpu(),
            }
            raise Exception('GeneratorNaN', data)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def get_L1_loss(self, output='scalar'):
        L1 = self.criterionL1(self.fake_B, self.real_B)
        if output=='tensor':
            return L1
        else:
            return L1.item() 

    def get_PSNR(self, output='scalar'):
        mse = self.criterionMSE(self.fake_B, self.real_B)
        PSNR = 10 * torch.log10(1/mse)
        if output=='tensor':
            return PSNR
        else:
            return PSNR.item() 

    def get_SSIM(self, output='scalar'):
        ssimMeasure = ssim.ssim(self.fake_B, self.real_B)
        if output=='tensor':
            return ssimMeasure
        else:
            return ssimMeasure.item() 

    def get_Descriptor_loss_and_matching(self, getMatching=False, useFakeRealB=False, useDetector=False, descType='HardNet', num_points=None, includeAllAlways=False):

        #Path to checkpoint
        checkpoint_path = self.opt.desc_weights_path

        # Initialise real_B
        real_B = None

        # Convert to Grayscale
        if not self.opt.per_channel_descriptor: # Use grayscale patches
            if not useFakeRealB:
                # Standard loss
                real_B = matching_utils.rgb2gray(self.real_B[0].permute(1, 2, 0)).unsqueeze(2)
            else:
                #Siamese loss
                fake_real_B = matching_utils.rgb2gray(self.fake_real_B[0].permute(1, 2, 0)).unsqueeze(2)
            fake_B = matching_utils.rgb2gray(self.fake_B[0].permute(1, 2, 0)).unsqueeze(2)
        else:   # Use 3 different channels for loss
            if not useFakeRealB:
                # Standard loss
                real_B = self.real_B[0].permute(1, 2, 0)
            else:
                # Siamese loss
                fake_real_B = self.fake_real_B[0].permute(1, 2, 0)
            fake_B = self.fake_B[0].permute(1, 2, 0)

        # Get real A to filter out empty patches (all 0)
        if self.opt.non_empty_patches_only or num_points is not None:
            real_A = matching_utils.rgb2gray(self.real_A[0].permute(1, 2, 0)).unsqueeze(2)
        else:
            real_A = None

        # Get patches coordinates
        if real_B is not None and real_B.shape[2] == 1:
            real_B_gray = real_B
        else:
            real_B_gray = matching_utils.rgb2gray(self.real_B[0].permute(1, 2, 0)).unsqueeze(2)
        
        indexes, mask = matching_utils.get_keypoints_coordinates(real_A, real_B_gray, use_detector=useDetector, num_points=num_points, nonEmptyOnly=self.opt.non_empty_patches_only)

        if not includeAllAlways:
            indexes = indexes[mask]
            mask = np.array([True]*len(indexes))

        nChannels = fake_B.shape[2]

        L1Loss = torch.tensor(0).to(fake_B.device)
        L1Loss_All = torch.tensor(0).to(fake_B.device)

        matching_score = 0
        matching_score_All = 0

        for channel in range(nChannels):
            if self.opt.non_empty_patches_only and list(mask).count(True)<=1:
                break

            if not useFakeRealB:
                real_B_T = real_B[..., channel]
                desc_real_B = matching_utils.compute_desc(real_B_T, indexes, descType=descType, checkpoint_path=checkpoint_path, gpu_ids=self.gpu_ids)
            else:
                fake_real_B_T = fake_real_B[..., channel]
                desc_real_B = matching_utils.compute_desc(fake_real_B_T, indexes, descType=descType, checkpoint_path=checkpoint_path, gpu_ids=self.gpu_ids)

            fake_B_T = fake_B[..., channel]
            desc_fake_B = matching_utils.compute_desc(fake_B_T, indexes, descType=descType, checkpoint_path=checkpoint_path, gpu_ids=self.gpu_ids)

            if getMatching:
                desc_real_B_N = desc_real_B.clone().cpu().detach().numpy()
                desc_fake_B_N = desc_fake_B.clone().cpu().detach().numpy()

                # match descriptors
                matches = matching_utils.match(desc_real_B_N[mask], desc_fake_B_N[mask])
                matches_np = matching_utils.convert_opencv_matches_to_numpy(matches)
                if len(matches_np) != 0:
                    true_matches = np.where(matches_np[:, 0] == matches_np[:, 1], 1., 0.)
                    matching_score += np.sum(true_matches) / len(true_matches)
                
                if includeAllAlways:
                    matches = matching_utils.match(desc_real_B_N, desc_fake_B_N)
                    matches_np = matching_utils.convert_opencv_matches_to_numpy(matches)
                    if len(matches_np) != 0:
                        true_matches = np.where(matches_np[:, 0] == matches_np[:, 1], 1., 0.)
                        matching_score_All += np.sum(true_matches) / len(true_matches)

            L1Loss = L1Loss + self.criterionL1(desc_real_B[np.where(mask)[0]], desc_fake_B[np.where(mask)[0]])
            if includeAllAlways:
                L1Loss_All = L1Loss_All + self.criterionL1(desc_real_B, desc_fake_B)

        if getMatching:
            return L1Loss/nChannels, matching_score/nChannels, L1Loss_All/nChannels, matching_score_All/nChannels
        else:
            return L1Loss/nChannels, L1Loss_All/nChannels

    # def get_Matching(self, useDetector=False, descType='HardNet'):
    #     #Path to checkpoint
    #     checkpoint_path = self.opt.desc_weights_path
    #     # Convert to Grayscale
    #     real_B = matching_utils.rgb2gray(self.real_B[0].permute(1, 2, 0))
    #     fake_B = matching_utils.rgb2gray(self.fake_B[0].permute(1, 2, 0))
    #     indexes = matching_utils.get_keypoints_coordinates(real_B, use_detector=useDetector)

    #     desc_real_B = matching_utils.compute_desc(real_B, indexes, descType=descType, checkpoint_path=checkpoint_path, gpu_ids=self.gpu_ids)
    #     desc_fake_B = matching_utils.compute_desc(fake_B, indexes, descType=descType, checkpoint_path=checkpoint_path, gpu_ids=self.gpu_ids)

    #     desc_real_B = desc_real_B.cpu().numpy()
    #     desc_fake_B = desc_fake_B.cpu().detach().numpy()

    #     # match descriptors
    #     matches = matching_utils.match(desc_real_B, desc_fake_B)
    #     matches_np = matching_utils.convert_opencv_matches_to_numpy(matches)
    #     true_matches = np.where(matches_np[:, 0] == matches_np[:, 1], 1., 0.)
    #     matching_score = np.sum(true_matches) / len(true_matches)
    #     return matching_score