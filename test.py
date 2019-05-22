"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np


if __name__ == '__main__':

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    L1_total = 0
    PSNR_total = 0
    SSIM_total = 0
    matching_total = 0
    descriptor_L1_total = 0
    matching_total_Det = 0
    descriptor_L1_total_Det = 0
    descriptorL1_Det = 0
    matching_Det = 0

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  
            print('processing (%04d)-th image... %s' % (i, img_path))

        # Get losses
        L1 = model.get_L1_loss()
        L1_total += L1
        PSNR = model.get_PSNR()
        PSNR_total += PSNR
        SSIM = model.get_SSIM()
        SSIM_total += SSIM
        descriptorL1, matching = model.get_Descriptor_loss_and_matching(getMatching=True)
        matching_total += matching
        descriptor_L1_total += descriptorL1
        if opt.use_detector:
            descriptorL1_Det, matching_Det = model.get_Descriptor_loss_and_matching(getMatching=True, useDetector=True, num_points=opt.num_points)
            matching_total_Det += matching_Det
            descriptor_L1_total_Det += descriptorL1_Det

        # save images to an HTML file
        if opt.save_fake_only:
            visuals = visuals['fake_B']
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, compression=opt.output_extension)
        webpage.add_text(("Losses - L1: %.4f, PSNR: %.4f, SSIM: %.4f, descriptor L1: %.4f, matching score: %.4f, descriptor L1 (det): %.4f, matching score (det): %.4f" % (L1, PSNR, SSIM, descriptorL1, matching, descriptorL1_Det, matching_Det)))

    test_size = len(dataset)
    webpage.add_header("Overall performance")
    webpage.add_text(("L1: %.4f, PSNR %.4f, SSIM: %.4f, descriptor L1: %.4f, matching score: %.4f, descriptor L1 (det): %.4f, matching score (det): %.4f" % (L1_total/test_size, PSNR_total/test_size, SSIM_total/test_size, descriptor_L1_total/test_size, matching_total/test_size, descriptor_L1_total_Det/test_size, matching_total_Det/test_size)))    
    webpage.save()  # save the HTML

    print("L1: %.4f, PSNR %.4f, SSIM: %.4f, descriptor L1: %.4f, matching score: %.4f, descriptor L1 (det): %.4f, matching score (det): %.4f" % (L1_total/test_size, PSNR_total/test_size, SSIM_total/test_size, descriptor_L1_total/test_size, matching_total/test_size, descriptor_L1_total_Det/test_size, matching_total_Det/test_size))
