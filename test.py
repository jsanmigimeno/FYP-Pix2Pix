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
from collections import OrderedDict
import csv
import pdb

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

    # load split
    splits = []
    if opt.dataset_split is not None:
        with open(opt.dataset_split, 'r') as f:
            reader = csv.reader(f)
            for line in enumerate(reader):
                splits.append(line[1])
        nSplits = len(splits)
    else:
        nSplits = 1
    splitProp = np.array([0]*nSplits)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    metrics = {
        'L1'                : np.array([0.]*nSplits),
        'PSNR'              : np.array([0.]*nSplits),
        'SSIM'              : np.array([0.]*nSplits),
        'DescL1_Grid'       : np.array([0.]*nSplits),
        'Matching_Grid'     : np.array([0.]*nSplits),
        'DescL1_GBest'      : np.array([0.]*nSplits),
        'Matching_GBest'    : np.array([0.]*nSplits),
        'DescL1_Det'        : np.array([0.]*nSplits),
        'Matching_Det'      : np.array([0.]*nSplits),
    }
    

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        img_name = os.path.split(img_path[0])[-1]
        if i % 5 == 0:  
            print('processing (%04d)-th image... %s' % (i, img_path))

        # Get splitId
        splitId = None
        if opt.dataset_split is not None:
            for i, split in enumerate(splits):
                if img_name in split:
                    splitId = i
                    break
            if splitId is None:
                raise Exception("Split not found for %s" % img_name)
        else:
            splitId = 0

        # Get losses
        splitProp[splitId] += 1

        L1 = model.get_L1_loss()
        metrics['L1'][splitId] += L1
        PSNR = model.get_PSNR()
        metrics['PSNR'][splitId] += PSNR
        SSIM = model.get_SSIM()
        metrics['SSIM'][splitId] += SSIM

        descriptorL1GBest = 0
        matchingGBest = 0
        pdb.set_trace()
        descriptorL1GBest, matchingGBest, descriptorL1, matching = model.get_Descriptor_loss_and_matching(getMatching=True, num_points=opt.num_points, includeAllAlways=True)
        metrics['DescL1_Grid'][splitId] += descriptorL1
        metrics['Matching_Grid'][splitId] += matching
        metrics['DescL1_GBest'][splitId] += descriptorL1GBest
        metrics['Matching_GBest'][splitId] += matchingGBest

        descriptorL1_Det = 0
        matching_Det = 0
        if opt.use_detector:
            descriptorL1_Det, matching_Det, *_ = model.get_Descriptor_loss_and_matching(getMatching=True, useDetector=True, num_points=opt.num_points)
            metrics['DescL1_Det'][splitId] += descriptorL1_Det
            metrics['Matching_Det'][splitId] += matching_Det

        # save images to an HTML file
        if opt.save_fake_only:
            od = OrderedDict()
            od['fake_B'] = visuals['fake_B']
            visuals = od

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, compression=opt.output_extension)
        webpage.add_text(("Losses - L1: %.4f, PSNR: %.4f, SSIM: %.4f, descriptor L1: %.4f, matching score: %.4f, descriptor L1 (best): %.4f, matching score (best): %.4f, descriptor L1 (det): %.4f, matching score (det): %.4f" % (L1, PSNR, SSIM, descriptorL1, matching, descriptorL1GBest, matchingGBest, descriptorL1_Det, matching_Det)))

    # Get average values
    for splitId in range(nSplits):
        for key, value in metrics.items():
            if splitProp[splitId] == 0:
                break
            metrics[key][splitId] = value[splitId]/splitProp[splitId]

    test_size = len(dataset)
    webpage.add_header("Overall performance")
    for splitId in range(nSplits):
        message = 'Split %i: \t' % splitId
        for key, value in metrics.items():
            message += key + ": %.6f\t" % value[splitId]
        webpage.add_text(message)
        print(message)

    splitProp = splitProp/np.sum(splitProp)
    message = 'Total: \t\t'
    for key, value in metrics.items():
        message += key + ": %.6f\t" % np.matmul(value, splitProp/np.sum(splitProp))
    webpage.add_text(message)
    print(message)
    webpage.save()  # save the HTML