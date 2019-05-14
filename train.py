"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import copy
import os
from util import html
from collections import OrderedDict

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # Validation
    if opt.use_validation:
        opt_val = copy.deepcopy(opt)
        print('\nUsing validation.  Parameters:')
        opt_val.phase = 'val'
        #opt_val.name = 'val_' + opt_val.name
        opt_val.num_threads = 0   # test code only supports num_threads = 1
        opt_val.batch_size = 1    # test code only supports batch_size = 1
        opt_val.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt_val.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        opt_val.display_id = -1     # no visdom display; the test code saves the results to a HTML file.
        dataset_val = create_dataset(opt_val)
        dataset_val_size = len(dataset_val)
        print('The number of validation images = %d' % dataset_val_size)
        
        opt_val.ntest = float("inf")
        opt_val.results_dir='./val/'
        opt_val.aspect_ratio =1.0
        opt_val.num_test = dataset_val_size
        opt_val.isTrain = False
        opt_val.load_size = opt_val.crop_size

        model_val = create_model(opt_val)
        model_val.setup(opt_val)
        model_val.netD = model.netD
        model_val.netG = model.netG
        total_val_iters = 0
        train_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'train_loss_log.txt')
        val_log_name = os.path.join(opt_val.checkpoints_dir, opt_val.name, 'val_loss_log.txt')
        
        # Training Log
        with open(train_log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        
        # Validation Log
        with open(val_log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Validation Loss (%s) ================\n' % now)

        # Best epoch
        bestEpochLoss = float('inf')

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        # Initialise var where to save epoch losses
        epoch_losses = OrderedDict()
        for name in model.loss_names:
            if isinstance(name, str):
                epoch_losses[name] = 0
        # Save total G loss of epoch
        currentGLoss = 0

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # Save iteration losses
            losses = model.get_current_losses()
            for name in losses:
                epoch_losses[name] = epoch_losses[name] + losses[name]
            currentGLoss += float(model.loss_G)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        
        if epoch % opt.save_epoch_freq == 0 or currentGLoss < bestEpochLoss:              # cache our model every <save_epoch_freq> epochs or if current loss is the lowest
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            bestEpochLoss = currentGLoss

        # Save losses to log
        message = 'Epoch: %i ' % epoch
        for loss_name, loss_val in epoch_losses.items():
            message += '%s: %.4f ' % (loss_name, loss_val/dataset_size)
        with open(train_log_name, 'a') as train_log:
            train_log.write('%s\n' % message)

        # Validation
        if opt.use_validation:
            L1_total = 0
            PSNR_total = 0
            SSIM_total = 0
            matching_total = 0
            descriptor_L1_total = 0

            for i, data in enumerate(dataset_val):
                model_val.set_input(data)  # unpack data from data loader
                model_val.test()           # run inference
                # Get losses
                L1_total += model_val.get_L1_loss()
                PSNR_total += model_val.get_PSNR()
                SSIM_total += model_val.get_SSIM()
                descriptorL1, matching = model_val.get_Descriptor_loss_and_matching(getMatching=True)
                matching_total += matching
                descriptor_L1_total += descriptorL1

                if i % 50 == 0:  # Print state  
                    img_path = model_val.get_image_paths()     # get image paths
                    print('processing (%04d)-th image... %s' % (i, img_path))

            message = "Epoch: %.4f Val_L1: %.4f Val_PSNR %.4f Val_SSIM: %.4f Val_Desc: %.4f Val_Match: %.4f" % (epoch, L1_total/dataset_val_size, PSNR_total/dataset_val_size, SSIM_total/dataset_val_size, descriptor_L1_total/dataset_val_size, matching_total/dataset_val_size)
            with open(val_log_name, 'a') as val_log:
                val_log.write('%s\n' % message)


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.