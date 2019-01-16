import datetime
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torchvision.utils as vutils

from torch.backends import cudnn

import utils
from sagan_models import Generator, Discriminator


class Trainer(object):

    def __init__(self, config):

        # Images data path & Output path
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.save_path = os.path.join(config.save_path, config.name)

        # Training settings
        self.batch_size = config.batch_size
        self.total_step = config.total_step
        self.d_steps_per_iter = config.d_steps_per_iter
        self.g_steps_per_iter = config.g_steps_per_iter
        self.d_lr = config.d_lr
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.inst_noise_sigma = config.inst_noise_sigma
        self.inst_noise_sigma_iters = config.inst_noise_sigma_iters
        self.start = 0 # Unless using pre-trained model

        # Image transforms
        self.shuffle = not config.dont_shuffle
        self.drop_last = not config.dont_drop_last
        self.resize = not config.dont_resize
        self.imsize = config.imsize
        self.centercrop = config.centercrop
        self.centercrop_size = config.centercrop_size

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.save_n_images = config.save_n_images
        self.max_frames_per_gif = config.max_frames_per_gif

        # Pretrained model
        self.pretrained_model = config.pretrained_model
        # Check if self.pretrained_model exists
        if self.pretrained_model:
            assert os.path.exists(self.pretrained_model), "Path of .pth pretrained_model doesn't exist! Given: " + self.pretrained_model

        # Misc
        self.manual_seed = config.manual_seed
        self.disable_cuda = config.disable_cuda
        self.parallel = config.parallel
        self.num_workers = config.num_workers

        # Output paths
        self.model_weights_path = os.path.join(self.save_path, config.model_weights_dir)
        self.sample_path = os.path.join(self.save_path, config.sample_dir)

        # Model hyper-parameters
        self.adv_loss = config.adv_loss
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.lambda_gp = config.lambda_gp

        # Model name
        self.name = config.name

        # Create directories if not exist
        utils.make_folder(self.save_path)
        utils.make_folder(self.model_weights_path)
        utils.make_folder(self.sample_path)

        # Copy files
        utils.write_config_to_file(config, self.save_path)
        utils.copy_scripts(self.save_path)

        # Make dataloader
        self.dataloader, self.num_of_classes = utils.make_dataloader(self.batch_size, self.dataset, self.data_path,
                                                                     self.shuffle, self.num_workers, self.drop_last,
                                                                     self.resize, self.imsize, self.centercrop, self.centercrop_size)

        # Data iterator
        self.data_iter = iter(self.dataloader)

        # Check for CUDA
        if not self.disable_cuda and torch.cuda.is_available():
            print("CUDA is available!")
            self.device = torch.device('cuda')
        else:
            print("Cuda is NOT available, running on CPU.")
            self.device = torch.device('cpu')

        if torch.cuda.is_available() and self.disable_cuda:
            print("WARNING: You have a CUDA device, so you should probably run without --disable_cuda")

        # Build G and D
        self.build_models()

        # Start with pretrained model
        if self.pretrained_model:
            self.load_pretrained_model()

        if self.adv_loss == 'dcgan':
            self.criterion = nn.BCELoss()

    def train(self):

        # Seed
        np.random.seed(self.manual_seed)
        random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)

        # For fast training
        cudnn.benchmark = True

        # For BatchNorm
        self.G.train()
        self.D.train()

        # Fixed noise for sampling from G
        fixed_noise = torch.randn(self.batch_size, self.z_dim, device=self.device)
        if self.num_of_classes < self.batch_size:
            fixed_labels = torch.from_numpy(np.tile(np.arange(self.num_of_classes), self.batch_size//self.num_of_classes + 1)[:self.batch_size]).to(self.device)
        else:
            fixed_labels = torch.from_numpy(np.arange(self.batch_size)).to(self.device)

        # For gan loss
        label = torch.full((self.batch_size,), 1, device=self.device)
        ones = torch.full((self.batch_size,), 1, device=self.device)

        # Losses file
        log_file_name = os.path.join(self.save_path, 'log.txt')
        log_file = open(log_file_name, "wt")

        # Init
        start_time = time.time()
        G_losses = []
        D_losses_real = []
        D_losses_fake = []
        D_losses = []
        D_xs = []
        D_Gz_trainDs = []
        D_Gz_trainGs = []

        # Instance noise - make random noise mean (0) and std for injecting
        inst_noise_mean = torch.full((self.batch_size, 3, self.imsize, self.imsize), 0, device=self.device)
        inst_noise_std = torch.full((self.batch_size, 3, self.imsize, self.imsize), self.inst_noise_sigma, device=self.device)

        # Start training
        for self.step in range(self.start, self.total_step):

            # Instance noise std is linearly annealed from self.inst_noise_sigma to 0 thru self.inst_noise_sigma_iters
            inst_noise_sigma_curr = 0 if self.step > self.inst_noise_sigma_iters else (1 - self.step/self.inst_noise_sigma_iters)*self.inst_noise_sigma
            inst_noise_std.fill_(inst_noise_sigma_curr)

            # ================== TRAIN D ================== #

            for _ in range(self.d_steps_per_iter):

                # Zero grad
                self.reset_grad()

                # TRAIN with REAL

                # Get real images & real labels
                real_images, real_labels = self.get_real_samples()

                # Get D output for real images & real labels
                inst_noise = torch.normal(mean=inst_noise_mean, std=inst_noise_std).to(self.device)
                d_out_real = self.D(real_images + inst_noise, real_labels)

                # Compute D loss with real images & real labels
                if self.adv_loss == 'hinge':
                    d_loss_real = torch.nn.ReLU()(ones - d_out_real).mean()
                elif self.adv_loss == 'wgan_gp':
                    d_loss_real = -d_out_real.mean()
                else:
                    label.fill_(1)
                    d_loss_real = self.criterion(d_out_real, label)

                # Backward
                d_loss_real.backward()

                # TRAIN with FAKE

                # Create random noise
                z = torch.randn(self.batch_size, self.z_dim, device=self.device)

                # Generate fake images for same real labels
                fake_images = self.G(z, real_labels)

                # Get D output for fake images & same real labels
                inst_noise = torch.normal(mean=inst_noise_mean, std=inst_noise_std).to(self.device)
                d_out_fake = self.D(fake_images.detach() + inst_noise, real_labels)

                # Compute D loss with fake images & real labels
                if self.adv_loss == 'hinge':
                    d_loss_fake = torch.nn.ReLU()(ones + d_out_fake).mean()
                elif self.adv_loss == 'dcgan':
                    label.fill_(0)
                    d_loss_fake = self.criterion(d_out_fake, label)
                else:
                    d_loss_fake = d_out_fake.mean()

                # Backward
                d_loss_fake.backward()

                # If WGAN_GP, compute GP and add to D loss
                if self.adv_loss == 'wgan_gp':
                    d_loss_gp = self.lambda_gp * self.compute_gradient_penalty(real_images, real_labels, fake_images.detach())
                    d_loss_gp.backward()

                # Optimize
                self.D_optimizer.step()

            # ================== TRAIN G ================== #

            for _ in range(self.g_steps_per_iter):

                # Zero grad
                self.reset_grad()

                # Get real images & real labels (only need real labels)
                real_images, real_labels = self.get_real_samples()

                # Create random noise
                z = torch.randn(self.batch_size, self.z_dim).to(self.device)

                # Generate fake images for same real labels
                fake_images = self.G(z, real_labels)

                # Get D output for fake images & same real labels
                inst_noise = torch.normal(mean=inst_noise_mean, std=inst_noise_std).to(self.device)
                g_out_fake = self.D(fake_images + inst_noise, real_labels)

                # Compute G loss with fake images & real labels
                if self.adv_loss == 'dcgan':
                    label.fill_(1)
                    g_loss = self.criterion(g_out_fake, label)
                else:
                    g_loss = -g_out_fake.mean()

                # Backward + Optimize
                g_loss.backward()
                self.G_optimizer.step()

            # Print out log info
            if self.step % self.log_step == 0:
                G_losses.append(g_loss.mean().item())
                D_losses_real.append(d_loss_real.mean().item())
                D_losses_fake.append(d_loss_fake.mean().item())
                D_loss = D_losses_real[-1] + D_losses_fake[-1]
                if self.adv_loss == 'wgan_gp':
                    D_loss += d_loss_gp.mean().item()
                D_losses.append(D_loss)
                D_xs.append(d_out_real.mean().item())
                D_Gz_trainDs.append(d_out_fake.mean().item())
                D_Gz_trainGs.append(g_out_fake.mean().item())
                curr_time = time.time()
                curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
                elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
                log = ("[{}] : Elapsed [{}], Iter [{} / {}], G_loss: {:.4f}, D_loss: {:.4f}, D_loss_real: {:.4f}, D_loss_fake: {:.4f}, D(x): {:.4f}, D(G(z))_trainD: {:.4f}, D(G(z))_trainG: {:.4f}".
                       format(curr_time_str, elapsed, self.step, self.total_step,
                              G_losses[-1], D_losses[-1], D_losses_real[-1], D_losses_fake[-1],
                              D_xs[-1], D_Gz_trainDs[-1], D_Gz_trainGs[-1]))
                print(log)
                log_file.write(log)
                log_file.flush()
                utils.make_plots(G_losses, D_losses, D_losses_real, D_losses_fake, D_xs, D_Gz_trainDs, D_Gz_trainGs,
                                 self.log_step, self.save_path)

            # Sample images
            if self.step % self.sample_step == 0:
                fake_images = self.G(fixed_noise, fixed_labels)
                sample_images = utils.denorm(fake_images.detach()[:self.save_n_images])
                # Save batch images
                vutils.save_image(sample_images, os.path.join(self.sample_path, 'fake_{:05d}.png'.format(self.step)))
                # Save gif
                utils.make_gif(sample_images[0].cpu().numpy().transpose(1, 2, 0)*255, self.step,
                               self.sample_path, self.name, max_frames_per_gif=self.max_frames_per_gif)

            # Save model
            if self.step % self.model_save_step == 0:
                utils.save_ckpt(self)

    def build_models(self):
        self.G = Generator(self.z_dim, self.g_conv_dim, self.num_of_classes).to(self.device)
        self.D = Discriminator(self.d_conv_dim, self.num_of_classes).to(self.device)
        if 'cuda' in self.device.type and self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.G_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        # print networks
        print(self.G)
        print(self.D)

    def reset_grad(self):
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()

    def get_real_samples(self):
        try:
            real_images, real_labels = next(self.data_iter)
        except:
            self.data_iter = iter(self.dataloader)
            real_images, real_labels = next(self.data_iter)

        real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
        return real_images, real_labels

    def load_pretrained_model(self):
        checkpoint = torch.load(self.pretrained_model)
        try:
            self.start = checkpoint['step'] + 1
            self.G.load_state_dict(checkpoint['G_state_dict'])
            self.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
        except:
            self.start = checkpoint['step'] + 1
            self.G = torch.load(checkpoint['G']).to(self.device)
            self.G_optimizer = torch.load(checkpoint['G_optimizer'])
            self.D = torch.load(checkpoint['D']).to(self.device)
            self.D_optimizer = torch.load(checkpoint['D_optimizer'])

    def compute_gradient_penalty(self, real_images, real_labels, fake_images):
        # Compute gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1).expand_as(real_images).to(device)
        interpolated = torch.tensor(alpha * real_images + (1 - alpha) * fake_images, requires_grad=True)
        out = self.D(interpolated, real_labels)
        exp_grad = torch.ones(out.size()).to(device)
        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated,
                                   grad_outputs=exp_grad,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
        return d_loss_gp
