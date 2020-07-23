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

        # Config
        self.config = config

        self.start = 0 # Unless using pre-trained model

        # Create directories if not exist
        utils.make_folder(self.config.save_path)
        utils.make_folder(self.config.model_weights_path)
        utils.make_folder(self.config.sample_images_path)

        # Copy files
        utils.write_config_to_file(self.config, self.config.save_path)
        utils.copy_scripts(self.config.save_path)

        # Check for CUDA
        utils.check_for_CUDA(self)

        # Make dataloader
        self.dataloader, self.num_of_classes = utils.make_dataloader(batch_size=self.config.batch_size_in_gpu,
                                                                     dataset_type=self.config.dataset,
                                                                     data_path=self.config.data_path,
                                                                     shuffle=self.config.shuffle,
                                                                     drop_last=self.config.drop_last,
                                                                     dataloader_args=self.config.dataloader_args,
                                                                     resize=self.config.resize,
                                                                     imsize=self.config.imsize,
                                                                     centercrop=self.config.centercrop,
                                                                     centercrop_size=self.config.centercrop_size,
                                                                     normalize=self.config.normalize,
                                                                     )

        # Data iterator
        self.data_iter = iter(self.dataloader)

        # Build G and D
        self.build_models()

        if self.config.adv_loss == 'dcgan':
            self.criterion = nn.BCELoss()

    def train(self):

        # Seed
        np.random.seed(self.config.manual_seed)
        random.seed(self.config.manual_seed)
        torch.manual_seed(self.config.manual_seed)

        # For fast training
        cudnn.benchmark = True

        # For BatchNorm
        self.G.train()
        self.D.train()

        # Fixed noise for sampling from G
        fixed_noise = torch.randn(self.config.batch_size_in_gpu, self.config.z_dim, device=self.device)
        if self.num_of_classes < self.config.batch_size_in_gpu:
            fixed_labels = torch.from_numpy(np.tile(np.arange(self.num_of_classes), self.config.batch_size_in_gpu//self.num_of_classes + 1)[:self.config.batch_size_in_gpu]).to(self.device)
        else:
            fixed_labels = torch.from_numpy(np.arange(self.config.batch_size_in_gpu)).to(self.device)

        # For gan loss
        label = torch.full((self.config.batch_size_in_gpu,), 1, device=self.device)
        ones = torch.full((self.config.batch_size_in_gpu,), 1, device=self.device)

        # Losses file
        log_file_name = os.path.join(self.config.save_path, 'log.txt')
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
        inst_noise_mean = torch.full((self.config.batch_size_in_gpu, 3, self.config.imsize, self.config.imsize), 0, device=self.device)
        inst_noise_std = torch.full((self.config.batch_size_in_gpu, 3, self.config.imsize, self.config.imsize), self.config.inst_noise_sigma, device=self.device)

        self.gpu_batches = self.config.batch_size//self.config.batch_size_in_gpu

        # Start training
        for self.step in range(self.start, self.config.total_step):

            # Instance noise std is linearly annealed from self.inst_noise_sigma to 0 thru self.inst_noise_sigma_iters
            inst_noise_sigma_curr = 0 if self.step > self.config.inst_noise_sigma_iters else (1 - self.step/self.config.inst_noise_sigma_iters)*self.config.inst_noise_sigma
            inst_noise_std.fill_(inst_noise_sigma_curr)

            # ================== TRAIN D ================== #

            for _ in range(self.config.d_steps_per_iter):

                # Zero grad
                self.reset_grad()

                # Accumulate losses for full batch_size
                # while running GPU computations on only batch_size_in_gpu
                for gpu_batch in range(self.gpu_batches):

                    # TRAIN with REAL

                    # Get real images & real labels
                    real_images, real_labels = self.get_real_samples()

                    # Get D output for real images & real labels
                    inst_noise = torch.normal(mean=inst_noise_mean, std=inst_noise_std).to(self.device)
                    d_out_real = self.D(real_images + inst_noise, real_labels)

                    # Compute D loss with real images & real labels
                    if self.config.adv_loss == 'hinge':
                        d_loss_real = torch.nn.ReLU()(ones - d_out_real).mean()
                    elif self.config.adv_loss == 'wgan_gp':
                        d_loss_real = -d_out_real.mean()
                    else:
                        label.fill_(1)
                        d_loss_real = self.criterion(d_out_real, label)

                    # Backward
                    d_loss_real /= self.gpu_batches
                    d_loss_real.backward()

                    # Delete loss, output
                    if self.step % self.config.log_step != 0 or gpu_batch < self.gpu_batches - 1:
                        del d_out_real, d_loss_real

                    # TRAIN with FAKE

                    # Create random noise
                    z = torch.randn(self.config.batch_size_in_gpu, self.config.z_dim, device=self.device)

                    # Generate fake images for same real labels
                    fake_images = self.G(z, real_labels)

                    # Get D output for fake images & same real labels
                    inst_noise = torch.normal(mean=inst_noise_mean, std=inst_noise_std).to(self.device)
                    d_out_fake = self.D(fake_images.detach() + inst_noise, real_labels)

                    # Compute D loss with fake images & real labels
                    if self.config.adv_loss == 'hinge':
                        d_loss_fake = torch.nn.ReLU()(ones + d_out_fake).mean()
                    elif self.config.adv_loss == 'dcgan':
                        label.fill_(0)
                        d_loss_fake = self.criterion(d_out_fake, label)
                    else:
                        d_loss_fake = d_out_fake.mean()

                    # If WGAN_GP, compute GP and add to D loss
                    if self.config.adv_loss == 'wgan_gp':
                        d_loss_gp = self.config.lambda_gp * self.compute_gradient_penalty(real_images, real_labels, fake_images.detach())
                        d_loss_fake += d_loss_gp

                    # Backward
                    d_loss_fake /= self.gpu_batches
                    d_loss_fake.backward()

                    # Delete loss, output
                    del fake_images
                    if self.step % self.config.log_step != 0 or gpu_batch < self.gpu_batches - 1:
                        del d_out_fake, d_loss_fake

                # Optimize
                self.D_optimizer.step()

            # ================== TRAIN G ================== #

            for _ in range(self.config.g_steps_per_iter):

                # Zero grad
                self.reset_grad()

                # Accumulate losses for full batch_size
                # while running GPU computations on only batch_size_in_gpu
                for gpu_batch in range(self.gpu_batches):

                    # Get real images & real labels (only need real labels)
                    real_images, real_labels = self.get_real_samples()

                    # Create random noise
                    z = torch.randn(self.config.batch_size_in_gpu, self.config.z_dim).to(self.device)

                    # Generate fake images for same real labels
                    fake_images = self.G(z, real_labels)

                    # Get D output for fake images & same real labels
                    inst_noise = torch.normal(mean=inst_noise_mean, std=inst_noise_std).to(self.device)
                    g_out_fake = self.D(fake_images + inst_noise, real_labels)

                    # Compute G loss with fake images & real labels
                    if self.config.adv_loss == 'dcgan':
                        label.fill_(1)
                        g_loss = self.criterion(g_out_fake, label)
                    else:
                        g_loss = -g_out_fake.mean()

                    # Backward
                    g_loss /= self.gpu_batches
                    g_loss.backward()

                    # Delete loss, output
                    del fake_images
                    if self.step % self.config.log_step != 0 or gpu_batch < self.gpu_batches - 1:
                        del g_out_fake, g_loss

                # Optimize
                self.G_optimizer.step()

            # Print out log info
            if self.step % self.config.log_step == 0:
                G_losses.append(g_loss.mean().item())
                D_losses_real.append(d_loss_real.mean().item())
                D_losses_fake.append(d_loss_fake.mean().item())
                D_loss = D_losses_real[-1] + D_losses_fake[-1]
                if self.config.adv_loss == 'wgan_gp':
                    D_loss += d_loss_gp.mean().item()
                D_losses.append(D_loss)
                D_xs.append(d_out_real.mean().item())
                D_Gz_trainDs.append(d_out_fake.mean().item())
                D_Gz_trainGs.append(g_out_fake.mean().item())
                curr_time = time.time()
                curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
                elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
                log = ("[{}] : Elapsed [{}], Iter [{} / {}], G_loss: {:.4f}, D_loss: {:.4f}, D_loss_real: {:.4f}, D_loss_fake: {:.4f}, D(x): {:.4f}, D(G(z))_trainD: {:.4f}, D(G(z))_trainG: {:.4f}\n".
                       format(curr_time_str, elapsed, self.step, self.config.total_step,
                              G_losses[-1], D_losses[-1], D_losses_real[-1], D_losses_fake[-1],
                              D_xs[-1], D_Gz_trainDs[-1], D_Gz_trainGs[-1]))
                print('\n' + log)
                log_file.write(log)
                log_file.flush()
                utils.make_plots(G_losses, D_losses, D_losses_real, D_losses_fake, D_xs, D_Gz_trainDs, D_Gz_trainGs,
                                 self.config.log_step, self.config.save_path)

                # Delete loss, output
                del d_out_real, d_loss_real, d_out_fake, d_loss_fake, g_out_fake, g_loss

            # Sample images
            if self.step % self.config.sample_step == 0:
                print("Saving image samples..")
                self.G.eval()
                fake_images = self.G(fixed_noise, fixed_labels)
                self.G.train()
                sample_images = utils.denorm(fake_images.detach()[:self.config.save_n_images])
                # Save batch images
                vutils.save_image(sample_images, os.path.join(self.config.sample_images_path, 'fake_{:05d}.png'.format(self.step)), nrow=self.config.nrow)
                # Save gif
                utils.make_gif(sample_images[0].cpu().numpy().transpose(1, 2, 0)*255, self.step,
                               self.config.sample_images_path, self.config.name, max_frames_per_gif=self.config.max_frames_per_gif)
                # Delete output
                del fake_images

            # Save model
            if self.step % self.config.model_save_step == 0:
                utils.save_ckpt(self)

    def build_models(self):
        self.G = Generator(self.config.z_dim, self.config.g_conv_dim, self.num_of_classes).to(self.device)
        self.D = Discriminator(self.config.d_conv_dim, self.num_of_classes).to(self.device)

        # Loss and optimizer
        # self.G_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.config.d_lr, [self.config.beta1, self.config.beta2])

        # Start with pretrained model (if it exists)
        if self.config.pretrained_model != '':
            utils.load_pretrained_model(self)

        if 'cuda' in self.device.type and self.config.parallel and torch.cuda.device_count() > 1:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

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
