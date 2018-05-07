#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Developed by Haozhe Xie <cshzxie@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils

from datetime import datetime as dt
from tensorboardX import SummaryWriter
from time import time

from core.test import test_net
from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner

def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark  = True

    # Set up data augmentation
    IMG_SIZE  = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = cfg.TRAIN.CROP_IMG_H, cfg.TRAIN.CROP_IMG_W
    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.RandomCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TRAIN.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ColorJitter(cfg.TRAIN.BRIGHTNESS, cfg.TRAIN.CONTRAST, cfg.TRAIN.SATURATION, cfg.TRAIN.HUE),
        utils.data_transforms.ToTensor(),
    ])
    val_transforms  = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
        utils.data_transforms.ToTensor(),
    ])
    
    # Set up data loader
    dataset_loader    = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.DATASET_NAME](cfg)
    n_views           = np.random.randint(cfg.CONST.N_VIEWS) + 1 if cfg.TRAIN.RANDOM_NUM_VIEWS else cfg.CONST.N_VIEWS
    train_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(cfg.TRAIN.DATASET_PORTION, n_views, train_transforms),
        batch_size=cfg.CONST.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKER, pin_memory=True, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_loader.get_dataset(cfg.TEST.DATASET_PORTION, n_views, val_transforms),
        batch_size=1,
        num_workers=1, pin_memory=True, shuffle=False)

    # Summary writer for TensorBoard
    output_dir   = os.path.join(cfg.DIR.OUT_PATH, '%s', dt.now().isoformat())
    log_dir      = output_dir % 'logs'
    img_dir      = output_dir % 'images'
    ckpt_dir     = output_dir % 'checkpoints'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    val_writer   = SummaryWriter(os.path.join(log_dir, 'test'))

    # Set up networks
    encoder      = Encoder(cfg)
    decoder      = Decoder(cfg)
    refiner      = Refiner(cfg)

    # Initialize weights of networks
    encoder.apply(utils.network_utils.init_weights)
    decoder.apply(utils.network_utils.init_weights)
    refiner.apply(utils.network_utils.init_weights)

    # Set up solver
    decoder_solver = None
    encoder_solver = None
    refiner_solver = None
    if cfg.TRAIN.POLICY == 'adam':
        encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        decoder_solver = torch.optim.Adam(decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
        refiner_solver = torch.optim.Adam(refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, betas=cfg.TRAIN.BETAS)
    elif cfg.TRAIN.POLICY == 'sgd':
        encoder_solver = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()), lr=cfg.TRAIN.ENCODER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        decoder_solver = torch.optim.SGD(decoder.parameters(), lr=cfg.TRAIN.DECODER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
        refiner_solver = torch.optim.SGD(refiner.parameters(), lr=cfg.TRAIN.REFINER_LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise Exception('[FATAL] %s Unknown optimizer %s.' % (dt.now(), cfg.TRAIN.POLICY))

    # Set up learning rate scheduler to decay learning rates dynamically
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver, milestones=cfg.TRAIN.ENCODER_LR_MILESTONES, gamma=0.1)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver, milestones=cfg.TRAIN.DECODER_LR_MILESTONES, gamma=0.1)
    refiner_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(refiner_solver, milestones=cfg.TRAIN.REFINER_LR_MILESTONES, gamma=0.1)

    if torch.cuda.is_available():
        torch.nn.DataParallel(encoder).cuda()
        torch.nn.DataParallel(decoder).cuda()
        torch.nn.DataParallel(refiner).cuda()

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Load pretrained model if exists
    init_epoch     = 0
    best_iou       = -1
    best_epoch     = -1
    if 'WEIGHTS' in cfg.CONST and cfg.TRAIN.RESUME_TRAIN:
        print('[INFO] %s Recovering from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        init_epoch = checkpoint['epoch_idx']
        best_iou   = checkpoint['best_iou']
        best_epoch = checkpoint['best_epoch']

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder_solver.load_state_dict(checkpoint['encoder_solver_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder_solver.load_state_dict(checkpoint['decoder_solver_state_dict'])
        refiner.load_state_dict(checkpoint['refiner_state_dict'])
        refiner_solver.load_state_dict(checkpoint['refiner_solver_state_dict'])
        
        print('[INFO] %s Recover complete. Current epoch #%d, Best IoU = %.4f at epoch #%d.' \
                 % (dt.now(), init_epoch, best_iou, best_epoch))

    # Training loop
    for epoch_idx in range(init_epoch, cfg.TRAIN.NUM_EPOCHES):
        # Tick / tock
        epoch_start_time = time()
        
        # Average meterics
        epoch_encoder_loss = []
        epoch_refiner_loss = []

        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_names, sample_names, rendering_images, voxels) in enumerate(train_data_loader):
            n_samples = len(voxels)
            # Ignore imcomplete batches at the end of each epoch
            if not n_samples == cfg.CONST.BATCH_SIZE:
                continue

            # Tick / tock
            batch_start_time = time()

            # switch models to training mode
            encoder.train();
            decoder.train();

            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            voxels           = utils.network_utils.var_or_cuda(voxels)

            # Train the encoder, decoder and refiner
            image_features, raw_features    = encoder(rendering_images)
            generated_voxels                = decoder(image_features)
            encoder_loss                    = bce_loss(generated_voxels, voxels) * 10

            generated_voxels                = refiner(generated_voxels, raw_features)
            refiner_loss                    = bce_loss(generated_voxels, voxels) * 10

            # Gradient decent
            encoder.zero_grad()
            decoder.zero_grad()
            refiner.zero_grad()
            encoder_loss.backward(retain_graph=True)
            refiner_loss.backward()
            encoder_solver.step()
            decoder_solver.step()
            refiner_solver.step()
            
            # Append loss to average metrics
            epoch_encoder_loss.append(encoder_loss.item())
            epoch_refiner_loss.append(refiner_loss.item())
            # Append loss to TensorBoard
            n_itr = epoch_idx * n_batches + batch_idx
            train_writer.add_scalar('EncoderDecoder/BatchLoss', encoder_loss.item(), n_itr)
            train_writer.add_scalar('Refiner/BatchLoss', encoder_loss.item(), n_itr)
            # Append rendering images of voxels to TensorBoard
            if n_itr % cfg.TRAIN.VISUALIZATION_FREQ == 0:
                # TODO: add GT here ...
                gv           = generated_voxels.cpu().data[:8].numpy()
                voxel_views  = utils.binvox_visualization.get_voxel_views(gv, os.path.join(img_dir, 'train'), n_itr)
                train_writer.add_image('Reconstructed Voxels', voxel_views, n_itr)

            # Tick / tock
            batch_end_time = time()
            print('[INFO] %s [Epoch %d/%d][Batch %d/%d] Total Time = %.3f (s) EDLoss = %.4f RLoss = %.4f' % \
                (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, batch_idx + 1, n_batches, \
                    batch_end_time - batch_start_time, encoder_loss.item(), refiner_loss.item()))

        # Append epoch loss to TensorBoard
        encoder_mean_loss = np.mean(epoch_encoder_loss)
        refiner_mean_loss = np.mean(epoch_refiner_loss)
        train_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_mean_loss, epoch_idx + 1)
        train_writer.add_scalar('Refiner/EpochLoss', refiner_mean_loss, epoch_idx + 1)

        # Tick / tock
        epoch_end_time = time()
        print('[INFO] %s Epoch [%d/%d] Total Time = %.3f (s) EDLoss = %.4f RLoss = %.4f' % 
            (dt.now(), epoch_idx + 1, cfg.TRAIN.NUM_EPOCHES, epoch_end_time - epoch_start_time, \
                encoder_mean_loss, refiner_mean_loss))

        # Validate the training models
        iou = test_net(cfg, epoch_idx + 1, output_dir, val_data_loader, val_writer, encoder, decoder, refiner)

        # Save weights to file
        if (epoch_idx + 1) % cfg.TRAIN.SAVE_FREQ == 0:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            
            utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'ckpt-epoch-%04d.pth.tar' % (epoch_idx + 1)), \
                    epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver, \
                    refiner, refiner_solver, best_iou, best_epoch)
        elif iou > best_iou:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            
            best_iou   = iou
            best_epoch = epoch_idx + 1
            utils.network_utils.save_checkpoints(os.path.join(ckpt_dir, 'best-ckpt.pth.tar'), \
                    epoch_idx + 1, encoder, encoder_solver, decoder, decoder_solver, \
                    refiner, refiner_solver, best_iou, best_epoch)

    # Close SummaryWriter for TensorBoard
    train_writer.close()
    val_writer.close()

