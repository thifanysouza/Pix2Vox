# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
# Modified

import json
import numpy as np
import logging
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.data_loaders
import utils.data_transforms
import utils.helpers

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger
from utils.average_meter import AverageMeter

def remove_module_prefix(state_dict): #Remoção do pré-fixo module. ao carregar os pesos. Evita problemas de chave, pois estamos executando na 'cpu'
    """Remove 'module.' prefix from keys in state_dict, if present."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def test_net(cfg,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Load taxonomies of dataset
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
            batch_size=1,
            num_workers=cfg.CONST.NUM_WORKER,
            pin_memory=True,
            shuffle=False)

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger = Merger(cfg)

        if torch.cuda.is_available():
            device = torch.device('cuda')
            encoder = torch.nn.DataParallel(encoder).to(device)
            decoder = torch.nn.DataParallel(decoder).to(device)
            refiner = torch.nn.DataParallel(refiner).to(device)
            merger = torch.nn.DataParallel(merger).to(device)
            logging.info('Usando GPU para execução.')
        else:
            device = torch.device('cpu')
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            refiner = refiner.to(device)
            merger = merger.to(device)
            logging.info('CPU device') #Informando ao usuário que estamos utilizando cpu

        logging.info('Carregando pesos do arquivo: %s ...' % cfg.CONST.WEIGHTS)
        checkpoint = torch.load(cfg.CONST.WEIGHTS, map_location=torch.device('cpu'))
        epoch_idx = checkpoint['epoch_idx']

        encoder_state_dict = remove_module_prefix(checkpoint['encoder_state_dict'])
        decoder_state_dict = remove_module_prefix(checkpoint['decoder_state_dict'])

        try:
            encoder.load_state_dict(encoder_state_dict)
        except RuntimeError as e:
            logging.error(f"State_dict to encoder failure: {e}")
            raise

        try:
            decoder.load_state_dict(decoder_state_dict)
        except RuntimeError as e:
            logging.error(f"State_dict to decoder failure: {e}")
            raise

        if cfg.NETWORK.USE_REFINER:
            refiner_state_dict = remove_module_prefix(checkpoint['refiner_state_dict'])
            refiner.load_state_dict(refiner_state_dict)
        if cfg.NETWORK.USE_MERGER:
            merger_state_dict = remove_module_prefix(checkpoint['merger_state_dict'])
            merger.load_state_dict(merger_state_dict)

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = AverageMeter()
    refiner_losses = AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        with torch.no_grad():
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)

            image_features = encoder(rendering_images)
            raw_features, generated_volume = decoder(image_features)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)

            encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refiner(generated_volume)
                refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10
            else:
                refiner_loss = encoder_loss

            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            if test_writer and sample_idx < 3:
                rendering_views = utils.helpers.get_volume_views(generated_volume.cpu().numpy())
                test_writer.add_image('Model%02d/Reconstructed' % sample_idx, rendering_views, epoch_idx)
                rendering_views = utils.helpers.get_volume_views(ground_truth_volume.cpu().numpy())
                test_writer.add_image('Model%02d/GroundTruth' % sample_idx, rendering_views, epoch_idx)

            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
                         (sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
                          refiner_loss.item(), ['%.4f' % si for si in sample_iou]))

    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()

    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()

    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

    logging.info('Tests finished')

    return max_iou

