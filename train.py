# train.py
# -*- coding: utf-8 -*-

import os
import time
import shutil
import sys
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import logging

from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from data import get_loaders
from vocab import deserialize_vocab
from model import SCAN, ContrastiveLoss
from evaluation import (
    AverageMeter,
    encode_data,
    LogCollector,
    i2t,
    t2i,
    shard_xattn
)

def setup_seed(seed):
    """Ensure reproducibility by setting random seeds."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def logging_func(log_file, message):
    """Append a message to a log file."""
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def main():
    # Debug print: confirm script launch and arguments
    print("[DEBUG] train.py started; sys.argv =", sys.argv)

    setup_seed(1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        default=r"E:\sami_bcan\data",
        help='Root path to dataset (should contain data_name subfolder)'
    )
    parser.add_argument(
        '--data_name',
        default='flickr8k_raw',
        help='Dataset name (e.g., f30k_raw or coco_raw)'
    )
    parser.add_argument(
        '--vocab_path',
        default=r"C:\Users\CGI\Downloads\BCAN-main\BCAN-main\vocab",
        help='Path to vocabulary JSON files (optional for CLIP)'
    )
    parser.add_argument('--margin', default=0.2, type=float, help='Contrastive loss margin')
    parser.add_argument('--grad_clip', default=2.0, type=float, help='Gradient clipping threshold')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--embed_size', default=512, type=int, help='Embedding size (matches CLIP’s 512)')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='Initial learning rate')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loader workers')
    parser.add_argument('--log_step', default=100, type=int, help='Log frequency (in batches)')
    parser.add_argument(
        '--logger_name',
        default='./runs/logs',
        help='Directory to save performance logs'
    )
    parser.add_argument(
        '--model_name',
        default='./runs/model',
        help='Directory to save model checkpoints'
    )
    parser.add_argument('--resume', default='', help='Path to latest checkpoint (optional)')

    parser.add_argument('--lambda_softmax', type=float, default=9.0,
                    help='lambda for softmax normalization in attention')

    opt = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    logging.info('Starting training...')

    # Load (or skip) Vocabulary
    # For the CLIP pipeline on raw images, we don't strictly need a word-index vocab.
    # But we keep this around for backward compatibility if someone runs f30k_precomp.
    if opt.data_name.endswith('_precomp'):
        vocab_file = os.path.join(opt.vocab_path, f'{opt.data_name}_vocab.json')
        print(f"[DEBUG] Loading vocabulary from {vocab_file}")
        vocab = deserialize_vocab(vocab_file)
        opt.vocab_size = len(vocab)
        print(f"[DEBUG] vocab_size = {opt.vocab_size}")
    else:
        vocab = None
        opt.vocab_size = 0
        print("[DEBUG] Skipping vocabulary load (using CLIP tokenization)")

    # Load data: train_loader, val_loader
    train_loader, val_loader = get_loaders(opt.data_name, vocab, opt.batch_size, opt.workers, opt)
    print(f"[DEBUG] train_loader and val_loader created: train_batches = {len(train_loader)}, val_batches = {len(val_loader)}")

    # Initialize SCAN model
    model = SCAN(opt)
    print("[DEBUG] SCAN model instantiated")
    model = model.to(device)
    model = nn.DataParallel(model)  # wrap for multi-GPU if available
    print("[DEBUG] Model wrapped with DataParallel")

    # Define Loss and Optimizer
    criterion = ContrastiveLoss(margin=opt.margin)
    optimizer = Adam(model.parameters(), lr=opt.learning_rate)
    print(f"[DEBUG] Optimizer created with lr = {opt.learning_rate}")

    best_rsum = 0
    start_epoch = 0

    # Resume from checkpoint if provided
    if opt.resume and os.path.isfile(opt.resume):
        checkpoint = torch.load(opt.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_rsum = checkpoint['best_rsum']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"[DEBUG] Resumed from checkpoint {opt.resume} at epoch {start_epoch}, best_rsum = {best_rsum}")

    # Training loop
    for epoch in range(start_epoch, opt.num_epochs):
        print(f"[DEBUG] --- Epoch {epoch}/{opt.num_epochs - 1} ---")
        if not os.path.exists(opt.model_name):
            os.makedirs(opt.model_name)
        if not os.path.exists(opt.logger_name):
            os.makedirs(opt.logger_name)

        log_file = os.path.join(opt.logger_name, "performance.log")
        logging_func(log_file, f"Epoch {epoch} started.")

        adjust_learning_rate(opt, optimizer, epoch)

        train_one_epoch(opt, train_loader, model, criterion, optimizer, epoch, device)

        rsum = validate(opt, val_loader, model, device)

        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(
            checkpoint,
            is_best,
            filename=f'checkpoint_{epoch}.pth.tar',
            prefix=opt.model_name + '/'
        )

def train_one_epoch(opt, train_loader, model, criterion, optimizer, epoch, device):
    """
    Train the model for one epoch.
    """
    model.train()
    start_time = time.time()

    for i, (images, captions, indices) in enumerate(train_loader):
        # images: [B, 3, 224, 224], captions: [B, 77]
        print(f"[DEBUG] Batch {i}: images.shape = {images.shape}, captions.shape = {captions.shape}")
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        optimizer.zero_grad()
        scores = model(images, captions)
        loss = criterion(scores)
        loss.backward()

        if opt.grad_clip > 0:
            clip_grad_norm_(model.parameters(), opt.grad_clip)

        optimizer.step()

        if (i + 1) % opt.log_step == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Time: {elapsed:.2f}s")
            start_time = time.time()

def validate(opt, val_loader, model, device):
    """
    Run validation: encode data and compute retrieval metrics.
    Returns the sum of recalls (r1+r5+r10 + r1i+r5i+r10i).
    """
    model.eval()
    with torch.no_grad():
        img_embs, img_means, cap_embs, cap_means = encode_data(model, val_loader, log_step=opt.log_step)

    start = time.time()
    
    # Calculate similarities
    print(f"[DEBUG] img_embs shape: {img_embs.shape}, cap_embs shape: {cap_embs.shape}")
    sims = shard_xattn(model, img_embs, img_means, cap_embs, cap_means, opt, shard_size=128)
    print(f"[DEBUG] Similarity matrix shape: {sims.shape}")
    
    end = time.time()
    print(f"[DEBUG] Validation similarity calculation time: {end - start:.2f}s")

    # Evaluate using the updated functions that return (metrics, ranks)
    (r1, r5, r10, medr, meanr), _ = i2t(img_embs, cap_embs, sims)
    print(f"Image to Text: R@1: {r1:.1f}, R@5: {r5:.1f}, R@10: {r10:.1f}, MedianR: {medr:.1f}, MeanR: {meanr:.1f}")
    
    (r1i, r5i, r10i, medri, meanri), _ = t2i(img_embs, cap_embs, sims)
    print(f"Text to Image: R@1: {r1i:.1f}, R@5: {r5i:.1f}, R@10: {r10i:.1f}, MedianR: {medri:.1f}, MeanR: {meanri:.1f}")

    # Calculate the sum of recalls
    rsum = r1 + r5 + r10 + r1i + r5i + r10i
    print(f"rSum: {rsum:.1f}")
    
    return rsum


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    """
    Save model state to disk; if is_best=True, also copy to model_best.pth.tar.
    """
    torch.save(state, prefix + filename)
    if is_best:
        print(f"[DEBUG] Best model saved at epoch {state['epoch']}")
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')

def adjust_learning_rate(opt, optimizer, epoch):
    """
    Decay learning rate by factor 0.1 every 15 epochs.
    """
    lr = opt.learning_rate * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
