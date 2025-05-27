# evaluation.py

import os
import torch
import numpy as np
import time
from collections import OrderedDict
from torch.autograd import Variable
from data import get_test_loader # Assuming this correctly handles your data
from model import SCAN # Assuming your model definition is consistent
from vocab import deserialize_vocab

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

class LogCollector(object):
    """A collection of logging objects that can switch from train to eval mode"""
    def __init__(self):
        self.meters = OrderedDict()

    def update(self, k, v, n=1):
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        return ' '.join([f"{k}: {v}" for k, v in self.meters.items()])

def encode_data(model, data_loader, log_step=10):
    batch_time = AverageMeter()
    val_logger = LogCollector() # This isn't currently used in encode_data, but kept for consistency

    model.eval()
    end = time.time()

    img_embs, img_means = [], []
    cap_embs, cap_means = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # Ensure batch unpacking matches what your data loader provides
            # If your data loader also gives IDs or paths, keep *_
            images, captions, *_ = batch
            
            # Ensure data is moved to the correct device (cuda if available)
            images = images.cuda() if torch.cuda.is_available() else images
            captions = captions.cuda() if torch.cuda.is_available() else captions

            forward_emb = model.module.forward_emb if hasattr(model, 'module') else model.forward_emb
            img_emb, img_mean, cap_emb, cap_mean = forward_emb(images, captions)

            img_embs.append(img_emb.cpu().numpy())
            img_means.append(img_mean.cpu().numpy())
            cap_embs.append(cap_emb.cpu().numpy())
            cap_means.append(cap_mean.cpu().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                print(f"Test [{i}/{len(data_loader)}] Time: {batch_time.avg:.3f}")

    img_embs = np.vstack(img_embs)
    img_means = np.vstack(img_means)
    cap_embs = np.vstack(cap_embs)
    cap_means = np.vstack(cap_means)

    return img_embs, img_means, cap_embs, cap_means

def evalrank(model_path, data_path=None, split='test'):
    """
    Evaluate a trained model on Flickr30K or other datasets.
    """
    checkpoint = torch.load(model_path, map_location='cpu') # Map to CPU first, then model to GPU
    opt = checkpoint['opt']
    if data_path:
        opt.data_path = data_path

    # Check if vocab is actually needed for the data_name
    if opt.data_name.endswith('_precomp'):
        vocab = deserialize_vocab(os.path.join(opt.vocab_path, f"{opt.data_name}_vocab.json"))
        opt.vocab_size = len(vocab)
    else:
        vocab = None
        opt.vocab_size = 0 # No vocab size needed for raw data (CLIP tokenization)
        print(f"[evalrank] Skipping vocabulary load for {opt.data_name}")


    model = SCAN(opt)
    model.load_state_dict(checkpoint['model'])
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    print("Loading test dataset...")
    # Pass opt to get_test_loader if it needs it for data_name or other args
    data_loader = get_test_loader(split, opt.data_name, vocab, opt.batch_size, 0, opt) 

    print("Encoding data...")
    # encode_data now returns 4 values, but evalrank's call only expects 2.
    # This is a potential mismatch between evalrank's usage and train.py's usage.
    # For evalrank's purpose, it typically uses only the embeddings, not the means directly.
    # Let's adjust this to correctly unpack the results from encode_data.
    img_embs, img_means, cap_embs, cap_means = encode_data(model, data_loader)


    print(f"Images: {img_embs.shape[0]}, Captions: {cap_embs.shape[0]}")

    print("Computing similarity scores...")
    # Use shard_xattn which uses means if your model computes them for similarity
    sims = shard_xattn(model, img_embs, img_means, cap_embs, cap_means, opt)


    print("Evaluating retrieval performance...")
    # The return values of i2t and t2i are (metrics_tuple, ranks_tuple)
    (r1_i, r5_i, r10_i, medr_i, meanr_i), i2t_ranks = i2t(img_embs, cap_embs, sims)
    (r1_t, r5_t, r10_t, medr_t, meanr_t), t2i_ranks = t2i(img_embs, cap_embs, sims)

    # Use the unpacked recall values for average calculation
    ar = (r1_i + r5_i + r10_i) / 3
    ari = (r1_t + r5_t + r10_t) / 3
    rsum = r1_i + r5_i + r10_i + r1_t + r5_t + r10_t

    print(f"rsum: {rsum:.1f}")
    print(f"Average Image-to-Text Recall: {ar:.1f}")
    print(f"Image to Text: R@1:{r1_i:.1f}, R@5:{r5_i:.1f}, R@10:{r10_i:.1f}, MedR:{medr_i:.1f}, MeanR:{meanr_i:.1f}")
    print(f"Average Text-to-Image Recall: {ari:.1f}")
    print(f"Text to Image: R@1:{r1_t:.1f}, R@5:{r5_t:.1f}, R@10:{r10_t:.1f}, MedR:{medr_t:.1f}, MeanR:{meanr_t:.1f}")

def compute_similarity(img_embs, cap_embs, opt, shard_size=500):
    """
    NOTE: This function is replaced by shard_xattn for SCAN models.
    It's only for compatibility with older models that don't use cross-attention means.
    You are likely using shard_xattn from train.py, which is correct for SCAN.
    Leaving this here but flagging it.
    """
    print("Warning: compute_similarity is typically used for non-attention models. "
          "For SCAN, ensure shard_xattn is used consistently.")
    n_img, n_cap = img_embs.shape[0], cap_embs.shape[0]
    d = np.zeros((n_img, n_cap))

    for i in range(0, n_img, shard_size):
        im_start, im_end = i, min(i + shard_size, n_img)
        for j in range(0, n_cap, shard_size):
            cap_start, cap_end = j, min(j + shard_size, n_cap)
            im = Variable(torch.from_numpy(img_embs[im_start:im_end])).float().cuda()
            cap = Variable(torch.from_numpy(cap_embs[cap_start:cap_end])).float().cuda()

            with torch.no_grad():
                sim = im.mm(cap.t())

            d[im_start:im_end, cap_start:cap_end] = sim.cpu().numpy()

    return d

def i2t(images, captions, sims):
    """
    Image-to-Text Retrieval Evaluation
    Returns:
        - tuple of (r1, r5, r10, medr, meanr): Recall metrics and median/mean rank
        - tuple of (ranks, top1_array): Rank position of ground truth for each query and top1 hits
    """
    npts = images.shape[0] # Number of images
    ranks = np.zeros(npts)
    top1 = np.zeros(npts) # Array to store 1 if R@1 hit, 0 otherwise

    # Assume 5 captions per image for Flickr8k (standard for this dataset)
    # Adjust `caps_per_image` if you're using a different dataset or setup
    caps_per_image = 5 
    n_captions = captions.shape[0] # Total number of captions

    for i in range(npts): # Loop through each image query
        # Calculate the indices of the 5 ground truth captions for the current image
        # These are usually contiguous: i*5, i*5+1, i*5+2, i*5+3, i*5+4
        gt_caption_indices = np.arange(i * caps_per_image, (i + 1) * caps_per_image)

        # Check if calculated ground truth indices are within valid bounds
        if np.max(gt_caption_indices) >= n_captions:
            print(f"[WARNING_i2t] Image {i}: Calculated ground truth caption index "
                  f"{np.max(gt_caption_indices)} is out of bounds for total captions {n_captions}.")
            # This indicates an issue with data loading or `caps_per_image` assumption.
            # Assigning a worst possible rank to not break the evaluation
            ranks[i] = n_captions - 1
            top1[i] = 0
            continue # Skip to the next image

        # Get similarity scores for the current image against all captions
        # `sims` shape is (N_images, N_captions)
        sim_scores_for_image = sims[i, :]

        # Sort captions by similarity score in descending order
        # `sorted_indices` will contain indices of captions, ordered by relevance
        sorted_indices = np.argsort(sim_scores_for_image)[::-1]

        # Find the rank of the *first* correct caption among the sorted_indices
        for rank, c_idx in enumerate(sorted_indices):
            if c_idx in gt_caption_indices: # Check if the current ranked caption is one of the ground truths
                ranks[i] = rank # Store the rank (0-indexed)
                break
        
        # Determine if R@1 is achieved (correct caption is at rank 0)
        if ranks[i] < 1: # if rank is 0
            top1[i] = 1

    # Compute recall metrics based on collected ranks
    r1 = 100.0 * np.sum(ranks < 1) / npts
    r5 = 100.0 * np.sum(ranks < 5) / npts
    r10 = 100.0 * np.sum(ranks < 10) / npts
    medr = np.floor(np.median(ranks)) + 1 # +1 because ranks are 0-indexed
    meanr = ranks.mean() + 1 # +1 because ranks are 0-indexed

    # Return the metrics tuple and the ranks/top1 tuple
    return (r1, r5, r10, medr, meanr), (ranks, top1) # CORRECTED RETURN VALUE


def t2i(images, captions, sims):
    """
    Text-to-Image Retrieval Evaluation
    Returns:
        - tuple of (r1, r5, r10, medr, meanr): Recall metrics and median/mean rank
        - tuple of (ranks, top1_array): Rank position of ground truth for each query and top1 hits
    """
    n_images = images.shape[0] # Total number of images
    n_captions = captions.shape[0] # Total number of captions
    
    ranks = np.zeros(n_captions) # Ranks for each caption query
    top1 = np.zeros(n_captions) # Array to store 1 if R@1 hit, 0 otherwise

    # Transpose similarity matrix for text-to-image retrieval: (N_captions, N_images)
    # The original `sims` is (N_images, N_captions)
    sims_t = sims.T 
    
    # Assume 5 captions per image (standard for Flickr8k)
    caps_per_image = 5

    for i in range(n_captions): # Loop through each caption query
        # Get the ground truth image index for the current caption
        # If captions are ordered as (img0_cap0, img0_cap1, ..., img1_cap0, ...),
        # then caption `i` corresponds to image `i // caps_per_image`
        gt_image_index = i // caps_per_image

        # Check if the calculated ground truth image index is within valid bounds
        if gt_image_index >= n_images:
            print(f"[WARNING_t2i] Caption {i}: Calculated ground truth image index "
                  f"{gt_image_index} is out of bounds for total images {n_images}.")
            # Assigning a worst possible rank
            ranks[i] = n_images - 1
            top1[i] = 0
            continue # Skip to the next caption

        # Get similarity scores for the current caption against all images
        # `sims_t` shape is (N_captions, N_images)
        sim_scores_for_caption = sims_t[i, :]

        # Sort images by similarity score in descending order
        # `sorted_indices` will contain indices of images, ordered by relevance
        sorted_indices = np.argsort(sim_scores_for_caption)[::-1]

        # Find the rank of the *first* correct image among the sorted_indices
        for rank, im_idx in enumerate(sorted_indices):
            if im_idx == gt_image_index: # Check if the current ranked image is the ground truth
                ranks[i] = rank # Store the rank (0-indexed)
                break

        # Determine if R@1 is achieved (correct image is at rank 0)
        if ranks[i] < 1: # if rank is 0
            top1[i] = 1

    # Compute recall metrics based on collected ranks
    r1 = 100.0 * np.sum(ranks < 1) / n_captions
    r5 = 100.0 * np.sum(ranks < 5) / n_captions
    r10 = 100.0 * np.sum(ranks < 10) / n_captions
    medr = np.floor(np.median(ranks)) + 1 # +1 because ranks are 0-indexed
    meanr = ranks.mean() + 1 # +1 because ranks are 0-indexed

    # Return the metrics tuple and the ranks/top1 tuple
    return (r1, r5, r10, medr, meanr), (ranks, top1) # CORRECTED RETURN VALUE

def shard_xattn(model, images, img_means, captions, cap_means, opt, shard_size=128):
    """
    Compute pairwise image-text similarity with memory-efficient sharding using cross-attention.
    This function is specific to SCAN-like models that utilize mean representations.
    """
    n_im = len(images)
    n_cap = len(captions)
    
    # Calculate number of shards
    n_im_shard = (n_im - 1) // shard_size + 1
    n_cap_shard = (n_cap - 1) // shard_size + 1

    d = np.zeros((n_im, n_cap))

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), n_im)
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), n_cap)

            # Convert NumPy arrays to PyTorch tensors and move to CUDA if available
            im = torch.from_numpy(images[im_start:im_end]).float().cuda() if torch.cuda.is_available() else torch.from_numpy(images[im_start:im_end]).float()
            im_mean = torch.from_numpy(img_means[im_start:im_end]).float().cuda() if torch.cuda.is_available() else torch.from_numpy(img_means[im_start:im_end]).float()
            cap = torch.from_numpy(captions[cap_start:cap_end]).float().cuda() if torch.cuda.is_available() else torch.from_numpy(captions[cap_start:cap_end]).float()
            cap_mean = torch.from_numpy(cap_means[cap_start:cap_end]).float().cuda() if torch.cuda.is_available() else torch.from_numpy(cap_means[cap_start:cap_end]).float()

            with torch.no_grad():
                # If model is wrapped by DataParallel, use model.module
                forward_sim = model.module.forward_sim if hasattr(model, 'module') else model.forward_sim
                sim = forward_sim(im, im_mean, cap, cap_mean)

            d[im_start:im_end, cap_start:cap_end] = sim.cpu().numpy()

    return d


if __name__ == '__main__':
    model_path = "./runs/model/model_best.pth.tar" # Adjusted default path
    # Ensure this path exists or provide the correct one
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}. Please provide a valid path.")
    else:
        evalrank(model_path)