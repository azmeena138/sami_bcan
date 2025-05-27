# model.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import clip

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X."""
    norm = torch.norm(X, p=2, dim=dim, keepdim=True) + eps
    return X / norm

class EncoderImageCLIP(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        print("[DEBUG] Initializing EncoderImageCLIP")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
        self.fc = nn.Linear(512, embed_size)

    def forward(self, images):
        print(f"[DEBUG] EncoderImageCLIP.forward called; images.shape = {images.shape}")
        images = images.to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(images).float()  # [B, 512]
        projected = self.fc(features)  # [B, embed_size]
        img_emb = l2norm(projected, dim=-1)
        return img_emb.unsqueeze(1), img_emb  # [B, 1, D], [B, D]

class EncoderTextCLIP(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        print("[DEBUG] Initializing EncoderTextCLIP")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
        self.fc = nn.Linear(512, embed_size)

    def forward(self, captions):
        print(f"[DEBUG] EncoderTextCLIP.forward called; captions.shape = {captions.shape}")
        tokens = captions.to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens).float()  # [B, 512]
        projected = self.fc(features)
        cap_emb = l2norm(projected, dim=-1)
        return cap_emb.unsqueeze(1), cap_emb  # [B, 1, D], [B, D]

def func_attention(query, context, g_sim, opt, eps=1e-8):
    """
    query: [B, Q, D], context: [B, C, D]
    """
    queryT = query.transpose(1, 2)            # [B, D, Q]
    attn = torch.bmm(context, queryT)          # [B, C, Q]
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, dim=1)                 # [B, C, Q]
    attn = attn.permute(0, 2, 1)               # [B, Q, C]
    attn = nn.Softmax(dim=2)(attn * opt.lambda_softmax)
    re_attn = g_sim.unsqueeze(1).unsqueeze(2) * attn
    attn_sum = re_attn.sum(dim=-1, keepdim=True)
    re_attn = re_attn / (attn_sum + eps)
    weighted_context = torch.bmm(re_attn, context)  # [B, Q, D]
    return weighted_context, re_attn

class SCAN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        print("[DEBUG] Initializing SCAN")
        self.img_enc = EncoderImageCLIP(opt.embed_size)
        self.txt_enc = EncoderTextCLIP(opt.embed_size)
        self.opt = opt

    def forward_emb(self, images, captions):
        img_emb_seq, img_mean = self.img_enc(images)   # [B, 1, D], [B, D]
        cap_emb_seq, cap_mean = self.txt_enc(captions) # [B, 1, D], [B, D]
        return img_emb_seq, img_mean, cap_emb_seq, cap_mean

    def forward_sim(self, img_emb_seq, img_mean, cap_emb_seq, cap_mean):
        g_sims = cap_mean.mm(img_mean.t())  # [B, B]
        similarities = []

        for i in range(cap_emb_seq.size(0)):
            g_sim = g_sims[i]  # [B]
            cap_i = cap_emb_seq[i].unsqueeze(0).expand(img_emb_seq.size(0), -1, -1)  # [B, 1, D]
            weiContext, _ = func_attention(cap_i, img_emb_seq, g_sim, self.opt)
            t2i_sim = (cap_i * weiContext).sum(dim=2).mean(dim=1, keepdim=True)

            img_i = img_emb_seq  # already [B, 1, D]
            cap_j = cap_emb_seq[i].unsqueeze(0).expand(img_emb_seq.size(0), -1, -1)  # [B, 1, D]
            weiContext, _ = func_attention(img_i, cap_j, g_sim, self.opt)
            i2t_sim = (img_i * weiContext).sum(dim=2).mean(dim=1, keepdim=True)

            sim = t2i_sim + i2t_sim  # [B, 1]
            similarities.append(sim)

        return torch.cat(similarities, dim=1)  # [B, B]

    def forward(self, images, captions):
        img_seq, img_mean, cap_seq, cap_mean = self.forward_emb(images, captions)
        return self.forward_sim(img_seq, img_mean, cap_seq, cap_mean)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        print(f"[DEBUG] Initializing ContrastiveLoss with margin = {margin}")
        self.margin = margin

    def forward(self, scores):
        diagonal = scores.diag().view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0), device=scores.device).bool()
        cost_s = cost_s.masked_fill(mask, 0)
        cost_im = cost_im.masked_fill(mask, 0)

        cost_s = cost_s.max(dim=1)[0]
        cost_im = cost_im.max(dim=0)[0]

        return cost_s.sum() + cost_im.sum()
