import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import clip

class DataLoaderX(DataLoader):
    """A DataLoader that uses background prefetching to speed up data loading.""" 
    def __iter__(self):
        from prefetch_generator import BackgroundGenerator
        return BackgroundGenerator(super().__iter__())

class CLIPDataset(data.Dataset):
    """
    Dataset for loading raw images and tokenizing captions via CLIP.
    
    Expects directory structure:
      data_path/
        train_caps.txt
        dev_caps.txt
        train/
          images/
        dev/
          images/
    """
    def __init__(self, data_path, split, vocab=None):
        super(CLIPDataset, self).__init__()
        self.vocab = vocab

        # Load captions
        caps_file = os.path.join(data_path, f"{split}_caps.txt")
        if not os.path.isfile(caps_file):
            raise FileNotFoundError(f"Caption file not found: {caps_file}")
        self.captions = []
        with open(caps_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.captions.append(line.strip())

        # Prepare image directory and list
        self.image_dir = os.path.join(data_path, split, "images")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        all_files = sorted(os.listdir(self.image_dir))
        self.image_files = [
            fname for fname in all_files
            if fname.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(self.image_files) != len(self.captions):
            raise ValueError(f"Number of images ({len(self.image_files)}) != number of captions ({len(self.captions)})")

        # CLIP preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ])

        # Load CLIP model just for tokenizer
        try:
            self.clip_model, _ = clip.load("ViT-B/16", device="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # stays on CPU

        caption = self.captions[index]
        tokenized = clip.tokenize(caption)[0]  # Remove [0] to keep batch shape if needed
        return image, tokenized, index

def collate_fn(data):
    images, captions, indices = zip(*data)
    images = torch.stack(images, dim=0)      # [B, 3, 224, 224]
    captions = torch.stack(captions, dim=0)  # [B, 77]
    return images, captions, indices

def get_clip_loader(data_path, split, vocab, batch_size=64, shuffle=True, num_workers=4):
    dataset = CLIPDataset(data_path, split, vocab)
    loader = DataLoaderX(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return loader

def get_loaders(data_name, vocab, batch_size, workers, opt):
    root = os.path.join(opt.data_path, data_name)
    train_loader = get_clip_loader(root, 'train', vocab, batch_size, True, workers)
    val_loader   = get_clip_loader(root, 'dev',   vocab, batch_size, False, workers)
    return train_loader, val_loader

def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    root = os.path.join(opt.data_path, data_name)
    return get_clip_loader(root, split_name, vocab, batch_size, False, workers)
