import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class BiomedCLIPDataset(Dataset):
    def __init__(self, json_path, tokenizer, preprocess, template="this image has the following issues: ", context_length=256):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.template = template
        self.context_length = context_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"])
        image_tensor = self.preprocess(image)
        
        # Handle multiple labels if present
        if isinstance(item["labels"], list):
            labels = item["labels"]
        else:
            labels = [item["labels"]]
            
        # Tokenize all labels with template
        text_tokens = self.tokenizer(
            [self.template + label for label in labels],
            context_length=self.context_length
        )
        
        return {
            "image": image_tensor,
            "text": text_tokens,
            "labels": labels
        }

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    # Concatenate all text tokens
    text_tokens = torch.cat([item["text"] for item in batch])
    # Create target tensor for cosine embedding loss
    targets = torch.ones(len(batch))
    
    return {
        "images": images,
        "text_tokens": text_tokens,
        "targets": targets
    }
