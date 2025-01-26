# Alternative Dataloader Usage

Proposed an alternative approach to dataloading in [finetune.py script](../nancy_newlin/finetune.py)

## Modified the dataloader
```python
from duong_mai.dataloader import BiomedCLIPDataset, collate_fn

dataset = BiomedCLIPDataset(
    json_path="../duong_mai/BiomedCLIP_sandbox_dataset.json",
    tokenizer=tokenizer,
    preprocess=preprocess
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)
```

## Modified the training loop as follows

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch["images"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        targets = batch["targets"].to(device)
        
        optimizer.zero_grad()
        image_features, text_features, logit_scale = model(images, text_tokens)
        loss = loss_func(logit_scale * image_features, text_features, targets)
        loss.backward()
        optimizer.step()
```

## A comprehensive alternative to model training

```python
import json
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from transformers import AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from duong_mai.dataloader import BiomedCLIPDataset, collate_fn


# Load model and config
model_name = "biomedclip_local"
with open("checkpoints/open_clip_config.json", "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]

if (not model_name.startswith(HF_HUB_PREFIX) 
    and model_name not in _MODEL_CONFIGS 
    and config is not None):
    _MODEL_CONFIGS[model_name] = model_cfg

tokenizer = get_tokenizer(model_name)
model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained="checkpoints/open_clip_pytorch_model.bin",
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)

# Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_func = torch.nn.CosineEmbeddingLoss()

# create dataloader
dataset = BiomedCLIPDataset(
    json_path="../duong_mai/BiomedCLIP_sandbox_dataset.json",
    tokenizer=tokenizer,
    preprocess=preprocess
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

# Training loop
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    print(f"EPOCH: {epoch}")
    for batch in dataloader:
        images = batch["images"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        targets = batch["targets"].to(device)
        
        optimizer.zero_grad()
        image_features, text_features, logit_scale = model(images, text_tokens)
        loss = loss_func(logit_scale * image_features, text_features, targets)
        print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

print("Done~")
```