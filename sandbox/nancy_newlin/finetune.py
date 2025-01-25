import json
import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from transformers import AdamW
import torch.nn as nn

"""
# Download the model and config files
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_pytorch_model.bin",
    local_dir="checkpoints"
)
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_config.json",
    local_dir="checkpoints"
)
"""

# Load the model and config files
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


# Zero-shot image classification
template = 'this is a photo of '

template = 'this image has the following issues'
labels = [
    'FOV cut off at top of brain',
    'Low Resolution'
]

dataset_url = 'https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/'
train_imgs = [
    'SLICE_sub-8510001B_view-sag_group-FOV_sample-4.png',
    'SLICE_sub-8471001A_view-sag_group-None_sample-17.png'
]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
#model.eval()
#loss_fct = nn.C
optimizer = AdamW(model.parameters(), lr=1e-5)
context_length = 256

images = torch.stack([preprocess(Image.open(img)) for img in train_imgs]).to(device)
texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)
num_epochs = 10
model.train()
optimizer.zero_grad()
loss_func  = torch.nn.CosineEmbeddingLoss() #torch.nn.CrossEntropyLoss()
third = torch.ones(2).to(device)

for epoch in range(num_epochs):
	print("EPOCH:",epoch)
	#print(texts.shape)
	#print(images.shape)
	#for image, text in images, texts:
	#labels = torch.tensor(texts, dtype=torch.long).view(context_length,-1).to(device)
	image_features, text_features, logit_scale = model(images, texts)
	#print(f"logit_scale:",logit_scale * image_features)
	loss = loss_func(logit_scale * image_features, text_features, third)
	print(loss)
	#print("Logit scale", logit_scale)
	#loss = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
	#print(f"Loss: {loss.shape} {loss}, Text Features: {text_features}, image features: {image_features}")
	#loss = loss_fct(logits, labels)
	print(loss.item())
	loss.backward()
	optimizer.step()


print("Done~")
exit()
with torch.no_grad():
    image_features, text_features, logit_scale = model(images, texts)

    logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()

top_k = -1

print("Done.")
exit()

for i, img in enumerate(test_imgs):
    pred = labels[sorted_indices[i][0]]

    top_k = len(labels) if top_k == -1 else top_k
    print(img.split('/')[-1] + ':')
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        print(f'{labels[jth_index]}: {logits[i][jth_index]}')
    print('\n')


def plot_images_with_metadata(images, metadata):
    num_images = len(images)
    fig, axes = plt.subplots(nrows=num_images, ncols=1, figsize=(5, 5 * num_images))

    for i, (img_path, metadata) in enumerate(zip(images, metadata)):
        img = Image.open(img_path)
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{metadata['filename']}\n{metadata['top_probs']}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f'Output_{i}.png')

metadata_list = []

top_k = 1
for i, img in enumerate(test_imgs):
    pred = labels[sorted_indices[i][0]]
    img_name = img.split('/')[-1]

    top_probs = []
    top_k = len(labels) if top_k == -1 else top_k
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        top_probs.append(f"{labels[jth_index]}: {logits[i][jth_index] * 100:.1f}")

    metadata = {'filename': img_name, 'top_probs': '\n'.join(top_probs)}
    metadata_list.append(metadata)

plot_images_with_metadata(test_imgs, metadata_list)
