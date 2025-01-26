import json
import torch
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
from huggingface_hub import hf_hub_download
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import os

# Create the directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)

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

# Initialize tokenizer and model
tokenizer = get_tokenizer(model_name)

model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained="checkpoints/open_clip_pytorch_model.bin",
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)

# Define possible issues and tools
issues = [
    "None",
    "eddy currents",
    "motion artifacts",
    "Field of View Cutoff",
    "Missing Slices",
    "Noise",
    "Low Resolution"
]

tool_suggestions = {
    "None": "image has no issues",
    "eddy currents": "FSL's eddy tool",
    "motion artifacts": "SPM's realignment function",
    "Field of View Cutoff": "NIfTI header editing tools like FSL's fslroi",
    "Missing Slices": "Missing Slices tool",
    "Noise": "Noise increaser LOL",
    "Low Resolution": "upsample"
}

# Define device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

context_length = 256

# Confidence threshold
confidence_threshold = 0.5

# Chatbot loop for user interaction
while True:
    user_question = input("Enter your question or type 'quit' to exit: ").strip()
    
    if user_question.lower() == 'quit':
        break
    
    image_file = input("Enter the file path to the image: ").strip()
    
    try:
        # Preprocess image
        image = preprocess(Image.open(image_file)).to(device).unsqueeze(0)
        
        # Update the template
        template = f"Question: {user_question} Answer:"
        
        # Tokenize text
        texts = tokenizer([template + " " + i for i in issues], context_length=context_length).to(device)
        
        with torch.no_grad():
            # Get image and text features
            image_features, text_features, logit_scale = model(image, texts)
            
            # Calculate logits and sort
            logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)
            
            logits = logits.cpu().numpy()
            sorted_indices = sorted_indices.cpu().numpy()
        
        # Identify issues and check confidence
        identified_issues = []
        for idx in sorted_indices[0]:
            confidence = logits[0][idx]
            if confidence >= confidence_threshold:
                identified_issues.append((issues[idx], confidence))
            else:
                break
        
        # Formulate response
        if not identified_issues:
            response = "I couldn't confidently identify any issues with the image. Here are some potential tools that might be useful:\n"
            response += "\n".join([f"- {tool}" for tool in set(tool_suggestions.values()) if tool != "image has no issues"])
        else:
            issue_descriptions = []
            for issue, confidence in identified_issues:
                if issue != "None":
                    issue_descriptions.append(f'{issue} (Confidence: {confidence:.4f})')
            if issue_descriptions:
                response = "The following potential issues were identified with the image:\n" + "\n".join(issue_descriptions)
            else:
                response = "No significant issues were identified with the image."

        if "fix" in user_question.lower() or "tool" in user_question.lower():
            if identified_issues:
                tool_list = ', '.join([tool_suggestions[i[0]] for i in identified_issues if i[0] != "None"])
                response += f"\nSuggested tools to address these issues include: {tool_list}"
            else:
                response += "\nNo tools required, the image seems to have no significant issues."
        
        print("\nResponse:\n" + response + "\n")
    
    except Exception as e:
        print(f"I apologize, but an error occurred while processing your request: {e}. Could you please check the file path and try again?")