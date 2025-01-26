import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QSplitter,
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from datetime import datetime
import torch
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
import json
import os
import torch
from huggingface_hub import hf_hub_download
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

# Initialize VLM components
def init_model():
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

    model_name = "biomedclip_local"
    
    with open("checkpoints/open_clip_config.json", "r") as f:
        config = json.load(f)
        model_cfg = config["model_cfg"]
        preprocess_cfg = config["preprocess_cfg"]

    # Register the model configuration
    if (not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and model_cfg is not None):
        _MODEL_CONFIGS[model_name] = model_cfg

    # Now initialize tokenizer after registering the model config
    tokenizer = get_tokenizer(model_name)
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained="checkpoints/open_clip_pytorch_model.bin",
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )
    
    weights_path = "/home/BrainHack/finetuned_model_3k_epoch_20.pt"
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    return model, tokenizer, preprocess, device

def generate_answer(window, image_path, question):
    try:
        # Preprocess image
        image = window.preprocess(Image.open(image_path)).to(window.device).unsqueeze(0)
        
        template = f"Question: {question} Answer:"
        texts = window.tokenizer([template + " " + i for i in window.issues], context_length=256).to(window.device)
        
        with torch.no_grad():
            image_features, text_features, logit_scale = window.model(image, texts)
            logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)
            
            logits = logits.cpu().numpy()
            sorted_indices = sorted_indices.cpu().numpy()
        
        # Process results similar to main.py
        identified_issues = []
        for idx in sorted_indices[0]:
            confidence = logits[0][idx]
            if confidence >= 0.0000001:  # confidence threshold
                identified_issues.append((window.issues[idx], confidence))
            else:
                break
        
        # Format response
        if not identified_issues:
            response = "I couldn't confidently identify any issues with the image."
        else:
            issue_descriptions = []
            for issue, confidence in identified_issues:
                if issue != "None":
                    issue_descriptions.append(f'{issue} (Confidence: {confidence:.4f})')
            if issue_descriptions:
                response = "The following potential issues were identified:\n" + "\n".join(issue_descriptions)
            else:
                response = "No significant issues were identified."

        if "fix" in question.lower() or "tool" in question.lower():
            if identified_issues:
                tool_list = ', '.join([window.tool_suggestions[i[0]] for i in identified_issues if i[0] != "None"])
                response += f"\n\nSuggested tools: {tool_list}"
        
        return response
        
    except Exception as e:
        return f"Error processing image: {str(e)}"

class ImageQAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Q&A System")
        self.setGeometry(100, 100, 1000, 800)

        # Main widget and splitter
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel for image and input
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.image_label)

        # Load image button
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        left_layout.addWidget(load_button)

        # Question input
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Enter your question...")
        left_layout.addWidget(self.question_input)

        # Submit button
        submit_button = QPushButton("Ask Question")
        submit_button.clicked.connect(self.process_question)
        left_layout.addWidget(submit_button)

        # Current answer display
        self.answer_display = QTextEdit()
        self.answer_display.setReadOnly(True)
        left_layout.addWidget(self.answer_display)

        # Right panel for history
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        history_label = QLabel("History")
        right_layout.addWidget(history_label)

        self.history_display = QTextEdit()
        self.history_display.setReadOnly(True)
        right_layout.addWidget(self.history_display)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        self.current_image_path = None
        self.qa_history = []

        # Initialize VLM components
        self.model, self.tokenizer, self.preprocess, self.device = init_model()
        
        # Define issues and tool suggestions
        self.issues = [
            "None",
            "eddy currents",
            "motion artifacts",
            "Field of View Cutoff",
            "Missing Slices",
            "Noise",
            "Low Resolution"
        ]
        
        self.tool_suggestions = {
            "None": "image has no issues",
            "eddy currents": "FSL's eddy tool",
            "motion artifacts": "SPM's realignment function",
            "Field of View Cutoff": "NIfTI header editing tools like FSL's fslroi",
            "Missing Slices": "Missing Slices tool",
            "Noise": "Noise reduction filters",
            "Low Resolution": "upsample"
        }

    def load_image(self):
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )

        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                400,
                300,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.qa_history.append(
                f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded image: {file_path}"
            )
            self.update_history()

    def process_question(self):
        if not self.current_image_path:
            self.answer_display.setText("Please load an image first.")
            return

        question = self.question_input.text()
        if not question:
            self.answer_display.setText("Please enter a question.")
            return

        try:
            # Pass self as the window parameter
            answer = generate_answer(self, self.current_image_path, question)
            self.answer_display.setText(answer)

            # Add to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.qa_history.append(f"\n[{timestamp}] Q: {question}")
            self.qa_history.append(f"A: {answer}")
            self.update_history()

            # Clear question input
            self.question_input.clear()

        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            self.answer_display.setText(error_msg)
            self.qa_history.append(
                f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {error_msg}"
            )
            self.update_history()

    def update_history(self):
        self.history_display.setText("\n".join(self.qa_history))
        # Scroll to bottom
        scrollbar = self.history_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def main():
    app = QApplication(sys.argv)
    window = ImageQAWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
