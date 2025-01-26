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


def generate_answer(image, question):
    return "hello world"


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
            # Replace generate_answer() with your actual question-answering function
            answer = generate_answer(self.current_image_path, question)
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
