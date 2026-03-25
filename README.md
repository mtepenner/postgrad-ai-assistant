# Vi - Modular Voice-Activated AI Assistant

Vi is an experimental, 100% offline, voice-activated artificial intelligence assistant built in Python.

## ✨ Features

* **Complete Offline Functionality:** No internet connection required. All transcription, processing, routing, and database searching happens entirely on your local machine.
* **Retrieval-Augmented Generation (RAG):** Ask questions about your personal files. Vi extracts answers from `.txt`, `.pdf`, and `.docx` files using a local vector database.
* **Conversational Memory:** Vi remembers the recent context of your general chats, allowing for more natural, multi-turn interactions.
* **Local Computer Vision:** Ask Vi to "look at this" or "describe what you see," and she will capture a webcam frame and generate an image caption.
* **Semantic Intent Routing (Zero-Shot):** Intelligently routes your prompt based on context and meaning, rather than relying on strict, hardcoded keywords.
* **Local STT & TTS:** Uses OpenAI's localized `whisper-tiny` model for transcription, `pyttsx3` for vocalization, and `openwakeword` for continuous listening.

## 🏗 Architecture

1.  **The Ear:** `openwakeword` and `SpeechRecognition` feed audio into a localized `Whisper` pipeline.
2.  **The Router:** `distilbart-mnli-12-3` dynamically matches the command to an AI expert.
3.  **The Experts:**
    * *Vision:* Passes `cv2` webcam frames to a `Salesforce/BLIP` image-to-text pipeline.
    * *RAG:* Vectorizes `./local_docs` using `unstructured` and `FAISS`, extracting answers via `RoBERTa-squad2`.
    * *NLP:* Specialized pipelines handle Sentiment, Summarization, and Context-Aware General Chat.
4.  **The Voice:** `pyttsx3` processes the expert's text output into audible speech.

## 📋 Prerequisites

* Python 3.8+
* A working microphone and webcam
* C++ Build Tools (Required on Windows for compiling FAISS and certain dependencies)

## 🚀 Installation

1.  **Install base dependencies:**
    ```bash
    pip install SpeechRecognition pyaudio transformers torch pyttsx3 numpy openwakeword soundfile opencv-python pillow langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers unstructured[all-docs]
    ```

    *(Note: The `unstructured[all-docs]` package handles dependencies like `pdfminer.six` and `python-docx` automatically).*

2.  **Add your documents:**
    Upon first run, Vi will automatically create a `./local_docs` directory in the project root. Drop any `.txt`, `.pdf`, or `.docx` files you want Vi to learn into this folder.

## 🗣️ Example Commands
* *"Hey Jarvis... What do you see right now?"* -> (Triggers Vision Camera)
* *"Hey Jarvis... Search my files to tell me where Vi was created."* -> (Triggers RAG search)
* *"Hey Jarvis... Summarize this sentence for me..."* -> (Triggers Summarization Expert)

## ⚙️ Customizing the Wake Word
Use the official [OpenWakeWord Google Colab Training Notebook](https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/openwakeword_model_training.ipynb) to generate a custom `.tflite` model, then update the `wakeword_models` parameter in `main.py`.
