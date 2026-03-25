# Vi - Modular Voice-Activated AI Assistant

Vi is an experimental, voice-activated artificial intelligence assistant built in Python.

## ✨ Features

* **Continuous Wake Word Detection:** Constantly listens for "Vi" to initialize.
* **Speech-to-Text (STT):** Transcribes audio using Google's STT API.
* **Text-to-Speech (TTS):** Vi now vocalizes her responses using the `pyttsx3` engine.
* **Dynamic Intent Routing:** Routes requests to specialized Hugging Face experts.

## 🏗 Architecture

1.  **The Ear:** `SpeechRecognition` monitors the microphone.
2.  **The Router:** Logic determines if the user wants Sentiment Analysis, Summarization, or General Chat.
3.  **The Voice:** `pyttsx3` processes the text output into audible speech.

## 📋 Prerequisites

* Python 3.8+
* A working microphone
* `espeak` (Required for Linux users: `sudo apt-get install espeak`)

## 🚀 Installation

1.  **Install dependencies:**
    ```bash
    pip install SpeechRecognition pyaudio transformers torch pyttsx3
    ```

## 🛠 Future Improvements

* Replace keyword routing with Zero-Shot Classification.
* Implement `OpenWakeWord` for offline detection.
