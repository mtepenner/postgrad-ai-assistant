# Vi - Modular Voice-Activated AI Assistant

Vi is an experimental, voice-activated artificial intelligence assistant built in Python. Instead of relying on a single monolithic Large Language Model (LLM), Vi uses an **Agentic Router Pattern**. It listens for a wake word, processes spoken commands into text, and dynamically routes the request to specialized, locally-hosted Hugging Face pipelines based on the user's intent.

## 🚀 Features

* **Continuous Wake Word Detection:** Constantly listens for the designation "Vi" to initialize the command loop.
* **Speech-to-Text (STT):** Captures user audio and transcribes it using Google's STT API (can be swapped for local Whisper models).
* **Dynamic Intent Routing:** Parses the transcript and routes the prompt to the most relevant specialized AI expert.
* **Specialized AI Experts:** Currently implements three distinct Hugging Face models:
    * **Sentiment Analysis** (`distilbert-base-uncased-finetuned-sst-2-english`)
    * **Text Summarization** (`sshleifer/distilbart-cnn-12-6`)
    * **General Chat / QA** (`distilgpt2`)

## 🧠 Architecture

Vi relies on a lightweight dispatcher model. By utilizing smaller, task-specific "distilled" models rather than a single massive LLM, the system can run locally without exhausting system RAM or VRAM. 

1.  **The Ear:** `SpeechRecognition` actively monitors the microphone.
2.  **The Router:** A parsing function determines the user's core intent.
3.  **The Experts:** The `transformers` pipeline processes the text through the selected model and returns the output.

## 🛠️ Prerequisites

* Python 3.8+
* A working microphone

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/vi-assistant.git](https://github.com/yourusername/vi-assistant.git)
   cd vi-assistant
````

2.  Install the required system dependencies (for PyAudio):

      * **Windows:** PyAudio should install normally via pip.
      * **macOS:** `brew install portaudio`
      * **Linux (Debian/Ubuntu):** `sudo apt-get install python3-pyaudio portaudio19-dev`

3.  Install the Python dependencies:

    ```bash
    pip install SpeechRecognition pyaudio transformers torch
    ```

## 💻 Usage

Run the main Python script to bring Vi online:

```bash
python main.py
```

Wait for the "Vi System Online" prompt. Speak the name "Vi", followed by your command.

**Example Commands:**

  * *"Vi, how does this sound: I am having a fantastic day today\!"* (Triggers Sentiment Expert)
  * *"Vi, summarize this: [read a long paragraph]"* (Triggers Summarization Expert)
  * *"Vi, what is the capital of France?"* (Triggers General Chat Expert)

## 🔮 Future Improvements

  * Replace the current keyword-based routing with a Zero-Shot Classification model for true semantic NLP routing.
  * Implement `OpenWakeWord` or `Porcupine` for entirely offline, highly efficient wake word detection.
  * Add a Text-to-Speech (TTS) engine so Vi can vocalize her responses.



Would you like me to draft a more advanced version of the `route_to_expert` function that uses a Zero-Shot Classification model instead of `if/else` keyword statements to add to your codebase?
```
