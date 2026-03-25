import speech_recognition as sr
from transformers import pipeline
import time
import pyttsx3
import pyaudio
import numpy as np
from openwakeword.model import Model
import warnings
import os
import cv2
from PIL import Image

# RAG specific imports updated for unstructured
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Suppress standard Hugging Face warnings for a cleaner console
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

class ViAssistant:
    def __init__(self):
        print("Initializing Vi's neural pathways...")
        
        # Initialize TTS Engine
        self.engine = pyttsx3.init()
        self.configure_voice()
        
        # Initialize OpenWakeWord
        print("Loading OpenWakeWord models...")
        self.oww_model = Model(wakeword_models=["hey_jarvis"])
        
        print("Loading Language and Vision Models (This may take a moment)...")
        
        # 1. Local Offline STT
        self.stt_expert = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
        
        # 2. Semantic Router
        self.router_expert = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
        
        # 3. Vision Expert (Image Captioning)
        self.vision_expert = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        
        # 4. Standard Language Experts
        self.sentiment_expert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.summary_expert = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        
        self.chat_expert = pipeline("text-generation", model="distilgpt2")
        self.chat_history = [] # Initialize conversation memory
        
        # 5. RAG / Extractive QA Expert
        self.qa_expert = pipeline("question-answering", model="deepset/roberta-base-squad2")
        self.setup_rag()
        
        self.recognizer = sr.Recognizer()

    def configure_voice(self):
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        self.engine.setProperty('rate', 175)

    def speak(self, text):
        print(f"\nVi: {text}\n")
        self.engine.say(text)
        self.engine.runAndWait()

    def setup_rag(self):
        print("Indexing local documents from './local_docs'...")
        docs_dir = "./local_docs"
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
            with open(os.path.join(docs_dir, "sample.txt"), "w") as f:
                f.write("Vi is an experimental AI. The creator of Vi wanted an assistant that runs entirely offline. Vi currently operates from Hillsboro, Oregon.")
        
        # Updated to use UnstructuredFileLoader for multi-format support (.txt, .pdf, .docx)
        loader = DirectoryLoader(docs_dir, glob="**/*.*", loader_cls=UnstructuredFileLoader)
        documents = loader.load()
        
        if not documents:
            self.rag_index = None
            print("No documents found in './local_docs'. RAG is disabled.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.rag_index = FAISS.from_documents(chunks, embeddings)
        print(f"RAG initialized with {len(chunks)} document chunks.")

    def listen_for_wake_word(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1280

        audio = pyaudio.PyAudio()
        mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("\nListening for wake word (Say 'Hey Jarvis')...")
        
        try:
            while True:
                pcm = mic_stream.read(CHUNK, exception_on_overflow=False)
                pcm_arr = np.frombuffer(pcm, dtype=np.int16)
                
                self.oww_model.predict(pcm_arr)
                
                for mdl, scores in self.oww_model.prediction_buffer.items():
                    if scores[-1] > 0.5:
                        mic_stream.stop_stream()
                        mic_stream.close()
                        audio.terminate()
                        return True
        except KeyboardInterrupt:
            mic_stream.stop_stream()
            mic_stream.close()
            audio.terminate()
            exit()

    def capture_command(self):
        with sr.Microphone() as source:
            self.speak("Vi is awake!")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Transcribing locally...")
                
                raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                result = self.stt_expert({"sampling_rate": 16000, "raw": audio_np})
                command = result["text"].strip()
                
                print(f"You said: {command}")
                return command
            except sr.WaitTimeoutError:
                self.speak("I didn't hear anything.")
                return None

    def execute_vision(self):
        print("Accessing webcam...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return "I couldn't access your camera."
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        print("Analyzing visual data...")
        result = self.vision_expert(pil_img)
        caption = result[0]['generated_text']
        return f"I see {caption}."

    def execute_rag(self, command):
        if not self.rag_index:
            return "I don't have any local documents to search through."
            
        print("Searching vector index...")
        docs = self.rag_index.similarity_search(command, k=1)
        if not docs:
            return "I couldn't find any relevant information in your files."
            
        context = docs[0].page_content
        print("Extracting answer from document context...")
        result = self.qa_expert(question=command, context=context)
        
        if result['score'] < 0.1:
            return "I found some documents, but couldn't extract a confident answer."
            
        return f"Based on your documents: {result['answer']}."

    def route_to_expert(self, command):
        candidate_labels = [
            "summarize text", 
            "analyze sentiment", 
            "describe what you see or look at camera",
            "search local documents or files",
            "general chat or answer question"
        ]
        
        print("Analyzing intent...")
        classification = self.router_expert(command, candidate_labels)
        top_intent = classification['labels'][0]
        print(f"Intent classified as: '{top_intent}' ({classification['scores'][0]:.2f})")
        
        if top_intent == "describe what you see or look at camera":
            return self.execute_vision()
            
        elif top_intent == "search local documents or files":
            return self.execute_rag(command)
            
        elif top_intent == "summarize text":
            result = self.summary_expert(command, max_length=50, min_length=10, do_sample=False)
            return result[0]['summary_text']
            
        elif top_intent == "analyze sentiment":
            result = self.sentiment_expert(command)
            return f"This sounds {result[0]['label']}."
            
        else:
            # Memory injection for general chat
            memory_context = "\n".join([f"User: {turn['user']}\nVi: {turn['vi']}" for turn in self.chat_history[-3:]])
            prompt = f"{memory_context}\nUser: {command}\nVi: " if memory_context else f"User: {command}\nVi: "
            
            # Truncate left side if prompt is too long for the model
            if len(prompt) > 800:
                prompt = prompt[-800:]
            
            result = self.chat_expert(prompt, max_new_tokens=40, pad_token_id=50256, truncation=True)
            generated_full = result[0]['generated_text']
            
            # Extract just Vi's response, ignoring the prepended prompt
            response = generated_full[len(prompt):].split('\n')[0].strip()
            if not response:
                response = "I'm not entirely sure how to answer that."
            
            self.chat_history.append({"user": command, "vi": response})
            return response

    def run(self):
        self.speak("Vi System Online.")
        while True:
            if self.listen_for_wake_word():
                command = self.capture_command()
                if command:
                    response = self.route_to_expert(command)
                    self.speak(response)
            time.sleep(0.5)

if __name__ == "__main__":
    vi = ViAssistant()
    vi.run()
