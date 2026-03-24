import speech_recognition as sr
from transformers import pipeline
import time

class ViAssistant:
    def __init__(self):
        print("Initializing Vi's neural pathways...")
        
        # 1. Load Specialized Hugging Face Models
        # NOTE: We use smaller models here to prevent RAM/VRAM exhaustion
        
        # Expert 1: Sentiment Analysis
        self.sentiment_expert = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Expert 2: Summarization
        self.summary_expert = pipeline(
            "summarization", 
            model="sshleifer/distilbart-cnn-12-6"
        )
        
        # Expert 3: General Chat / QA
        self.chat_expert = pipeline(
            "text-generation", 
            model="distilgpt2" 
        )
        
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def listen_for_wake_word(self):
        """
        In a production environment, replace this with OpenWakeWord or Porcupine 
        for offline, low-latency wake word detection.
        """
        with self.mic as source:
            print("Listening for 'Vi'...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            
        try:
            # Using Google's free tier STT for quick prototyping
            transcript = self.recognizer.recognize_google(audio).lower()
            if "vi" in transcript or "vee" in transcript:
                return True
        except sr.UnknownValueError:
            pass
        return False

    def capture_command(self):
        with self.mic as source:
            print("Vi is awake! What is your command?")
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        try:
            command = self.recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            print("Vi didn't catch that.")
            return None

    def route_to_expert(self, command):
        """
        The Router: Determines which model handles the request.
        """
        command_lower = command.lower()
        
        if "summarize" in command_lower:
            print("Routing to Summarization Expert...")
            # In reality, you'd extract the text to summarize here
            text_to_summarize = command.replace("summarize", "")
            result = self.summary_expert(text_to_summarize, max_length=50, min_length=10, do_sample=False)
            return result[0]['summary_text']
            
        elif "how does this sound" in command_lower or "sentiment" in command_lower:
            print("Routing to Sentiment Expert...")
            result = self.sentiment_expert(command)
            return f"This sounds {result[0]['label']} (Confidence: {result[0]['score']:.2f})"
            
        else:
            print("Routing to General Chat Expert...")
            result = self.chat_expert(command, max_length=50, num_return_sequences=1)
            return result[0]['generated_text']

    def run(self):
        print("Vi System Online.")
        while True:
            # 1. Block and wait for wake word
            if self.listen_for_wake_word():
                
                # 2. Capture the actual instruction
                command = self.capture_command()
                
                if command:
                    # 3. Route and execute
                    response = self.route_to_expert(command)
                    print(f"\nVi: {response}\n")
                    
            time.sleep(0.5)

if __name__ == "__main__":
    # Ensure you pip install SpeechRecognition pyaudio transformers torch
    vi = ViAssistant()
    vi.run()
