import speech_recognition as sr
from transformers import pipeline
import time
import pyttsx3  

class ViAssistant:
    def __init__(self):
        print("Initializing Vi's neural pathways...")
        
        # Initialize TTS Engine
        self.engine = pyttsx3.init()
        self.configure_voice()
        
        # 1. Load Specialized Hugging Face Models
        self.sentiment_expert = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        self.summary_expert = pipeline(
            "summarization", 
            model="sshleifer/distilbart-cnn-12-6"
        )
        
        self.chat_expert = pipeline(
            "text-generation", 
            model="distilgpt2" 
        )
        
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def configure_voice(self):
        """Sets the voice properties for Vi."""
        voices = self.engine.getProperty('voices')
        # Selecting a female-sounding voice if available
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        self.engine.setProperty('rate', 175) # Speed of speech

    def speak(self, text):
        """Vocalizes the provided text."""
        print(f"\nVi: {text}\n")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen_for_wake_word(self):
        with self.mic as source:
            print("Listening for 'Vi'...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            
        try:
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
            self.speak("Vi didn't catch that.")
            return None

    def route_to_expert(self, command):
        command_lower = command.lower()
        
        if "summarize" in command_lower:
            print("Routing to Summarization Expert...")
            text_to_summarize = command.replace("summarize", "")
            result = self.summary_expert(text_to_summarize, max_length=50, min_length=10, do_sample=False)
            return result[0]['summary_text']
            
        elif "how does this sound" in command_lower or "sentiment" in command_lower:
            print("Routing to Sentiment Expert...")
            result = self.sentiment_expert(command)
            label = result[0]['label']
            return f"This sounds {label}." # Shortened for better vocalization
            
        else:
            print("Routing to General Chat Expert...")
            result = self.chat_expert(command, max_length=50, num_return_sequences=1)
            return result[0]['generated_text']

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
