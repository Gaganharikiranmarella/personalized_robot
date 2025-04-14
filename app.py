import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import speech_recognition as sr
import pyttsx3

# Load DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Text-to-speech setup using pyttsx3
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Speech-to-text setup using microphone
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("🗣️ You said:", text)
        return text
    except Exception as e:
        print("Speech error:", e)
        return ""

# Load CSV file as conversation context
def load_dataset(path):
    try:
        df = pd.read_csv(path)
        return "\n".join(df.astype(str).apply(" - ".join, axis=1).tolist())
    except Exception as e:
        print("CSV Load Error:", e)
        return ""

# Generate response from the model
chat_history_ids = None
def get_response(prompt):
    global chat_history_ids
    new_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids

    chat_history_ids = model.generate(
        input_ids, 
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response.strip()

# Start the assistant
if __name__ == "__main__":
    print("📥 Loading mental health dataset...")
    context_data = load_dataset("datasets/your_dataset.csv")

    print("🧠 Mental health assistant ready!")
    speak("Hello! I'm here to talk about mental health.")

    while True:
        user_input = listen()
        if user_input.lower() in ['exit', 'quit', 'stop']:
            speak("Goodbye. Take care of your mental health!")
            break
        
        combined_prompt = f"{context_data}\nUser: {user_input}"
        reply = get_response(combined_prompt)
        print("🤖:", reply)
        speak(reply)
