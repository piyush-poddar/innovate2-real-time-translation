import sounddevice as sd
import numpy as np
import speech_recognition as sr
from io import BytesIO
import soundfile as sf
import queue
from gtts import gTTS
import google.generativeai as genai
import pygame
import threading
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure the generative model API
genai.configure(api_key='GEMINI-API-KEY')
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Set up parameters
virtual_audio_device_name = "CABLE Output (VB-Audio Virtual Cable)"
sampling_rate = 48000  
block_size = 1024      
channels = 2           
record_duration = 7    

recognizer = sr.Recognizer()
pygame.mixer.init()

audio_queue = queue.Queue()

# Find the device index of the VAC
device_info = sd.query_devices()
device_index = next((i for i, device in enumerate(device_info) if virtual_audio_device_name in device['name']), None)
if device_index is None:
    raise ValueError(f"Device '{virtual_audio_device_name}' not found. Check your VAC setup.")
# print(device_info)
# print(device_index)

# Callback function to capture audio blocks
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Error: {status}")
    audio_queue.put(indata.copy())  # Put each audio block into the queue

# Threaded function to process audio chunks
def process_audio():
    while True:
        audio_data = []
        start_time = time.time()
        
        while time.time() - start_time < record_duration:
            if not audio_queue.empty():
                audio_data.append(audio_queue.get())
        
        # Ensure there's enough data to process
        if audio_data:
            audio_data = np.concatenate(audio_data, axis=0)
            audio_buffer = BytesIO()
            sf.write(audio_buffer, audio_data, samplerate=sampling_rate, format='WAV')
            audio_buffer.seek(0)
            
            # Perform speech recognition on the current audio chunk
            with sr.AudioFile(audio_buffer) as source:
                recognizer.adjust_for_ambient_noise(source)
                audio_content = recognizer.record(source)
                try:
                    recognized_text = recognizer.recognize_google(audio_content, language="hi-IN")
                    print("Recognized Text:", recognized_text)

                    prompt = f"Translate {recognized_text} to English and return the translated text in a single line. Only return the translated text and nothing else. If no text is present then return empty string."
                    response = model.generate_content([prompt], safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                    })
                    translated_text = response.text
                    print("Translated Text:", translated_text)

                    # Convert the translation to speech
                    speak = gTTS(text=translated_text, lang='en', slow=False)
                    audio_buffer = BytesIO()
                    speak.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)

                    # Play the translated audio asynchronously
                    pygame.mixer.music.load(audio_buffer, 'mp3')
                    pygame.mixer.music.play()

                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand the audio.")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")

# Start capturing audio in a separate thread
def start_audio_capture():
    with sd.InputStream(device=device_index, channels=channels, samplerate=sampling_rate, blocksize=block_size, callback=audio_callback):
        print("Recording audio in chunks...")
        threading.Thread(target=process_audio, daemon=True).start()
        while True:
            time.sleep(1)  # To keep the main thread alive

start_audio_capture()