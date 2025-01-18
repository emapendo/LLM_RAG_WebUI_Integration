from flask import Flask, jsonify, request
import threading, logging, subprocess, sys, os
from config import Config
from llm_handler import LLMHandler
from document_processor import DocumentProcessor
import whisper
from gtts import gTTS
from playsound import playsound

# Initialize Flask app and key components
app = Flask(__name__)
messages = []  # Stores messages for conversation
config = Config()  # Configuration settings
llm_handler = LLMHandler(config.OLLAMA_API_URL, config.SYSTEM_PROMPT)  # LLM handler instance
doc_processor = DocumentProcessor()  # Document processor instance for RAG mode

# Initialize Whisper model for audio transcription
whisper_model = whisper.load_model("tiny")  # Load a small Whisper model for transcription

# Text-to-Speech function using gTTS
def speak_text(text):
    """Convert text to speech using gTTS and play it."""
    try:
        tts = gTTS(text)  # Generate speech from text
        audio_file = "response.mp3"  # Temporary file for storing audio
        tts.save(audio_file)  # Save the audio file
        playsound(audio_file)  # Play the audio file
        os.remove(audio_file)  # Clean up the audio file after playing
    except Exception as e:
        print(f"Error in TTS: {e}")  # Handle any errors in TTS

# Transcription function for audio files
def transcribe_audio(audio_path):
    """Transcribe audio file to text using Whisper."""
    result = whisper_model.transcribe(audio_path)  # Transcribe audio to text
    return result.get("text", "")  # Return the transcribed text

# Function to handle user input
def handle_user_input(text: str):
    """Process user input and generate appropriate responses."""
    if text.startswith("/mode"):
        # Handle mode switching (chat or RAG)
        mode = text.split()[1]  # Extract mode from input
        llm_handler.switch_mode(mode)  # Switch LLM handler mode
        if mode == "rag":
            # Load documents for RAG mode
            doc_processor.load_documents(config.RAG_URLS)
        messages.append({"role": "system", "content": f"Switched to {mode} mode", "color": "yellow"})
    elif text == "/speak":
        # Handle audio transcription input
        audio_file = input("Provide the audio file path: ")
        text = transcribe_audio(audio_file)  # Transcribe the provided audio file
        print(f"Transcribed audio: {text}")  # Print the transcribed text
    else:
        # Generate response in the current mode
        context = None
        if llm_handler.current_mode == "rag":
            # Retrieve relevant context for RAG mode
            context = doc_processor.get_relevant_context(text)
        messages.append({"role": "user", "content": text, "color": "blue"})  # Log user query
        response = llm_handler.generate_response(text, context)  # Generate response using LLM
        messages.append({"role": "assistant", "content": response, "color": "green"})  # Log assistant response
        speak_text(response)  # Use TTS to speak the response

# Thread to handle terminal input
def input_thread():
    """Handle user input through the terminal."""
    print("Enter messages (use /mode chat or /mode rag to switch modes):")
    while True:
        text = input("> ")  # Read user input from terminal
        handle_user_input(text)  # Process the input

# Expose messages to Streamlit UI
@app.route('/get_messages', methods=['GET'])
def get_messages():
    """Provide messages for the Streamlit WebUI."""
    try:
        if not messages:
            return jsonify({"messages": []})
        return jsonify({"messages": messages})
    except Exception as e:
        return jsonify({"error": str(e)})

# Main function to run the Flask app and Streamlit UI
if __name__ == '__main__':
    # Start the Streamlit WebUI in a separate process
    stt = subprocess.Popen(["streamlit", "run", "streamlit_app.py"])

    try:
        # Configure logging for the Flask server
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.WARNING)

        # Start the terminal input thread
        thread = threading.Thread(target=input_thread)
        thread.daemon = True  # Ensure thread exits with the main program
        thread.start()

        # Run the Flask app
        app.run(port=config.FLASK_PORT)
    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        stt.terminate()  # Terminate Streamlit process
        sys.exit(0)  # Exit the program