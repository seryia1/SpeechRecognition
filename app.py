import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import librosa
import numpy as np
import tempfile
import os
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import gc
import io
import wave
import threading
import time

# Configure page
st.set_page_config(
    page_title="🎙️ Enhanced Speech Recognition App",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .api-success {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .api-error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .timestamp-box {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        border-left: 3px solid #2196f3;
    }
    .recording-indicator {
        background: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #f44336;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .paused-indicator {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ff9800;
        text-align: center;
    }
    .recorded-audio-box {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #4caf50;
        margin: 1rem 0;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .error-message {
        background: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    .success-message {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .info-message {
        background: #e3f2fd;
        color: #1565c0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-message {
        background: #fff3e0;
        color: #ef6c00;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'paused' not in st.session_state:
        st.session_state.paused = False
    if 'recorded_audio_bytes' not in st.session_state:
        st.session_state.recorded_audio_bytes = None
    if 'recorded_audio_filename' not in st.session_state:
        st.session_state.recorded_audio_filename = None
    if 'recorded_audio_file' not in st.session_state:
        st.session_state.recorded_audio_file = None
    if 'recorded_audio_data' not in st.session_state:
        st.session_state.recorded_audio_data = None
    if 'transcription_history' not in st.session_state:
        st.session_state.transcription_history = []
    if 'microphone_available' not in st.session_state:
        st.session_state.microphone_available = None

# Language options for speech recognition
LANGUAGE_OPTIONS = {
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "Spanish (Spain)": "es-ES",
    "Spanish (Mexico)": "es-MX",
    "French (France)": "fr-FR",
    "German (Germany)": "de-DE",
    "Italian (Italy)": "it-IT",
    "Portuguese (Brazil)": "pt-BR",
    "Russian (Russia)": "ru-RU",
    "Japanese (Japan)": "ja-JP",
    "Korean (South Korea)": "ko-KR",
    "Chinese (Mandarin)": "zh-CN",
    "Arabic (Saudi Arabia)": "ar-SA",
    "Hindi (India)": "hi-IN",
    "Dutch (Netherlands)": "nl-NL",
    "Swedish (Sweden)": "sv-SE",
    "Norwegian (Norway)": "no-NO",
    "Danish (Denmark)": "da-DK",
    "Finnish (Finland)": "fi-FI",
    "Polish (Poland)": "pl-PL"
}

# Speech Recognition API options
API_OPTIONS = {
    "Google Speech Recognition": "google",
    "OpenAI Whisper (Local)": "whisper",
    "Sphinx (Offline)": "sphinx"
}

@st.cache_resource
def load_asr_model(model_name):
    """Load and cache the ASR model"""
    try:
        return pipeline("automatic-speech-recognition", model=model_name)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def check_microphone_availability():
    """Check if microphone is available and working"""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        # Check if there are any input devices
        device_count = p.get_device_count()
        if device_count == 0:
            p.terminate()
            return False, "No audio devices found"
        
        # Try to find a working input device
        input_device_found = False
        for i in range(device_count):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_device_found = True
                    break
            except:
                continue
        
        p.terminate()
        
        if not input_device_found:
            return False, "No input devices (microphones) found"
        
        return True, "Microphone available"
        
    except ImportError:
        return False, "PyAudio not installed - recording not available"
    except Exception as e:
        return False, f"Microphone check failed: {str(e)}"

class AudioProcessor:
    """Handle audio file conversion and preprocessing"""
    
    @staticmethod
    def convert_to_wav(input_file, output_file, target_sr=16000):
        """Convert any audio format to PCM WAV format"""
        try:
            # Method 1: Try with librosa (most reliable)
            try:
                audio, sr = librosa.load(input_file, sr=target_sr, mono=True)
                sf.write(output_file, audio, target_sr, format='WAV', subtype='PCM_16')
                return True, "Converted using librosa"
            except Exception as e1:
                # Method 2: Try with pydub
                try:
                    audio = AudioSegment.from_file(input_file)
                    audio = audio.set_channels(1).set_frame_rate(target_sr)
                    audio.export(output_file, format="wav")
                    return True, "Converted using pydub"
                except Exception as e2:
                    # Method 3: Try with soundfile directly
                    try:
                        data, sr = sf.read(input_file)
                        if len(data.shape) > 1:
                            data = np.mean(data, axis=1)
                        if sr != target_sr:
                            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                        sf.write(output_file, data, target_sr, format='WAV', subtype='PCM_16')
                        return True, "Converted using soundfile"
                    except Exception as e3:
                        return False, f"All conversion methods failed: librosa({str(e1)}), pydub({str(e2)}), soundfile({str(e3)})"
        except Exception as e:
            return False, f"Audio conversion error: {str(e)}"

def safe_file_cleanup(file_path, max_retries=3, delay=0.1):
    """Safely delete a file with retries"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                gc.collect()
                time.sleep(delay)
                os.unlink(file_path)
                return True
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
                continue
            else:
                st.warning(f"Could not delete temporary file: {str(e)}")
                return False
    return True

class AudioRecorder:
    """Handle audio recording with better error handling and validation"""
    
    def __init__(self):
        self.sample_rate = 44100  # Higher quality
        self.chunk_size = 1024
        self.format = None
        self.channels = 1
        self.pyaudio_available = False
        
        # Try to import and initialize PyAudio
        try:
            import pyaudio
            self.pyaudio = pyaudio
            self.format = pyaudio.paInt16
            self.pyaudio_available = True
        except ImportError:
            self.pyaudio_available = False
    
    def is_available(self):
        """Check if recording is available"""
        return self.pyaudio_available
    
    def get_microphone_info(self):
        """Get information about available microphones"""
        if not self.pyaudio_available:
            return []
        
        try:
            p = self.pyaudio.PyAudio()
            devices = []
            
            for i in range(p.get_device_count()):
                try:
                    device_info = p.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        devices.append({
                            'index': i,
                            'name': device_info['name'],
                            'channels': device_info['maxInputChannels'],
                            'sample_rate': device_info['defaultSampleRate']
                        })
                except:
                    continue
            
            p.terminate()
            return devices
        except Exception as e:
            return []
    
    def test_microphone(self, duration=2):
        """Test microphone by recording a short sample"""
        if not self.pyaudio_available:
            return False, "PyAudio not available"
        
        try:
            p = self.pyaudio.PyAudio()
            
            # Try to open a stream
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Record a short test
            frames = []
            for _ in range(int(self.sample_rate / self.chunk_size * duration)):
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return False, f"Recording failed: {str(e)}"
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Check if we got audio data
            if frames:
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                max_amplitude = np.max(np.abs(audio_data))
                
                if max_amplitude < 50:  # Very quiet
                    return False, f"Microphone appears to be muted or very quiet (max amplitude: {max_amplitude})"
                elif max_amplitude < 500:  # Quiet but working
                    return True, f"Microphone working but quiet (max amplitude: {max_amplitude}) - try speaking louder"
                else:
                    return True, f"Microphone working well (max amplitude: {max_amplitude})"
            else:
                return False, "No audio data recorded"
                
        except Exception as e:
            return False, f"Microphone test failed: {str(e)}"
    
    def record_audio(self, duration):
        """Record audio and return raw audio data with better validation"""
        if not self.pyaudio_available:
            raise Exception("PyAudio not available - cannot record audio")
        
        try:
            p = self.pyaudio.PyAudio()
            
            # Check if microphone is available
            if p.get_device_count() == 0:
                raise Exception("No audio devices found")
            
            # Get default input device info
            try:
                device_info = p.get_default_input_device_info()
                st.info(f"🎤 Using microphone: {device_info['name']}")
            except:
                st.info("🎤 Using default microphone")
            
            stream = p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            total_frames = int(self.sample_rate / self.chunk_size * duration)
            
            # Add progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(total_frames):
                if not st.session_state.recording:
                    break
                
                # Handle pause/resume
                while st.session_state.paused and st.session_state.recording:
                    time.sleep(0.1)
                
                if not st.session_state.recording:
                    break
                
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Update progress
                    progress = (i + 1) / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Recording... {progress:.0%} ({i+1}/{total_frames})")
                    
                except Exception as e:
                    st.warning(f"Audio read error: {str(e)}")
                    break
            
            progress_bar.empty()
            status_text.empty()
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if frames:
                # Convert to numpy array
                audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                
                # Detailed audio analysis
                max_amplitude = np.max(np.abs(audio_data))
                rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
                
                st.info(f"📊 Audio Analysis: Max amplitude: {max_amplitude}, RMS: {rms:.1f}")
                
                # Check if recording is not silent with better thresholds
                if max_amplitude < 50:
                    raise Exception(f"Recording appears to be silent (max amplitude: {max_amplitude}) - check microphone volume and permissions")
                elif max_amplitude < 200:
                    st.warning(f"⚠️ Recording is very quiet (max amplitude: {max_amplitude}) - transcription may be poor")
                
                return audio_data
            else:
                return None
                
        except Exception as e:
            raise Exception(f"Recording error: {str(e)}")
    
    def save_as_mp3_bytes(self, audio_data):
        """Save audio data as MP3 in memory and return bytes"""
        try:
            # Convert numpy array to AudioSegment
            audio_segment = AudioSegment(
                audio_data.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=audio_data.dtype.itemsize,
                channels=self.channels
            )
            
            # Normalize audio (make it louder) but be careful not to clip
            normalized = audio_segment.normalize()
            
            # Apply some gain if it's still quiet
            if normalized.max_possible_amplitude < 10000:
                normalized = normalized + 6  # Add 6dB gain
            
            # Export as MP3 to bytes buffer
            mp3_buffer = io.BytesIO()
            normalized.export(mp3_buffer, format="mp3", bitrate="128k")
            mp3_buffer.seek(0)
            
            return mp3_buffer.getvalue()
            
        except Exception as e:
            raise Exception(f"Error creating MP3: {str(e)}")
    
    def save_as_wav_bytes(self, audio_data):
        """Save audio data as WAV in memory and return bytes"""
        try:
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            wav_buffer.seek(0)
            return wav_buffer.getvalue()
            
        except Exception as e:
            raise Exception(f"Error creating WAV: {str(e)}")

class SpeechRecognitionManager:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_processor = AudioProcessor()
        
    def transcribe_audio_whisper(self, audio_file, model_pipeline):
        """Transcribe audio file using Whisper"""
        try:
            # Load audio as numpy array (forces mono, resample to 16kHz)
            audio, sr = librosa.load(audio_file, sr=16000)
            
            # Check if audio is too short
            if len(audio) < 1600:  # Less than 0.1 seconds
                raise Exception("Audio file is too short (less than 0.1 seconds)")
            
            # Check if audio is silent
            if np.max(np.abs(audio)) < 0.001:
                raise Exception("Audio appears to be silent or very quiet")
            
            # Run the ASR pipeline with timestamps
            result = model_pipeline(audio, return_timestamps=True)
            
            return {
                "text": result["text"].strip(),
                "confidence": 1.0,
                "chunks": result.get("chunks", []),
                "method": "whisper"
            }
        except Exception as e:
            raise Exception(f"Whisper transcription error: {str(e)}")
    
    def transcribe_audio_file(self, audio_file, api_name, language="en-US", model_pipeline=None):
        """Transcribe audio file using selected API with proper format conversion"""
        temp_wav_path = None
        try:
            if api_name == "whisper":
                if not model_pipeline:
                    raise Exception("Whisper model not loaded")
                return self.transcribe_audio_whisper(audio_file, model_pipeline)
            else:
                # For other APIs, convert to proper format first
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                    temp_wav_path = temp_wav.name
                
                conversion_success, conversion_msg = self.audio_processor.convert_to_wav(
                    audio_file, temp_wav_path
                )
                
                if not conversion_success:
                    raise Exception(f"Audio conversion failed: {conversion_msg}")
                
                try:
                    result = self._transcribe_with_speech_recognition(temp_wav_path, api_name, language)
                    return result
                finally:
                    # Safe cleanup
                    if temp_wav_path:
                        safe_file_cleanup(temp_wav_path)
                        
        except Exception as e:
            # Ensure cleanup even on error
            if temp_wav_path:
                safe_file_cleanup(temp_wav_path)
            raise Exception(f"Transcription failed: {str(e)}")
    
    def _transcribe_with_speech_recognition(self, audio_file, api_name, language):
        """Transcribe using speech_recognition library APIs"""
        try:
            with sr.AudioFile(audio_file) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
            
            if not hasattr(audio, 'frame_data') or len(audio.frame_data) == 0:
                raise Exception("No audio data found in file")
            
            kwargs = {"language": language}
            
            if api_name == "google":
                text = self.recognizer.recognize_google(audio, **kwargs)
            elif api_name == "sphinx":
                lang_code = language.split('-')[0]
                text = self.recognizer.recognize_sphinx(audio, language=lang_code)
            else:
                raise Exception(f"Unsupported API: {api_name}")
            
            if not text or text.strip() == "":
                raise Exception("No speech detected in audio")
            
            return {
                "text": text.strip(),
                "confidence": 0.95,
                "language": language,
                "method": api_name
            }
            
        except sr.UnknownValueError:
            raise Exception("Could not understand audio - speech was unclear, too quiet, or silent")
        except sr.RequestError as e:
            if "quota" in str(e).lower():
                raise Exception(f"API quota exceeded: {str(e)}")
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise Exception(f"Network connection error: {str(e)}")
            else:
                raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Speech recognition error: {str(e)}")

def save_transcript_to_file(text, filename, file_format):
    """Save transcript to different file formats"""
    try:
        if file_format == "txt":
            return text.encode('utf-8')
        elif file_format == "json":
            data = {
                "transcript": text,
                "timestamp": datetime.now().isoformat(),
                "word_count": len(text.split()),
                "character_count": len(text)
            }
            return json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
        elif file_format == "csv":
            df = pd.DataFrame([{
                "transcript": text,
                "timestamp": datetime.now().isoformat(),
                "word_count": len(text.split()),
                "character_count": len(text)
            }])
            return df.to_csv(index=False).encode('utf-8')
        else:
            return text.encode('utf-8')
    except Exception as e:
        raise Exception(f"File saving error: {str(e)}")

def save_transcription_history(text, timestamp, method, language=None, confidence=None, file_name=None):
    """Save transcription to history"""
    st.session_state.transcription_history.append({
        'text': text,
        'timestamp': timestamp,
        'method': method,
        'language': language or "Unknown",
        'confidence': confidence or "N/A",
        'file_name': file_name
    })

def export_transcriptions(file_format="txt"):
    """Export transcription history as text file"""
    if not st.session_state.transcription_history:
        return None
    
    if file_format == "txt":
        content = "Speech-to-Text Transcription History\n"
        content += "=" * 50 + "\n\n"
        
        for i, item in enumerate(st.session_state.transcription_history, 1):
            content += f"Transcription #{i}\n"
            content += f"Method: {item['method']}\n"
            content += f"Language: {item.get('language', 'Unknown')}\n"
            content += f"Confidence: {item.get('confidence', 'N/A')}\n"
            content += f"Timestamp: {item['timestamp']}\n"
            if item.get('file_name'):
                content += f"File: {item['file_name']}\n"
            content += f"Text: {item['text']}\n"
            content += "-" * 30 + "\n\n"
        
        return content.encode('utf-8')
    else:
        return save_transcript_to_file(
            "\n".join([item['text'] for item in st.session_state.transcription_history]),
            "history",
            file_format
        )

def display_error_message(error_msg, error_type="general"):
    """Display formatted error messages"""
    error_icon = "❌"
    if "network" in error_msg.lower() or "connection" in error_msg.lower():
        error_icon = "🌐"
        error_type = "network"
    elif "api" in error_msg.lower() or "key" in error_msg.lower() or "auth" in error_msg.lower():
        error_icon = "🔑"
        error_type = "api"
    elif "audio" in error_msg.lower() or "microphone" in error_msg.lower() or "format" in error_msg.lower():
        error_icon = "🎤"
        error_type = "audio"
    elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
        error_icon = "⚠️"
        error_type = "quota"
    
    st.markdown(f"""
    <div class="error-message">
        <strong>{error_icon} Error ({error_type.title()}):</strong><br>
        {error_msg}
    </div>
    """, unsafe_allow_html=True)

def display_success_message(message):
    """Display formatted success messages"""
    st.markdown(f"""
    <div class="success-message">
        <strong>✅ Success:</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def display_info_message(message):
    """Display formatted info messages"""
    st.markdown(f"""
    <div class="info-message">
        <strong>ℹ️ Info:</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def display_warning_message(message):
    """Display formatted warning messages"""
    st.markdown(f"""
    <div class="warning-message">
        <strong>⚠️ Warning:</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Enhanced Speech Recognition App</h1>
        <p>Upload Files OR Record Live Audio | Multi-API Support | Professional Quality</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    sr_manager = SpeechRecognitionManager()
    audio_recorder = AudioRecorder()
    
    # Check microphone availability once
    if st.session_state.microphone_available is None:
        mic_available, mic_message = check_microphone_availability()
        st.session_state.microphone_available = mic_available
        if not mic_available:
            display_warning_message(f"Live recording not available: {mic_message}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Selection
        st.subheader("🔌 Speech Recognition API")
        selected_api_name = st.selectbox(
            "Choose API",
            options=list(API_OPTIONS.keys()),
            help="Select the speech recognition service to use"
        )
        api_name = API_OPTIONS[selected_api_name]
        
        # Whisper model selection and loading
        asr_pipeline = None
        if api_name == "whisper":
            st.subheader("🤖 Whisper Model")
            model_options = {
                "Whisper Tiny (Fast)": "openai/whisper-tiny",
                "Whisper Base (Balanced)": "openai/whisper-base",
                "Whisper Small (Better Quality)": "openai/whisper-small"
            }
            
            selected_model = st.selectbox(
                "Choose Model",
                options=list(model_options.keys()),
                help="Larger models provide better accuracy but are slower"
            )
            
            model_name = model_options[selected_model]
            
            # Load model
            with st.spinner(f"Loading {selected_model}..."):
                asr_pipeline = load_asr_model(model_name)
            
            if asr_pipeline:
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model")
                return
        
        # Language Selection
        st.subheader("🌍 Language Settings")
        selected_language = st.selectbox(
            "Speech Language",
            options=list(LANGUAGE_OPTIONS.keys()),
            help="Select the language you'll be speaking"
        )
        language_code = LANGUAGE_OPTIONS[selected_language]
        
        # Test API Connection
        st.subheader("🔍 API Status")
        if st.button("Test API Connection"):
            with st.spinner("Testing API connection..."):
                if api_name == "whisper":
                    is_working = asr_pipeline is not None
                elif api_name == "google":
                    is_working = True
                elif api_name == "sphinx":
                    is_working = True
                else:
                    is_working = False
                
                if is_working:
                    st.markdown('<div class="api-status api-success">✅ API Ready</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="api-status api-error">❌ API Not Available</div>', unsafe_allow_html=True)
        
        # Microphone testing section
        if st.session_state.microphone_available and audio_recorder.is_available():
            st.subheader("🎤 Microphone Test")
            
            # Show available microphones
            mics = audio_recorder.get_microphone_info()
            if mics:
                st.write("📱 Available microphones:")
                for mic in mics[:3]:  # Show first 3
                    st.write(f"• {mic['name']}")
            
            if st.button("🔍 Test Microphone"):
                with st.spinner("Testing microphone for 2 seconds..."):
                    test_success, test_message = audio_recorder.test_microphone(duration=2)
                    if test_success:
                        st.success(f"✅ {test_message}")
                    else:
                        st.error(f"❌ {test_message}")
        
        st.divider()
        
        # Recording Settings (only show if microphone available)
        if st.session_state.microphone_available:
            st.subheader("🎤 Recording Settings")
            recording_duration = st.slider("Recording Duration (seconds)", 5, 60, 15)
            
            # Audio quality settings
            st.subheader("🎵 Audio Quality")
            st.info("📀 **Format:** MP3 (128kbps)\n📊 **Sample Rate:** 44.1kHz\n🎧 **Channels:** Mono")
        
        # File Export Settings
        st.subheader("💾 Export Settings")
        export_format = st.selectbox(
            "Export Format",
            ["txt", "json", "csv"],
            help="Choose format for saving transcripts"
        )
        
        st.divider()
        
        # History management
        st.subheader("📝 Transcription History")
        if st.session_state.transcription_history:
            st.write(f"Total transcriptions: {len(st.session_state.transcription_history)}")
            
            if st.button("📥 Export History"):
                export_content = export_transcriptions(export_format)
                if export_content:
                    st.download_button(
                        label=f"Download History .{export_format.upper()}",
                        data=export_content,
                        file_name=f"transcription_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                        mime=f"text/{export_format}"
                    )
            
            if st.button("🗑️ Clear History"):
                st.session_state.transcription_history = []
                # Clear recorded audio
                st.session_state.recorded_audio_bytes = None
                st.session_state.recorded_audio_filename = None
                st.session_state.recorded_audio_file = None
                st.session_state.recorded_audio_data = None
                st.rerun()
        else:
            st.write("No transcriptions yet")
        
        # Audio Format Info
        st.subheader("📋 Supported Formats")
        st.info("📁 **Upload:** MP3, WAV, FLAC, M4A, OGG, AIFF\n🎙️ **Record:** MP3 (High Quality)\n🔄 **Auto-converts** to compatible format")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📁 File Upload Transcription</h3>
            <p>Upload audio files - automatic format conversion included</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_audio = st.file_uploader(
            "Upload an audio file", 
            type=["wav", "mp3", "m4a", "flac", "ogg", "aiff", "aif"],
            help="All formats supported - automatic conversion to compatible format"
        )
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio)
            
            # Show file info
            file_size = len(uploaded_audio.read()) / (1024 * 1024)  # MB
            uploaded_audio.seek(0)  # Reset file pointer
            st.write(f"📊 File: {uploaded_audio.name} ({file_size:.2f} MB)")
            
            if st.button("🚀 Transcribe Uploaded File", type="primary"):
                temp_file_path = None
                try:
                    with st.spinner(f"Processing and transcribing with {selected_api_name}..."):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_audio.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_audio.read())
                            temp_file_path = tmp_file.name
                        
                        # Transcribe
                        result = sr_manager.transcribe_audio_file(
                            temp_file_path, api_name, language_code, asr_pipeline
                        )
                        
                        # Display results
                        display_success_message(f"File transcription completed using {result.get('method', api_name)}!")
                        
                        st.subheader("📝 Transcription Result:")
                        transcript_text = result["text"]
                        
                        if not transcript_text or transcript_text.strip() == "":
                            st.warning("⚠️ No speech detected in the audio file")
                        else:
                            st.text_area("Transcript", transcript_text, height=150)
                            
                            # Show confidence if available
                            if "confidence" in result and result["confidence"] != "N/A":
                                st.write(f"🎯 Confidence: {result['confidence']:.2%}")
                            
                            # Save to history
                            save_transcription_history(
                                transcript_text,
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                f"File Upload ({selected_api_name})",
                                selected_language,
                                result.get("confidence", "N/A"),
                                uploaded_audio.name
                            )
                            
                            # Download options
                            col1a, col1b = st.columns(2)
                            
                            with col1a:
                                file_data = save_transcript_to_file(transcript_text, "transcript", export_format)
                                st.download_button(
                                    f"📥 Download .{export_format.upper()}",
                                    data=file_data,
                                    file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                                    mime=f"text/{export_format}"
                                )
                            
                            with col1b:
                                st.write(f"📊 Words: {len(transcript_text.split())}")
                            
                            # Display timestamps for Whisper
                            if "chunks" in result and result["chunks"]:
                                st.subheader("🕑 Detailed Timestamps:")
                                for chunk in result["chunks"]:
                                    start = chunk.get('timestamp', [None, None])[0]
                                    end = chunk.get('timestamp', [None, None])[1]
                                    text = chunk.get('text', "")
                                    
                                    if start is not None and end is not None:
                                        st.markdown(f"""
                                        <div class="timestamp-box">
                                            <strong>{start:.2f}s - {end:.2f}s</strong> → {text}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="timestamp-box">
                                            <strong>No timestamp</strong> → {text}
                                        </div>
                                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    display_error_message(str(e), "transcription")
                finally:
                    # Clean up uploaded file
                    if temp_file_path:
                        safe_file_cleanup(temp_file_path)
    
    with col2:
        if st.session_state.microphone_available and audio_recorder.is_available():
            st.markdown("""
            <div class="feature-box">
                <h3>🎙️ Record Audio & Transcribe</h3>
                <p>Record high-quality MP3 audio, listen to it, then transcribe</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recording controls
            col2a, col2b, col2c = st.columns(3)
            
            with col2a:
                if not st.session_state.recording:
                    if st.button("🔴 Start Recording", type="primary"):
                        st.session_state.recording = True
                        st.session_state.paused = False
                        # Clear previous recording
                        st.session_state.recorded_audio_file = None
                        st.session_state.recorded_audio_data = None
                        st.rerun()
            
            with col2b:
                if st.session_state.recording:
                    if not st.session_state.paused:
                        if st.button("⏸️ Pause"):
                            st.session_state.paused = True
                            st.rerun()
                    else:
                        if st.button("▶️ Resume"):
                            st.session_state.paused = False
                            st.rerun()
            
            with col2c:
                if st.session_state.recording:
                    if st.button("⏹️ Stop Recording"):
                        st.session_state.recording = False
                        st.session_state.paused = False
                        st.rerun()
            
            # Recording status indicators
            if st.session_state.recording:
                if st.session_state.paused:
                    st.markdown("""
                    <div class="paused-indicator">
                        <h4>⏸️ RECORDING PAUSED</h4>
                        <p>Click Resume to continue recording</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="recording-indicator">
                        <h4>🔴 RECORDING IN PROGRESS</h4>
                        <p>Speak clearly into your microphone...</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Handle recording
            if st.session_state.recording and not st.session_state.paused:
                try:
                    audio_data = audio_recorder.record_audio(recording_duration)
                    st.session_state.recording = False
                    
                    if audio_data is not None:
                        # Save as MP3 file
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        mp3_filename = f"recording_{timestamp}.mp3"
                        
                        # Create temporary file for MP3
                        temp_mp3_path = os.path.join(tempfile.gettempdir(), mp3_filename)
                        
                        try:
                            # Create MP3 bytes
                            mp3_bytes = audio_recorder.save_as_mp3_bytes(audio_data)
                            
                            # Save to temporary file
                            with open(temp_mp3_path, 'wb') as f:
                                f.write(mp3_bytes)
                            
                            # Store the file path and data in session state
                            st.session_state.recorded_audio_file = temp_mp3_path
                            st.session_state.recorded_audio_data = audio_data
                            st.session_state.recorded_audio_bytes = mp3_bytes
                            
                            display_success_message("Recording completed and saved as MP3!")
                            st.rerun()
                        except Exception as e:
                            display_error_message(f"Failed to save recording as MP3: {str(e)}", "audio")
                
                except Exception as e:
                    display_error_message(str(e), "recording")
                    st.session_state.recording = False
                    st.session_state.paused = False
            
            # Display recorded audio if available
            if st.session_state.recorded_audio_file and os.path.exists(st.session_state.recorded_audio_file):
                st.markdown("""
                <div class="recorded-audio-box">
                    <h4>🎵 Recorded Audio</h4>
                    <p>Listen to your recording and then transcribe it</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display audio player
                with open(st.session_state.recorded_audio_file, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')
                
                # Show file info
                file_size = os.path.getsize(st.session_state.recorded_audio_file) / (1024 * 1024)
                st.write(f"📊 Recording: {os.path.basename(st.session_state.recorded_audio_file)} ({file_size:.2f} MB)")
                
                # Transcription controls
                col2d, col2e = st.columns(2)
                
                with col2d:
                    if st.button("🚀 Transcribe Recording", type="primary"):
                        try:
                            with st.spinner(f"Transcribing with {selected_api_name}..."):
                                result = sr_manager.transcribe_audio_file(
                                    st.session_state.recorded_audio_file, 
                                    api_name, 
                                    language_code, 
                                    asr_pipeline
                                )
                                
                                display_success_message(f"Recording transcription completed using {result.get('method', api_name)}!")
                                
                                transcript_text = result["text"]
                                
                                if not transcript_text or transcript_text.strip() == "":
                                    st.warning("⚠️ No speech detected in the recording")
                                else:
                                    st.text_area("Recording Transcript", transcript_text, height=150)
                                    
                                    # Show confidence if available
                                    if "confidence" in result and result["confidence"] != "N/A":
                                        st.write(f"🎯 Confidence: {result['confidence']:.2%}")
                                    
                                    # Save to history
                                    save_transcription_history(
                                        transcript_text,
                                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        f"Live Recording ({selected_api_name})",
                                        selected_language,
                                        result.get("confidence", "N/A"),
                                        os.path.basename(st.session_state.recorded_audio_file)
                                    )
                                    
                                    # Download options
                                    col2f, col2g = st.columns(2)
                                    
                                    with col2f:
                                        file_data = save_transcript_to_file(transcript_text, "recording_transcript", export_format)
                                        st.download_button(
                                            f"📥 Transcript .{export_format.upper()}",
                                            data=file_data,
                                            file_name=f"recording_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                                            mime=f"text/{export_format}"
                                        )
                                    
                                    with col2g:
                                        # Download the MP3 recording
                                        st.download_button(
                                            "📥 Download MP3",
                                            data=audio_bytes,
                                            file_name=f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                                            mime="audio/mp3"
                                        )
                                    
                                    # Display timestamps for Whisper
                                    if "chunks" in result and result["chunks"]:
                                        st.subheader("🕑 Detailed Timestamps:")
                                        for chunk in result["chunks"]:
                                            start = chunk.get('timestamp', [None, None])[0]
                                            end = chunk.get('timestamp', [None, None])[1]
                                            text = chunk.get('text', "")
                                            
                                            if start is not None and end is not None:
                                                st.markdown(f"""
                                                <div class="timestamp-box">
                                                    <strong>{start:.2f}s - {end:.2f}s</strong> → {text}
                                                </div>
                                                """, unsafe_allow_html=True)
                        
                        except Exception as e:
                            display_error_message(str(e), "transcription")
                
                with col2e:
                    if st.button("🗑️ Delete Recording"):
                        # Clean up the recorded file
                        if st.session_state.recorded_audio_file:
                            safe_file_cleanup(st.session_state.recorded_audio_file)
                        st.session_state.recorded_audio_file = None
                        st.session_state.recorded_audio_data = None
                        st.session_state.recorded_audio_bytes = None
                        st.session_state.recorded_audio_filename = None
                        st.rerun()
        
        else:
            # Show alternative when recording is not available
            st.markdown("""
            <div class="feature-box">
                <h3>🎙️ Live Recording Not Available</h3>
                <p>Use file upload instead - works great with phone recordings!</p>
            </div>
            """, unsafe_allow_html=True)
            
            display_info_message("""
            **Recording Tips:**
            • Record audio on your phone or computer
            • Save as MP3, WAV, or M4A format
            • Upload the file using the left panel
            • Works just as well as live recording!
            """)
    
    # Recent transcriptions
    if st.session_state.transcription_history:
        st.divider()
        st.subheader("📋 Recent Transcriptions")
        
        # Show last 5 transcriptions
        recent_transcriptions = st.session_state.transcription_history[-5:]
        
        for i, item in enumerate(reversed(recent_transcriptions)):
            file_info = f" | 📁 {item.get('file_name', 'Live')}" if item.get('file_name') else ""
            with st.expander(f"🎯 {item['method']} - {item['timestamp']} ({item.get('language', 'Unknown')}){file_info}"):
                st.write(f"**Confidence:** {item.get('confidence', 'N/A')}")
                st.write(f"**Words:** {len(item['text'].split())}")
                st.write(f"**Text:** {item['text']}")
                
                # Individual download
                file_data = save_transcript_to_file(item['text'], f"transcript_{i}", export_format)
                st.download_button(
                    f"📥 Download .{export_format.upper()}",
                    data=file_data,
                    file_name=f"transcript_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                    mime=f"text/{export_format}",
                    key=f"download_{i}"
                )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h4>💡 Tips for Better Recognition</h4>
        <p>🎧 Use headphones to avoid feedback | 🔇 Record in a quiet environment | 🗣️ Speak clearly and at normal pace</p>
        <p>🌐 Different APIs work better for different languages | 📱 Phone recordings work great too!</p>
        <p>🎵 High-quality audio for better transcription accuracy | 🎙️ Test microphone before important recordings</p>
        <p><strong>🔧 Troubleshooting:</strong> If recording is silent, check microphone permissions in your browser</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
