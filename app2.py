import streamlit as st
import speech_recognition as sr
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
import base64

# Try to import the audio recorder component
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False
    st.warning("⚠️ audio-recorder-streamlit not installed. Install with: pip install audio-recorder-streamlit")

# Try to import transformers for Whisper
try:
    from transformers import pipeline
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

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
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .feature-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .api-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
        transition: all 0.3s ease;
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
    .api-warning {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .timestamp-box {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        border-left: 3px solid #2196f3;
    }
    .recording-indicator {
        background: linear-gradient(45deg, #ffebee, #ffcdd2);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #f44336;
        text-align: center;
        animation: pulse 2s infinite;
        box-shadow: 0 2px 4px rgba(244, 67, 54, 0.3);
    }
    .paused-indicator {
        background: linear-gradient(45deg, #fff3e0, #ffe0b2);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ff9800;
        text-align: center;
        box-shadow: 0 2px 4px rgba(255, 152, 0, 0.3);
    }
    .recorded-audio-box {
        background: linear-gradient(45deg, #e8f5e8, #c8e6c9);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.3);
    }
    .live-recorder-box {
        background: linear-gradient(45deg, #f3e5f5, #e1bee7);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #9c27b0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(156, 39, 176, 0.3);
    }
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.02); }
        100% { opacity: 1; transform: scale(1); }
    }
    .error-message {
        background: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(244, 67, 54, 0.1);
    }
    .success-message {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.1);
    }
    .info-message {
        background: #e3f2fd;
        color: #1565c0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(33, 150, 243, 0.1);
    }
    .warning-message {
        background: #fff3e0;
        color: #ef6c00;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(255, 152, 0, 0.1);
    }
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .progress-container {
        background: #f0f0f0;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    .audio-quality-indicator {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .quality-excellent {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .quality-good {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .quality-poor {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'recording': False,
        'paused': False,
        'recorded_audio_bytes': None,
        'recorded_audio_filename': None,
        'recorded_audio_file': None,
        'recorded_audio_data': None,
        'transcription_history': [],
        'microphone_available': None,
        'live_audio_bytes': None,
        'audio_quality_score': None,
        'last_transcription': None,
        'api_test_results': {},
        'recording_duration': 0,
        'audio_level': 0
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

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
    "Sphinx (Offline)": "sphinx"
}

if WHISPER_AVAILABLE:
    API_OPTIONS["OpenAI Whisper (Local)"] = "whisper"

@st.cache_resource
def load_asr_model(model_name):
    """Load and cache the ASR model"""
    try:
        if not WHISPER_AVAILABLE:
            st.error("Transformers library not available. Install with: pip install transformers")
            return None
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

def analyze_audio_quality(audio_data, sample_rate=44100):
    """Analyze audio quality and return score and recommendations"""
    try:
        if audio_data is None or len(audio_data) == 0:
            return 0, "No audio data"
        
        # Convert to float32 if needed
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Calculate various audio metrics
        max_amplitude = np.max(np.abs(audio_float))
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Check for clipping
        clipping_ratio = np.sum(np.abs(audio_float) > 0.95) / len(audio_float)
        
        # Check for silence
        silence_threshold = 0.001
        silence_ratio = np.sum(np.abs(audio_float) < silence_threshold) / len(audio_float)
        
        # Calculate signal-to-noise ratio (simplified)
        signal_power = np.mean(audio_float ** 2)
        noise_floor = np.percentile(np.abs(audio_float), 10) ** 2
        snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))
        
        # Quality scoring
        quality_score = 100
        recommendations = []
        
        if max_amplitude < 0.01:
            quality_score -= 40
            recommendations.append("Audio is very quiet - increase microphone volume")
        elif max_amplitude < 0.1:
            quality_score -= 20
            recommendations.append("Audio is quiet - consider speaking louder")
        
        if clipping_ratio > 0.01:
            quality_score -= 30
            recommendations.append("Audio clipping detected - reduce microphone gain")
        
        if silence_ratio > 0.8:
            quality_score -= 30
            recommendations.append("Too much silence detected")
        
        if snr < 10:
            quality_score -= 20
            recommendations.append("High background noise detected")
        
        quality_score = max(0, quality_score)
        
        if quality_score >= 80:
            quality_label = "Excellent"
            quality_class = "quality-excellent"
        elif quality_score >= 60:
            quality_label = "Good"
            quality_class = "quality-good"
        else:
            quality_label = "Poor"
            quality_class = "quality-poor"
        
        return quality_score, quality_label, quality_class, recommendations, {
            'max_amplitude': max_amplitude,
            'rms': rms,
            'clipping_ratio': clipping_ratio,
            'silence_ratio': silence_ratio,
            'snr': snr
        }
        
    except Exception as e:
        return 0, f"Quality analysis failed: {str(e)}", "quality-poor", [], {}

class AudioProcessor:
    """Handle audio file conversion and preprocessing"""
    
    @staticmethod
    def convert_to_wav(input_file, output_file, target_sr=16000):
        """Convert any audio format to PCM WAV format with better error handling"""
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
    
    @staticmethod
    def preprocess_audio_for_transcription(audio_data, sample_rate=16000):
        """Preprocess audio data for better transcription results"""
        try:
            # Convert to float32
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Normalize audio
            max_val = np.max(np.abs(audio_float))
            if max_val > 0:
                audio_float = audio_float / max_val * 0.8  # Normalize to 80% to avoid clipping
            
            # Apply simple noise reduction (high-pass filter)
            from scipy import signal
            b, a = signal.butter(1, 100, btype='high', fs=sample_rate)
            audio_filtered = signal.filtfilt(b, a, audio_float)
            
            return audio_filtered
        except Exception as e:
            st.warning(f"Audio preprocessing failed: {str(e)}, using original audio")
            return audio_data

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
                quality_score, quality_label, quality_class, recommendations, metrics = analyze_audio_quality(audio_data, self.sample_rate)
                
                if quality_score < 30:
                    return False, f"Microphone quality is poor (score: {quality_score}). " + "; ".join(recommendations)
                else:
                    return True, f"Microphone working {quality_label.lower()} (score: {quality_score})"
            else:
                return False, "No audio data recorded"
                
        except Exception as e:
            return False, f"Microphone test failed: {str(e)}"

class SpeechRecognitionManager:
    """Enhanced speech recognition with better error handling and preprocessing"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_processor = AudioProcessor()
        
        # Optimize recognizer settings
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = None
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.8
    
    def transcribe_audio_whisper(self, audio_file, model_pipeline):
        """Transcribe audio file using Whisper with better preprocessing"""
        try:
            # Load audio as numpy array (forces mono, resample to 16kHz)
            audio, sr = librosa.load(audio_file, sr=16000)
            
            # Check if audio is too short
            if len(audio) < 1600:  # Less than 0.1 seconds
                raise Exception("Audio file is too short (less than 0.1 seconds)")
            
            # Preprocess audio
            audio_preprocessed = self.audio_processor.preprocess_audio_for_transcription(audio, 16000)
            
            # Analyze audio quality
            quality_score, quality_label, quality_class, recommendations, metrics = analyze_audio_quality(audio_preprocessed, 16000)
            
            # Check if audio is silent
            if quality_score < 20:
                raise Exception(f"Audio quality is too poor for transcription (score: {quality_score}). Issues: {', '.join(recommendations)}")
            
            # Run the ASR pipeline with timestamps
            result = model_pipeline(audio_preprocessed, return_timestamps=True)
            
            return {
                "text": result["text"].strip(),
                "confidence": quality_score / 100.0,  # Use quality score as confidence
                "chunks": result.get("chunks", []),
                "method": "whisper",
                "quality": quality_label,
                "quality_score": quality_score,
                "recommendations": recommendations
            }
        except Exception as e:
            raise Exception(f"Whisper transcription error: {str(e)}")
    
    def transcribe_audio_bytes(self, audio_bytes, api_name, language="en-US", model_pipeline=None):
        """Transcribe audio from bytes data with enhanced error handling"""
        temp_file_path = None
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            # Use existing transcription method
            result = self.transcribe_audio_file(temp_file_path, api_name, language, model_pipeline)
            return result
            
        except Exception as e:
            raise Exception(f"Transcription from bytes failed: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_file_path:
                safe_file_cleanup(temp_file_path)
    
    def transcribe_audio_file(self, audio_file, api_name, language="en-US", model_pipeline=None):
        """Transcribe audio file using selected API with enhanced preprocessing"""
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
        """Transcribe using speech_recognition library APIs with enhanced error handling"""
        try:
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise with longer duration for better results
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                audio = self.recognizer.record(source)
            
            if not hasattr(audio, 'frame_data') or len(audio.frame_data) == 0:
                raise Exception("No audio data found in file")
            
            # Analyze the recorded audio quality
            try:
                audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
                quality_score, quality_label, quality_class, recommendations, metrics = analyze_audio_quality(audio_data)
                
                if quality_score < 30:
                    st.warning(f"⚠️ Audio quality is {quality_label.lower()} (score: {quality_score}). Recommendations: {', '.join(recommendations)}")
            except:
                quality_score = 50  # Default score if analysis fails
                quality_label = "Unknown"
                recommendations = []
            
            kwargs = {"language": language}
            
            if api_name == "google":
                # Add show_all parameter to get confidence scores
                try:
                    result = self.recognizer.recognize_google(audio, show_all=True, **kwargs)
                    if isinstance(result, dict) and 'alternative' in result:
                        # Extract best result with confidence
                        best_result = result['alternative'][0]
                        text = best_result.get('transcript', '')
                        confidence = best_result.get('confidence', 0.5)
                    else:
                        # Fallback to simple recognition
                        text = self.recognizer.recognize_google(audio, **kwargs)
                        confidence = 0.8
                except:
                    # Fallback to simple recognition
                    text = self.recognizer.recognize_google(audio, **kwargs)
                    confidence = 0.8
                    
            elif api_name == "sphinx":
                lang_code = language.split('-')[0]
                text = self.recognizer.recognize_sphinx(audio, language=lang_code)
                confidence = 0.7  # Sphinx doesn't provide confidence scores
            else:
                raise Exception(f"Unsupported API: {api_name}")
            
            if not text or text.strip() == "":
                raise Exception("No speech detected in audio")
            
            return {
                "text": text.strip(),
                "confidence": confidence,
                "language": language,
                "method": api_name,
                "quality": quality_label,
                "quality_score": quality_score,
                "recommendations": recommendations
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

def save_transcript_to_file(text, filename, file_format, metadata=None):
    """Save transcript to different file formats with metadata"""
    try:
        timestamp = datetime.now().isoformat()
        
        if file_format == "txt":
            content = text
            if metadata:
                header = f"Speech Transcription - {timestamp}\n"
                header += f"Quality: {metadata.get('quality', 'Unknown')}\n"
                header += f"Method: {metadata.get('method', 'Unknown')}\n"
                header += f"Language: {metadata.get('language', 'Unknown')}\n"
                header += "=" * 50 + "\n\n"
                content = header + text
            return content.encode('utf-8')
            
        elif file_format == "json":
            data = {
                "transcript": text,
                "timestamp": timestamp,
                "word_count": len(text.split()),
                "character_count": len(text),
                "metadata": metadata or {}
            }
            return json.dumps(data, indent=2, ensure_ascii=False).encode('utf-8')
            
        elif file_format == "csv":
            df = pd.DataFrame([{
                "transcript": text,
                "timestamp": timestamp,
                "word_count": len(text.split()),
                "character_count": len(text),
                "quality": metadata.get('quality', 'Unknown') if metadata else 'Unknown',
                "method": metadata.get('method', 'Unknown') if metadata else 'Unknown',
                "language": metadata.get('language', 'Unknown') if metadata else 'Unknown',
                "confidence": metadata.get('confidence', 'N/A') if metadata else 'N/A'
            }])
            return df.to_csv(index=False).encode('utf-8')
        else:
            return text.encode('utf-8')
    except Exception as e:
        raise Exception(f"File saving error: {str(e)}")

def save_transcription_history(text, timestamp, method, language=None, confidence=None, file_name=None, quality=None, recommendations=None):
    """Save transcription to history with enhanced metadata"""
    st.session_state.transcription_history.append({
        'text': text,
        'timestamp': timestamp,
        'method': method,
        'language': language or "Unknown",
        'confidence': confidence or "N/A",
        'file_name': file_name,
        'quality': quality or "Unknown",
        'recommendations': recommendations or [],
        'word_count': len(text.split()),
        'character_count': len(text)
    })

def display_error_message(error_msg, error_type="general"):
    """Display formatted error messages with icons"""
    error_icon = "❌"
    if "network" in error_msg.lower() or "connection" in error_msg.lower():
        error_icon = "🌐"
    elif "api" in error_msg.lower() or "key" in error_msg.lower() or "auth" in error_msg.lower():
        error_icon = "🔑"
    elif "audio" in error_msg.lower() or "microphone" in error_msg.lower() or "format" in error_msg.lower():
        error_icon = "🎤"
    elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
        error_icon = "⚠️"
    
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

def display_audio_quality(quality_score, quality_label, quality_class, recommendations):
    """Display audio quality analysis"""
    st.markdown(f"""
    <div class="audio-quality-indicator {quality_class}">
        <strong>🎵 Audio Quality: {quality_label} ({quality_score}/100)</strong>
    </div>
    """, unsafe_allow_html=True)
    
    if recommendations:
        st.markdown("**🔧 Recommendations:**")
        for rec in recommendations:
            st.markdown(f"• {rec}")

def test_api_connection(api_name, model_pipeline=None):
    """Test API connection and functionality"""
    try:
        if api_name == "google":
            # Test with a small silent audio file
            recognizer = sr.Recognizer()
            # Create a small silent audio sample
            duration = 0.1
            sample_rate = 16000
            silent_audio = np.zeros(int(duration * sample_rate), dtype=np.int16)
            
            # Convert to AudioData
            audio_data = sr.AudioData(silent_audio.tobytes(), sample_rate, 2)
            
            try:
                # This should fail but tells us the API is reachable
                recognizer.recognize_google(audio_data, language="en-US")
                return True, "API is reachable"
            except sr.UnknownValueError:
                return True, "API is working (expected response to silent audio)"
            except sr.RequestError as e:
                return False, f"API request failed: {str(e)}"
                
        elif api_name == "sphinx":
            # Sphinx is always available if installed
            try:
                recognizer = sr.Recognizer()
                # Test with silent audio
                duration = 0.1
                sample_rate = 16000
                silent_audio = np.zeros(int(duration * sample_rate), dtype=np.int16)
                audio_data = sr.AudioData(silent_audio.tobytes(), sample_rate, 2)
                
                recognizer.recognize_sphinx(audio_data, language="en")
                return True, "Sphinx is working"
            except sr.UnknownValueError:
                return True, "Sphinx is working (expected response to silent audio)"
            except Exception as e:
                return False, f"Sphinx not available: {str(e)}"
                
        elif api_name == "whisper":
            if model_pipeline is None:
                return False, "Whisper model not loaded"
            
            # Test with a small audio array
            test_audio = np.random.normal(0, 0.01, 16000)  # 1 second of quiet noise
            try:
                result = model_pipeline(test_audio)
                return True, "Whisper model is working"
            except Exception as e:
                return False, f"Whisper test failed: {str(e)}"
        else:
            return False, f"Unknown API: {api_name}"
            
    except Exception as e:
        return False, f"API test failed: {str(e)}"

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
            display_warning_message(f"Recording not available: {mic_message}")
    
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
            if WHISPER_AVAILABLE:
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
                    st.markdown('<div class="api-status api-success">✅ Model loaded successfully!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="api-status api-error">❌ Failed to load model</div>', unsafe_allow_html=True)
                    return
            else:
                st.markdown('<div class="api-status api-error">❌ Transformers library not available</div>', unsafe_allow_html=True)
                display_error_message("Install transformers: pip install transformers torch", "dependency")
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
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test API", use_container_width=True):
                with st.spinner("Testing API..."):
                    is_working, message = test_api_connection(api_name, asr_pipeline)
                    st.session_state.api_test_results[api_name] = (is_working, message)
        
        # Display API status
        if api_name in st.session_state.api_test_results:
            is_working, message = st.session_state.api_test_results[api_name]
            if is_working:
                st.markdown(f'<div class="api-status api-success">✅ {message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="api-status api-error">❌ {message}</div>', unsafe_allow_html=True)
        
        # Microphone testing section
        if st.session_state.microphone_available and audio_recorder.is_available():
            st.subheader("🎤 Microphone Test")
            
            # Show available microphones
            mics = audio_recorder.get_microphone_info()
            if mics:
                st.write("📱 Available microphones:")
                for mic in mics[:3]:  # Show first 3
                    st.write(f"• {mic['name']}")
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("🔍 Test Mic", use_container_width=True):
                    with st.spinner("Testing microphone..."):
                        test_success, test_message = audio_recorder.test_microphone(duration=2)
                        if test_success:
                            st.markdown(f'<div class="api-status api-success">✅ {test_message}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="api-status api-error">❌ {test_message}</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Recording Settings
        if st.session_state.microphone_available:
            st.subheader("🎤 Recording Settings")
            recording_duration = st.slider("Recording Duration (seconds)", 5, 60, 15)
            
            # Audio quality settings
            st.subheader("🎵 Audio Quality")
            st.info("📀 **Format:** WAV (High Quality)\n📊 **Sample Rate:** 44.1kHz\n🎧 **Channels:** Mono\n🔊 **Bit Depth:** 16-bit")
        
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
            
            total_words = sum(item.get('word_count', 0) for item in st.session_state.transcription_history)
            st.write(f"Total words transcribed: {total_words:,}")
            
            if st.button("📥 Export History", use_container_width=True):
                try:
                    # Create comprehensive export
                    export_data = []
                    for item in st.session_state.transcription_history:
                        export_data.append({
                            'timestamp': item['timestamp'],
                            'text': item['text'],
                            'method': item['method'],
                            'language': item.get('language', 'Unknown'),
                            'confidence': item.get('confidence', 'N/A'),
                            'quality': item.get('quality', 'Unknown'),
                            'word_count': item.get('word_count', len(item['text'].split())),
                            'file_name': item.get('file_name', 'Live Recording'),
                            'recommendations': '; '.join(item.get('recommendations', []))
                        })
                    
                    if export_format == "json":
                        export_content = json.dumps(export_data, indent=2, ensure_ascii=False).encode('utf-8')
                    elif export_format == "csv":
                        df = pd.DataFrame(export_data)
                        export_content = df.to_csv(index=False).encode('utf-8')
                    else:  # txt
                        content = "Speech-to-Text Transcription History\n"
                        content += "=" * 50 + "\n\n"
                        for i, item in enumerate(export_data, 1):
                            content += f"Transcription #{i}\n"
                            content += f"Timestamp: {item['timestamp']}\n"
                            content += f"Method: {item['method']}\n"
                            content += f"Language: {item['language']}\n"
                            content += f"Quality: {item['quality']}\n"
                            content += f"Confidence: {item['confidence']}\n"
                            content += f"Words: {item['word_count']}\n"
                            content += f"File: {item['file_name']}\n"
                            if item['recommendations']:
                                content += f"Recommendations: {item['recommendations']}\n"
                            content += f"Text: {item['text']}\n"
                            content += "-" * 30 + "\n\n"
                        export_content = content.encode('utf-8')
                    
                    st.download_button(
                        label=f"📥 Download History .{export_format.upper()}",
                        data=export_content,
                        file_name=f"transcription_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                        mime=f"text/{export_format}",
                        use_container_width=True
                    )
                except Exception as e:
                    display_error_message(f"Export failed: {str(e)}", "export")
            
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.transcription_history = []
                # Clear recorded audio
                for key in ['recorded_audio_bytes', 'recorded_audio_filename', 'recorded_audio_file', 'recorded_audio_data', 'live_audio_bytes']:
                    st.session_state[key] = None
                st.rerun()
        else:
            st.write("No transcriptions yet")
        
        # Audio Format Info
        st.subheader("📋 Supported Formats")
        st.info("📁 **Upload:** MP3, WAV, FLAC, M4A, OGG, AIFF\n🎙️ **Record:** WAV (High Quality)\n🔄 **Auto-converts** to compatible format")
    
    # Main content area - Three columns
    col1, col2, col3 = st.columns(3)
    
    # Column 1: File Upload
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📁 File Upload Transcription</h3>
            <p>Upload audio files with automatic format conversion</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_audio = st.file_uploader(
            "Choose an audio file", 
            type=["wav", "mp3", "m4a", "flac", "ogg", "aiff", "aif"],
            help="All major audio formats supported with automatic conversion"
        )
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio)
            
            # Show file info
            file_size = len(uploaded_audio.read()) / (1024 * 1024)  # MB
            uploaded_audio.seek(0)  # Reset file pointer
            st.write(f"📊 **File:** {uploaded_audio.name}")
            st.write(f"📦 **Size:** {file_size:.2f} MB")
            
            if st.button("🚀 Transcribe File", type="primary", use_container_width=True):
                temp_file_path = None
                try:
                    with st.spinner(f"Processing with {selected_api_name}..."):
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
                        
                        transcript_text = result["text"]
                        
                        if not transcript_text or transcript_text.strip() == "":
                            st.warning("⚠️ No speech detected in the audio file")
                        else:
                            # Display audio quality if available
                            if 'quality_score' in result:
                                display_audio_quality(
                                    result['quality_score'], 
                                    result.get('quality', 'Unknown'), 
                                    f"quality-{result.get('quality', 'unknown').lower()}", 
                                    result.get('recommendations', [])
                                )
                            
                            st.subheader("📝 Transcription Result:")
                            st.text_area("Transcript", transcript_text, height=150, key="file_transcript")
                            
                            # Show metadata
                            col1a, col1b = st.columns(2)
                            
                            with col1a:
                                if "confidence" in result and result["confidence"] != "N/A":
                                    st.metric("🎯 Confidence", f"{result['confidence']:.0%}")
                                st.metric("📊 Words", len(transcript_text.split()))
                            
                            with col1b:
                                if 'quality' in result:
                                    st.metric("🎵 Quality", result['quality'])
                                st.metric("🔤 Characters", len(transcript_text))
                            
                            # Save to history
                            save_transcription_history(
                                transcript_text,
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                f"File Upload ({selected_api_name})",
                                selected_language,
                                result.get("confidence", "N/A"),
                                uploaded_audio.name,
                                result.get('quality', 'Unknown'),
                                result.get('recommendations', [])
                            )
                            
                            # Download options
                            metadata = {
                                'method': f"File Upload ({selected_api_name})",
                                'language': selected_language,
                                'quality': result.get('quality', 'Unknown'),
                                'confidence': result.get('confidence', 'N/A'),
                                'file_name': uploaded_audio.name
                            }
                            
                            file_data = save_transcript_to_file(transcript_text, "file_transcript", export_format, metadata)
                            st.download_button(
                                f"📥 Download .{export_format.upper()}",
                                data=file_data,
                                file_name=f"file_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                                mime=f"text/{export_format}",
                                use_container_width=True
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
                finally:
                    # Clean up uploaded file
                    if temp_file_path:
                        safe_file_cleanup(temp_file_path)
    
    # Column 2: Live Recording with audio-recorder-streamlit
    with col2:
        if AUDIO_RECORDER_AVAILABLE:
            st.markdown("""
            <div class="live-recorder-box">
                <h3>🎙️ Live Audio Recorder</h3>
                <p>One-click recording with instant transcription</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Audio recorder component with enhanced settings
            audio_bytes = audio_recorder(
                text="🎤 Click to Record",
                recording_color="#e74c3c",
                neutral_color="#6c757d",
                icon_name="microphone",
                icon_size="2x",
                pause_threshold=1.5,
                sample_rate=44100,
                key="live_audio_recorder"
            )
            
            # Handle recorded audio
            if audio_bytes:
                # Check if this is new audio (different from last recording)
                if st.session_state.live_audio_bytes != audio_bytes:
                    st.session_state.live_audio_bytes = audio_bytes
                    
                    st.markdown("""
                    <div class="recorded-audio-box">
                        <h4>🎵 Live Recording Complete</h4>
                        <p>Audio captured successfully - ready for transcription!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display audio player
                    st.audio(audio_bytes, format='audio/wav')
                    
                    # Analyze audio quality
                    try:
                        # Convert bytes to numpy array for quality analysis
                        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_file.flush()
                            
                            audio_data, sample_rate = librosa.load(tmp_file.name, sr=None)
                            audio_int16 = (audio_data * 32767).astype(np.int16)
                            
                            quality_score, quality_label, quality_class, recommendations, metrics = analyze_audio_quality(audio_int16, sample_rate)
                            
                            # Display quality analysis
                            display_audio_quality(quality_score, quality_label, quality_class, recommendations)
                            
                            # Show detailed metrics
                            col2a, col2b = st.columns(2)
                            with col2a:
                                st.write(f"📊 **Size:** {len(audio_bytes) / 1024:.1f} KB")
                                st.write(f"🎵 **Quality:** {quality_label}")
                            with col2b:
                                st.write(f"⏱️ **Duration:** ~{len(audio_data)/sample_rate:.1f}s")
                                st.write(f"🔊 **Score:** {quality_score}/100")
                    
                    except Exception as e:
                        st.warning(f"Quality analysis failed: {str(e)}")
                    
                    # Transcription button
                    if st.button("🚀 Transcribe Live Recording", type="primary", use_container_width=True, key="transcribe_live"):
                        try:
                            with st.spinner(f"Transcribing with {selected_api_name}..."):
                                result = sr_manager.transcribe_audio_bytes(
                                    audio_bytes, api_name, language_code, asr_pipeline
                                )
                                
                                display_success_message(f"Live recording transcription completed using {result.get('method', api_name)}!")
                                
                                transcript_text = result["text"]
                                
                                if not transcript_text or transcript_text.strip() == "":
                                    st.warning("⚠️ No speech detected in the live recording")
                                else:
                                    st.text_area("Live Recording Transcript", transcript_text, height=150, key="live_transcript")
                                    
                                    # Show metadata
                                    col2c, col2d = st.columns(2)
                                    
                                    with col2c:
                                        if "confidence" in result and result["confidence"] != "N/A":
                                            st.metric("🎯 Confidence", f"{result['confidence']:.0%}")
                                        st.metric("📊 Words", len(transcript_text.split()))
                                    
                                    with col2d:
                                        if 'quality' in result:
                                            st.metric("🎵 Quality", result['quality'])
                                        st.metric("🔤 Characters", len(transcript_text))
                                    
                                    # Save to history
                                    save_transcription_history(
                                        transcript_text,
                                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        f"Live Recording ({selected_api_name})",
                                        selected_language,
                                        result.get("confidence", "N/A"),
                                        "Live Recording",
                                        result.get('quality', 'Unknown'),
                                        result.get('recommendations', [])
                                    )
                                    
                                    # Download options
                                    col2e, col2f = st.columns(2)
                                    
                                    with col2e:
                                        metadata = {
                                            'method': f"Live Recording ({selected_api_name})",
                                            'language': selected_language,
                                            'quality': result.get('quality', 'Unknown'),
                                            'confidence': result.get('confidence', 'N/A')
                                        }
                                        file_data = save_transcript_to_file(transcript_text, "live_transcript", export_format, metadata)
                                        st.download_button(
                                            f"📄 Transcript .{export_format.upper()}",
                                            data=file_data,
                                            file_name=f"live_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                                            mime=f"text/{export_format}",
                                            key="download_live_transcript",
                                            use_container_width=True
                                        )
                                    
                                    with col2f:
                                        # Download the audio recording
                                        st.download_button(
                                            "🎵 Audio WAV",
                                            data=audio_bytes,
                                            file_name=f"live_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                                            mime="audio/wav",
                                            key="download_live_audio",
                                            use_container_width=True
                                        )
                        
                        except Exception as e:
                            display_error_message(str(e), "transcription")
                    
                    # Clear recording button
                    if st.button("🗑️ Clear Recording", use_container_width=True, key="clear_live"):
                        st.session_state.live_audio_bytes = None
                        st.rerun()
        
        else:
            # Show installation instructions when audio-recorder-streamlit is not available
            st.markdown("""
            <div class="feature-box">
                <h3>🎙️ Live Recording</h3>
                <p>One-click recording component</p>
            </div>
            """, unsafe_allow_html=True)
            
            display_info_message("""
            **To enable live recording:**
            
            ```bash
            pip install audio-recorder-streamlit
            ```
            
            Then restart the app for instant recording functionality.
            """)
    
    # Column 3: PyAudio Recording (Traditional)
    with col3:
        if st.session_state.microphone_available and audio_recorder.is_available():
            st.markdown("""
            <div class="feature-box">
                <h3>🎤 Advanced Recording</h3>
                <p>Professional recording with duration control</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recording controls
            col3a, col3b, col3c = st.columns(3)
            
            with col3a:
                if not st.session_state.recording:
                    if st.button("🔴 Record", type="primary", use_container_width=True, key="pyaudio_start"):
                        st.session_state.recording = True
                        st.session_state.paused = False
                        # Clear previous recording
                        for key in ['recorded_audio_file', 'recorded_audio_data', 'recorded_audio_bytes']:
                            st.session_state[key] = None
                        st.rerun()
            
            with col3b:
                if st.session_state.recording:
                    if not st.session_state.paused:
                        if st.button("⏸️ Pause", use_container_width=True, key="pyaudio_pause"):
                            st.session_state.paused = True
                            st.rerun()
                    else:
                        if st.button("▶️ Resume", use_container_width=True, key="pyaudio_resume"):
                            st.session_state.paused = False
                            st.rerun()
            
            with col3c:
                if st.session_state.recording:
                    if st.button("⏹️ Stop", use_container_width=True, key="pyaudio_stop"):
                        st.session_state.recording = False
                        st.session_state.paused = False
                        st.rerun()
            
            # Show recording duration setting
            if not st.session_state.recording:
                st.write(f"⏱️ Duration: {recording_duration} seconds")
            
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
                        <h4>🔴 RECORDING ACTIVE</h4>
                        <p>Speak clearly into your microphone...</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Handle recording process
            if st.session_state.recording and not st.session_state.paused:
                try:
                    # Import PyAudio for recording
                    import pyaudio
                    
                    # Recording parameters
                    sample_rate = 44100
                    chunk_size = 1024
                    format = pyaudio.paInt16
                    channels = 1
                    
                    # Initialize PyAudio
                    p = pyaudio.PyAudio()
                    
                    # Open stream
                    stream = p.open(
                        format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size
                    )
                    
                    frames = []
                    total_frames = int(sample_rate / chunk_size * recording_duration)
                    
                    # Progress tracking
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
                            data = stream.read(chunk_size, exception_on_overflow=False)
                            frames.append(data)
                            
                            # Update progress
                            progress = (i + 1) / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"Recording... {progress:.0%}")
                            
                        except Exception as e:
                            st.warning(f"Audio read error: {str(e)}")
                            break
                    
                    # Clean up
                    progress_bar.empty()
                    status_text.empty()
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    
                    # Stop recording
                    st.session_state.recording = False
                    
                    if frames:
                        # Convert to numpy array
                        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
                        
                        # Analyze audio quality
                        quality_score, quality_label, quality_class, recommendations, metrics = analyze_audio_quality(audio_data, sample_rate)
                        
                        # Check quality
                        if quality_score < 30:
                            display_warning_message(f"Recording quality is {quality_label.lower()} (score: {quality_score}). Consider re-recording.")
                        
                        # Convert to bytes for storage and playback
                        wav_buffer = io.BytesIO()
                        with wave.open(wav_buffer, 'wb') as wav_file:
                            wav_file.setnchannels(channels)
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(sample_rate)
                            wav_file.writeframes(audio_data.tobytes())
                        
                        wav_bytes = wav_buffer.getvalue()
                        
                        # Store in session state
                        st.session_state.recorded_audio_data = audio_data
                        st.session_state.recorded_audio_bytes = wav_bytes
                        
                        display_success_message("Recording completed successfully!")
                        st.rerun()
                    else:
                        display_error_message("No audio data recorded", "recording")
                
                except Exception as e:
                    display_error_message(f"Recording error: {str(e)}", "recording")
                    st.session_state.recording = False
                    st.session_state.paused = False
            
            # Display recorded audio if available
            if st.session_state.recorded_audio_bytes is not None:
                st.markdown("""
                <div class="recorded-audio-box">
                    <h4>🎵 Professional Recording</h4>
                    <p>High-quality audio ready for transcription</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display audio player
                st.audio(st.session_state.recorded_audio_bytes, format='audio/wav')
                
                # Analyze and display quality
                if st.session_state.recorded_audio_data is not None:
                    quality_score, quality_label, quality_class, recommendations, metrics = analyze_audio_quality(st.session_state.recorded_audio_data)
                    
                    # Display quality analysis
                    display_audio_quality(quality_score, quality_label, quality_class, recommendations)
                
                # Show file info
                file_size = len(st.session_state.recorded_audio_bytes) / (1024 * 1024)
                st.write(f"📦 **Size:** {file_size:.2f} MB")
                st.write(f"⏱️ **Duration:** ~{recording_duration} seconds")
                
                # Transcription controls
                col3d, col3e = st.columns(2)
                
                with col3d:
                    if st.button("🚀 Transcribe", type="primary", use_container_width=True, key="transcribe_pyaudio"):
                        try:
                            with st.spinner(f"Transcribing with {selected_api_name}..."):
                                result = sr_manager.transcribe_audio_bytes(
                                    st.session_state.recorded_audio_bytes, 
                                    api_name, 
                                    language_code, 
                                    asr_pipeline
                                )
                                
                                display_success_message(f"Professional recording transcription completed using {result.get('method', api_name)}!")
                                
                                transcript_text = result["text"]
                                
                                if not transcript_text or transcript_text.strip() == "":
                                    st.warning("⚠️ No speech detected in the recording")
                                else:
                                    st.text_area("Professional Recording Transcript", transcript_text, height=150, key="pyaudio_transcript")
                                    
                                    # Show metadata
                                    col3f, col3g = st.columns(2)
                                    
                                    with col3f:
                                        if "confidence" in result and result["confidence"] != "N/A":
                                            st.metric("🎯 Confidence", f"{result['confidence']:.0%}")
                                        st.metric("📊 Words", len(transcript_text.split()))
                                    
                                    with col3g:
                                        if 'quality' in result:
                                            st.metric("🎵 Quality", result['quality'])
                                        st.metric("🔤 Characters", len(transcript_text))
                                    
                                    # Save to history
                                    save_transcription_history(
                                        transcript_text,
                                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        f"Professional Recording ({selected_api_name})",
                                        selected_language,
                                        result.get("confidence", "N/A"),
                                        "Professional Recording",
                                        result.get('quality', 'Unknown'),
                                        result.get('recommendations', [])
                                    )
                                    
                                    # Download options
                                    col3h, col3i = st.columns(2)
                                    
                                    with col3h:
                                        metadata = {
                                            'method': f"Professional Recording ({selected_api_name})",
                                            'language': selected_language,
                                            'quality': result.get('quality', 'Unknown'),
                                            'confidence': result.get('confidence', 'N/A')
                                        }
                                        file_data = save_transcript_to_file(transcript_text, "professional_transcript", export_format, metadata)
                                        st.download_button(
                                            f"📄 Transcript .{export_format.upper()}",
                                            data=file_data,
                                            file_name=f"professional_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                                            mime=f"text/{export_format}",
                                            key="download_pyaudio_transcript",
                                            use_container_width=True
                                        )
                                    
                                    with col3i:
                                        # Download the audio recording
                                        st.download_button(
                                            "🎵 Audio WAV",
                                            data=st.session_state.recorded_audio_bytes,
                                            file_name=f"professional_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                                            mime="audio/wav",
                                            key="download_pyaudio_audio",
                                            use_container_width=True
                                        )
                        
                        except Exception as e:
                            display_error_message(str(e), "transcription")
                
                with col3e:
                    if st.button("🗑️ Delete", use_container_width=True, key="delete_pyaudio"):
                        # Clear the recorded audio
                        for key in ['recorded_audio_data', 'recorded_audio_bytes']:
                            st.session_state[key] = None
                        st.rerun()
        
        else:
            # Show alternative when recording is not available
            st.markdown("""
            <div class="feature-box">
                <h3>🎤 Professional Recording</h3>
                <p>Advanced recording features</p>
            </div>
            """, unsafe_allow_html=True)
            
            display_info_message("""
            **Professional Recording Tips:**
            
            • Record with external microphone for best quality
            • Use quiet environment to minimize background noise
            • Speak clearly at consistent volume
            • Test microphone before important recordings
            
            **Alternative:** Use the live recorder in the middle column or upload pre-recorded files.
            """)
    
    # Recent transcriptions section
    if st.session_state.transcription_history:
        st.divider()
        st.subheader("📋 Recent Transcriptions")
        
        # Show statistics
        total_transcriptions = len(st.session_state.transcription_history)
        total_words = sum(item.get('word_count', 0) for item in st.session_state.transcription_history)
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("📝 Total Transcriptions", total_transcriptions)
        
        with col_stat2:
            st.metric("📊 Total Words", f"{total_words:,}")
        
        with col_stat3:
            avg_quality = []
            for item in st.session_state.transcription_history:
                if item.get('quality') in ['Excellent', 'Good', 'Poor']:
                    quality_score = {'Excellent': 90, 'Good': 70, 'Poor': 40}.get(item['quality'], 50)
                    avg_quality.append(quality_score)
            
            if avg_quality:
                st.metric("🎵 Avg Quality", f"{np.mean(avg_quality):.0f}/100")
            else:
                st.metric("🎵 Avg Quality", "N/A")
        
        with col_stat4:
            recent_methods = [item['method'] for item in st.session_state.transcription_history[-5:]]
            most_common = max(set(recent_methods), key=recent_methods.count) if recent_methods else "None"
            st.metric("🔌 Most Used", most_common.split('(')[0].strip())
        
        # Show last 5 transcriptions
        recent_transcriptions = st.session_state.transcription_history[-5:]
        
        for i, item in enumerate(reversed(recent_transcriptions)):
            file_info = f" | 📁 {item.get('file_name', 'Live')}" if item.get('file_name') else ""
            quality_info = f" | 🎵 {item.get('quality', 'Unknown')}" if item.get('quality') else ""
            
            with st.expander(f"🎯 {item['method']} - {item['timestamp']} ({item.get('language', 'Unknown')}){file_info}{quality_info}"):
                # Metadata
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                
                with col_meta1:
                    st.write(f"**Confidence:** {item.get('confidence', 'N/A')}")
                    st.write(f"**Words:** {item.get('word_count', len(item['text'].split()))}")
                
                with col_meta2:
                    st.write(f"**Quality:** {item.get('quality', 'Unknown')}")
                    st.write(f"**Characters:** {item.get('character_count', len(item['text']))}")
                
                with col_meta3:
                    st.write(f"**Language:** {item.get('language', 'Unknown')}")
                    st.write(f"**Method:** {item['method']}")
                
                # Recommendations if available
                if item.get('recommendations'):
                    st.write("**🔧 Recommendations:**")
                    for rec in item['recommendations']:
                        st.write(f"• {rec}")
                
                # Transcript text
                st.write(f"**📝 Text:** {item['text']}")
                
                # Individual download
                metadata = {
                    'method': item['method'],
                    'language': item.get('language', 'Unknown'),
                    'quality': item.get('quality', 'Unknown'),
                    'confidence': item.get('confidence', 'N/A'),
                    'timestamp': item['timestamp']
                }
                
                file_data = save_transcript_to_file(item['text'], f"transcript_{i}", export_format, metadata)
                st.download_button(
                    f"📥 Download .{export_format.upper()}",
                    data=file_data,
                    file_name=f"transcript_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                    mime=f"text/{export_format}",
                    key=f"download_history_{i}",
                    use_container_width=True
                )
    
    # Footer with tips and troubleshooting
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <h4>💡 Tips for Better Transcription Results</h4>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div>
                <h5>🎤 Recording Quality</h5>
                <p>• Use external microphone when possible<br>
                • Record in quiet environment<br>
                • Speak clearly at normal pace<br>
                • Maintain consistent distance from mic</p>
            </div>
            
            <div>
                <h5>🔧 Technical Tips</h5>
                <p>• Check microphone permissions in browser<br>
                • Ensure stable internet for online APIs<br>
                • Use WAV format for best quality<br>
                • Test audio levels before recording</p>
            </div>
            
            <div>
                <h5>🌍 Language & API Selection</h5>
                <p>• Match language setting to spoken language<br>
                • Google API: Best for online, multiple languages<br>
                • Whisper: Great for offline, high accuracy<br>
                • Sphinx: Basic offline functionality</p>
            </div>
        </div>
        
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #ddd;">
            <h5>🚨 Troubleshooting Common Issues</h5>
            <p><strong>No audio detected:</strong> Check microphone permissions and volume levels<br>
            <strong>Poor transcription quality:</strong> Improve audio quality, reduce background noise<br>
            <strong>API errors:</strong> Check internet connection and API quotas<br>
            <strong>Live recording not working:</strong> Install audio-recorder-streamlit package</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
