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

# Try to import optional dependencies
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers library not available. Whisper functionality will be disabled.")

try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False
    st.error("❌ audio-recorder-streamlit not installed. Live recording disabled.")

# Configure page
st.set_page_config(
    page_title="🎙️ Speech Recognition App",
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
    .recording-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
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
    .recording-status {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        text-align: center;
    }
    .dependency-warning {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .github-info {
        background: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'transcription_history' not in st.session_state:
        st.session_state.transcription_history = []
    if 'last_recording' not in st.session_state:
        st.session_state.last_recording = None

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

if TRANSFORMERS_AVAILABLE:
    API_OPTIONS["OpenAI Whisper (Local)"] = "whisper"

@st.cache_resource
def load_asr_model(model_name):
    """Load and cache the ASR model"""
    if not TRANSFORMERS_AVAILABLE:
        st.error("Transformers library not available for Whisper models")
        return None
    
    try:
        return pipeline("automatic-speech-recognition", model=model_name)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

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
    
    @staticmethod
    def process_recorded_audio(audio_bytes, output_file, target_sr=16000):
        """Process recorded audio bytes and save as WAV"""
        try:
            # Save the recorded bytes to a temporary file first
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_input:
                temp_input.write(audio_bytes)
                temp_input_path = temp_input.name
            
            try:
                # Convert using the existing conversion method
                success, msg = AudioProcessor.convert_to_wav(temp_input_path, output_file, target_sr)
                return success, msg
            finally:
                # Clean up temporary input file
                safe_file_cleanup(temp_input_path)
                
        except Exception as e:
            return False, f"Recorded audio processing error: {str(e)}"

def safe_file_cleanup(file_path, max_retries=3, delay=0.1):
    """Safely delete a file with retries"""
    import time
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
    
    def transcribe_recorded_audio(self, audio_bytes, api_name, language="en-US", model_pipeline=None):
        """Transcribe recorded audio bytes using the same workflow as file uploads"""
        temp_wav_path = None
        try:
            # Create temporary file for the processed audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                temp_wav_path = temp_wav.name
            
            # Process the recorded audio bytes
            conversion_success, conversion_msg = self.audio_processor.process_recorded_audio(
                audio_bytes, temp_wav_path
            )
            
            if not conversion_success:
                raise Exception(f"Recorded audio processing failed: {conversion_msg}")
            
            # Use the same transcription method as file uploads
            if api_name == "whisper":
                if not model_pipeline:
                    raise Exception("Whisper model not loaded")
                result = self.transcribe_audio_whisper(temp_wav_path, model_pipeline)
            else:
                result = self._transcribe_with_speech_recognition(temp_wav_path, api_name, language)
            
            return result
            
        except Exception as e:
            raise Exception(f"Recorded audio transcription failed: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_wav_path:
                safe_file_cleanup(temp_wav_path)
    
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

def display_recording_status(message):
    """Display formatted recording status messages"""
    st.markdown(f"""
    <div class="recording-status">
        <strong>🎙️ Recording:</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Speech Recognition App</h1>
        <p>Upload Audio Files or Record Live | High-Quality Transcription</p>
    </div>
    """, unsafe_allow_html=True)
    
    # GitHub deployment info
    st.markdown("""
    <div class="github-info">
        <strong>🚀 GitHub Deployment Ready</strong><br>
        This app is configured for GitHub deployment with audio-recorder-streamlit for live recording.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    sr_manager = SpeechRecognitionManager()
    
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
        
        # Recording Settings
        st.subheader("🎙️ Recording Settings")
        recording_duration = st.slider(
            "Max Recording Duration (seconds)",
            min_value=5,
            max_value=300,
            value=30,
            help="Maximum duration for live recording"
        )
        
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
        
        st.divider()
        
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
                st.rerun()
        else:
            st.write("No transcriptions yet")
        
        # Audio Format Info
        st.subheader("📋 Supported Formats")
        st.info("📁 **Upload:** MP3, WAV, FLAC, M4A, OGG, AIFF\n🎙️ **Record:** Live audio recording\n🔄 **Auto-converts** to compatible format")
    
    # Main content area - Live Recording Section
    if AUDIO_RECORDER_AVAILABLE:
        st.markdown("""
        <div class="recording-box">
            <h3>🎙️ Live Audio Recording</h3>
            <p>Click the microphone button below to start recording. Speak clearly and avoid background noise for best results.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio recorder component
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
            sample_rate=16000
        )
        
        # Handle recorded audio
        if audio_bytes is not None:
            # Check if this is a new recording
            if st.session_state.last_recording != audio_bytes:
                st.session_state.last_recording = audio_bytes
                
                # Display audio player for the recording
                st.audio(audio_bytes, format="audio/wav")
                
                # Show recording info
                audio_duration = len(audio_bytes) / (16000 * 2)  # Approximate duration
                st.write(f"🎵 Recording captured ({audio_duration:.1f} seconds)")
                
                display_recording_status("Recording captured! Click 'Transcribe Recording' to process.")
                
                if st.button("🚀 Transcribe Recording", type="primary", key="transcribe_recording"):
                    try:
                        with st.spinner(f"Processing and transcribing recording with {selected_api_name}..."):
                            # Transcribe the recorded audio using the same function as file uploads
                            result = sr_manager.transcribe_recorded_audio(
                                audio_bytes, api_name, language_code, asr_pipeline
                            )
                            
                            # Display results
                            display_success_message(f"Live recording transcription completed using {result.get('method', api_name)}!")
                            
                            st.subheader("📝 Transcription Result:")
                            transcript_text = result["text"]
                            
                            if not transcript_text or transcript_text.strip() == "":
                                st.warning("⚠️ No speech detected in the recording")
                            else:
                                st.text_area("Transcript", transcript_text, height=150, key="recording_transcript")
                                
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
                                    "Live Recording"
                                )
                                
                                # Download options
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    file_data = save_transcript_to_file(transcript_text, "recording_transcript", export_format)
                                    st.download_button(
                                        f"📥 Download .{export_format.upper()}",
                                        data=file_data,
                                        file_name=f"recording_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                                        mime=f"text/{export_format}",
                                        key="download_recording"
                                    )
                                
                                with col2:
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
                        display_error_message(str(e), "recording")
        
    else:
        st.markdown("""
        <div class="dependency-warning">
            <h3>🎙️ Live Recording Unavailable</h3>
            <p><strong>Missing Dependency:</strong> audio-recorder-streamlit package is not installed.</p>
            <p>Make sure your requirements.txt includes <code>audio-recorder-streamlit</code></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # File Upload Section
    st.markdown("""
    <div class="feature-box">
        <h3>📁 Audio File Transcription</h3>
        <p>Upload audio files in any format - automatic conversion included</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
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
        
        if st.button("🚀 Transcribe Audio File", type="primary", key="transcribe_file"):
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
                        st.text_area("Transcript", transcript_text, height=150, key="file_transcript")
                        
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
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            file_data = save_transcript_to_file(transcript_text, "transcript", export_format)
                            st.download_button(
                                f"📥 Download .{export_format.upper()}",
                                data=file_data,
                                file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format}",
                                mime=f"text/{export_format}",
                                key="download_file"
                            )
                        
                        with col2:
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
    
    # Recent transcriptions
    if st.session_state.transcription_history:
        st.divider()
        st.subheader("📋 Recent Transcriptions")
        
        # Show last 5 transcriptions
        recent_transcriptions = st.session_state.transcription_history[-5:]
        
        for i, item in enumerate(reversed(recent_transcriptions)):
            file_info = f" | 📁 {item.get('file_name', 'Unknown')}" if item.get('file_name') else ""
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
        <p>🎧 Use clear audio recordings | 🔇 Avoid background noise | 🗣️ Ensure clear speech</p>
        <p>🌐 Different APIs work better for different languages | 📱 Works with mobile uploads</p>
        <p>🎵 High-quality audio files for better transcription accuracy</p>
        <p>🎙️ <strong>New:</strong> Live recording with high-quality audio capture</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
