import subprocess
from pathlib import Path
import os
from openai import OpenAI
from utils import print_step, print_error
import wave
import simpleaudio as sa
from tqdm import tqdm
import logging

class AudioExtractor:
    def __init__(self):
        self.client = OpenAI()
        self.logger = logging.getLogger('aimdb.audio')
        
    def extract_audio(self, movie_path: str) -> str:
        """Extract audio from video file"""
        self.logger.info(f"Extracting audio from {movie_path}")
        movie_name = os.path.splitext(os.path.basename(movie_path))[0]
        audio_dir = f"{movie_name}/audio"
        os.makedirs(audio_dir, exist_ok=True)
        
        # Extract to WAV instead of MP3 for better compatibility
        audio_path = f"{audio_dir}/audio.wav"
        try:
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-i', movie_path,
                '-acodec', 'pcm_s16le',  # Use PCM format
                '-ac', '1',  # Convert to mono
                '-ar', '16000',  # 16kHz sample rate
                '-y',  # Overwrite output file if it exists
                audio_path
            ]
            self.logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg with detailed output capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Log progress
            self.logger.info("FFmpeg process started")
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"FFmpeg failed with return code {process.returncode}")
                self.logger.error(f"FFmpeg stderr: {stderr}")
                return None
                
            if os.path.exists(audio_path):
                self.logger.info(f"Audio extracted successfully to {audio_path}")
                self.logger.debug(f"Audio file size: {os.path.getsize(audio_path)} bytes")
                return audio_path
            else:
                self.logger.error(f"Audio file not created at {audio_path}")
                return None
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg subprocess error: {str(e)}")
            self.logger.error(f"FFmpeg stderr: {e.stderr.decode() if e.stderr else 'No stderr'}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during audio extraction: {str(e)}", exc_info=True)
            return None
            
    def split_audio(self, audio_path: str, chunk_duration: int = 10) -> list:
        """Split audio into chunks (Whisper API has a 25MB limit)"""
        self.logger.info(f"Splitting audio file: {audio_path}")
        try:
            chunks_dir = os.path.join(os.path.dirname(audio_path), "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            
            # Read the wave file
            self.logger.debug("Opening wave file")
            with wave.open(audio_path, 'rb') as wav_file:
                self.logger.debug(f"Wave file parameters: {wav_file.getparams()}")
                frames = wav_file.readframes(wav_file.getnframes())
                params = wav_file.getparams()
                
                # Calculate frames per chunk
                frames_per_chunk = int(params.framerate * chunk_duration)
                chunk_size = frames_per_chunk * params.sampwidth * params.nchannels
                total_chunks = len(frames) // chunk_size + (1 if len(frames) % chunk_size else 0)
                
                self.logger.info(f"Splitting into approximately {total_chunks} chunks")
                
                chunk_paths = []
                for i, pos in enumerate(range(0, len(frames), chunk_size)):
                    chunk_frames = frames[pos:pos + chunk_size]
                    if not chunk_frames:
                        break
                        
                    chunk_path = f"{chunks_dir}/chunk_{i}.wav"
                    self.logger.debug(f"Writing chunk {i} to {chunk_path}")
                    with wave.open(chunk_path, 'wb') as chunk_file:
                        chunk_file.setparams(params)
                        chunk_file.writeframes(chunk_frames)
                    chunk_paths.append(chunk_path)
                    self.logger.debug(f"Chunk {i} written successfully")
                    
            self.logger.info(f"Successfully split audio into {len(chunk_paths)} chunks")
            return chunk_paths
        except Exception as e:
            self.logger.error(f"Audio splitting failed: {str(e)}", exc_info=True)
            return []
            
    async def transcribe_audio(self, audio_paths: list) -> dict:
        """Transcribe audio chunks using OpenAI Whisper API"""
        self.logger.info(f"Starting transcription of {len(audio_paths)} chunks")
        try:
            transcriptions = []
            
            with tqdm(total=len(audio_paths), desc="Transcribing chunks") as pbar:
                for chunk_path in audio_paths:
                    self.logger.debug(f"Transcribing chunk: {chunk_path}")
                    with open(chunk_path, "rb") as audio_file:
                        # Call Whisper API (synchronous)
                        self.logger.debug("Calling Whisper API")
                        response = self.client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="json"
                        )
                        
                        transcriptions.append(response)
                        self.logger.debug(f"Chunk transcribed: {os.path.basename(chunk_path)}")
                        pbar.update(1)
            
            # Combine transcriptions
            combined = {
                "text": " ".join(t.text for t in transcriptions)
            }
            
            # Save transcription to file
            audio_dir = os.path.dirname(os.path.dirname(audio_paths[0]))  # Go up from chunks dir
            transcription_path = os.path.join(audio_dir, "transcription.txt")
            self.logger.info(f"Saving transcription to {transcription_path}")
            with open(transcription_path, "w", encoding="utf-8") as f:
                f.write(combined["text"])
            
            self.logger.info("Transcription completed successfully")
            self.logger.debug(f"Total transcribed text length: {len(combined['text'])}")
            return combined
            
        except Exception as e:
            self.logger.error("Transcription failed", exc_info=True)
            return None 