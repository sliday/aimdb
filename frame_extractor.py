import cv2
import os
import logging
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

class FrameExtractor:
    def __init__(self):
        self.logger = logging.getLogger('aimdb.frames')
        
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.error("FFmpeg not found. Please install FFmpeg.")
            return False
            
    def _parse_fraction(self, fraction_str: str) -> float:
        """Parse a string fraction (e.g., '25/1') to float."""
        try:
            if '/' in fraction_str:
                num, den = map(float, fraction_str.split('/'))
                return num / den if den != 0 else 0
            return float(fraction_str)
        except (ValueError, ZeroDivisionError) as e:
            self.logger.warning(f"Failed to parse fraction '{fraction_str}': {str(e)}")
            return 0
            
    def _get_video_info(self, video_path: str) -> dict:
        """Get video information using FFmpeg."""
        try:
            # First, get duration using ffprobe
            duration_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
            duration = float(duration_result.stdout.strip())
            
            # Then get video stream info
            stream_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            stream_result = subprocess.run(stream_cmd, capture_output=True, text=True, check=True)
            info = stream_result.stdout.strip().split('\n')
            
            # Parse frame rate
            fps = self._parse_fraction(info[2])
            if fps == 0:
                self.logger.warning("Could not determine FPS, using default of 25")
                fps = 25
                
            # Calculate total frames
            total_frames = int(duration * fps)
            
            return {
                'width': int(info[0]),
                'height': int(info[1]),
                'duration': duration,
                'total_frames': total_frames,
                'fps': fps
            }
        except subprocess.SubprocessError as e:
            self.logger.error(f"Failed to get video info: {str(e)}")
            return None
        except (ValueError, IndexError) as e:
            self.logger.error(f"Failed to parse video info: {str(e)}")
            return None
            
    def extract_frames(self, movie_path: str, fps: int = 2) -> str:
        """Extract frames from video file at specified FPS."""
        try:
            if not self._check_ffmpeg():
                return None
                
            self.logger.info(f"Extracting frames from {movie_path}")
            movie_name = os.path.splitext(os.path.basename(movie_path))[0]
            frames_dir = f"{movie_name}/frames"
            os.makedirs(frames_dir, exist_ok=True)
            
            # Get video info
            video_info = self._get_video_info(movie_path)
            if not video_info:
                return None
                
            # Calculate total frames to extract
            duration = video_info['duration']
            total_frames = int(duration * fps)
            self.logger.info(f"Video duration: {duration:.2f}s, extracting ~{total_frames} frames at {fps} fps")
            
            # Extract frames using FFmpeg
            cmd = [
                'ffmpeg',
                '-i', movie_path,
                '-vf', f'fps={fps}',
                '-frame_pts', '1',
                '-vsync', '0',
                os.path.join(frames_dir, 'frame_%04d.jpg')
            ]
            
            # Run FFmpeg with progress monitoring
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor progress
            with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                while True:
                    line = process.stderr.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        # Update progress bar (approximate)
                        pbar.update(1)
            
            if process.returncode != 0:
                self.logger.error("FFmpeg frame extraction failed")
                return None
            
            # Count extracted frames
            extracted_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            self.logger.info(f"Extracted {extracted_frames} frames to {frames_dir}")
            
            return frames_dir
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {str(e)}", exc_info=True)
            return None
            
    def get_frame_paths(self, frames_dir: str) -> list:
        """Get sorted list of frame paths."""
        try:
            frame_paths = sorted([
                os.path.join(frames_dir, f) 
                for f in os.listdir(frames_dir) 
                if f.endswith('.jpg')
            ])
            self.logger.debug(f"Found {len(frame_paths)} frames in {frames_dir}")
            return frame_paths
        except Exception as e:
            self.logger.error(f"Failed to get frame paths: {str(e)}")
            return []