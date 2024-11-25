import os
import logging
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class SheetGenerator:
    def __init__(self):
        self.logger = logging.getLogger('aimdb.sheets')
        self.grid_size = (10, 20)  # 10x20 grid (width x height)
        self.thumb_size = (160, 90)  # 16:9 aspect ratio
        
    def generate_sheets(self, frames_dir: str) -> str:
        """Generate contact sheets from frames."""
        try:
            self.logger.info(f"Generating sheets from frames in {frames_dir}")
            
            # Create sheets directory
            movie_name = os.path.dirname(frames_dir)  # Parent directory is movie name
            sheets_dir = os.path.join(movie_name, "sheets")
            os.makedirs(sheets_dir, exist_ok=True)
            
            # Get frame paths
            frame_paths = sorted([
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.endswith('.jpg')
            ])
            
            if not frame_paths:
                self.logger.error("No frames found in directory")
                return None
                
            # Calculate number of sheets needed
            frames_per_sheet = self.grid_size[0] * self.grid_size[1]  # 200 frames per sheet
            num_sheets = (len(frame_paths) + frames_per_sheet - 1) // frames_per_sheet
            
            self.logger.debug(f"Creating {num_sheets} sheets from {len(frame_paths)} frames")
            self.logger.debug(f"Sheet dimensions: {self.thumb_size[0] * self.grid_size[0]}x{self.thumb_size[1] * self.grid_size[1]} pixels")
            
            # Generate sheets
            with tqdm(total=num_sheets, desc="Generating sheets") as pbar:
                for sheet_num in range(num_sheets):
                    start_idx = sheet_num * frames_per_sheet
                    end_idx = min(start_idx + frames_per_sheet, len(frame_paths))
                    sheet_frames = frame_paths[start_idx:end_idx]
                    
                    # Create sheet
                    sheet = self._create_sheet(sheet_frames)
                    if sheet:
                        sheet_path = os.path.join(sheets_dir, f"sheet_{sheet_num:02d}.jpg")
                        sheet.save(sheet_path, quality=95)
                        self.logger.debug(f"Saved sheet {sheet_num} to {sheet_path}")
                    
                    pbar.update(1)
            
            self.logger.info(f"Generated {num_sheets} sheets in {sheets_dir}")
            return sheets_dir
            
        except Exception as e:
            self.logger.error(f"Sheet generation failed: {str(e)}", exc_info=True)
            return None
            
    def _create_sheet(self, frame_paths: list) -> Image.Image:
        """Create a single contact sheet."""
        try:
            # Calculate sheet size (1600x1800)
            sheet_width = self.thumb_size[0] * self.grid_size[0]   # 160 * 10 = 1600
            sheet_height = self.thumb_size[1] * self.grid_size[1]  # 90 * 20 = 1800
            
            # Create blank sheet
            sheet = Image.new('RGB', (sheet_width, sheet_height), 'black')
            
            # Place thumbnails
            for idx, frame_path in enumerate(frame_paths):
                try:
                    # Calculate position
                    row = idx // self.grid_size[0]  # Integer division by 10 for row
                    col = idx % self.grid_size[0]   # Modulo 10 for column
                    x = col * self.thumb_size[0]    # x = column * 160
                    y = row * self.thumb_size[1]    # y = row * 90
                    
                    # Open and resize frame
                    with Image.open(frame_path) as frame:
                        thumb = frame.resize(self.thumb_size, Image.Resampling.LANCZOS)
                        sheet.paste(thumb, (x, y))
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process frame {frame_path}: {str(e)}")
                    continue
                    
            return sheet
            
        except Exception as e:
            self.logger.error(f"Failed to create sheet: {str(e)}")
            return None 