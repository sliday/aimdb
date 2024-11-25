import argparse
import asyncio
import logging
import os
from pathlib import Path
from frame_extractor import FrameExtractor
from sheet_generator import SheetGenerator
from expert_generator import ExpertGenerator
from rating_system import ExpertReviewSystem
from audio_extractor import AudioExtractor
from utils import setup_logging, print_step, print_error

def check_files_exist(movie_name: str) -> tuple:
    """Check if processed files already exist."""
    frames_dir = f"{movie_name}/frames"
    sheets_dir = f"{movie_name}/sheets"
    audio_path = f"{movie_name}/audio/audio.wav"
    transcription_path = f"{movie_name}/audio/transcription.txt"
    
    has_frames = os.path.isdir(frames_dir) and any(f.endswith('.jpg') for f in os.listdir(frames_dir))
    has_sheets = os.path.isdir(sheets_dir) and any(f.endswith('.jpg') for f in os.listdir(sheets_dir))
    has_audio = os.path.isfile(audio_path)
    has_transcription = os.path.isfile(transcription_path)
    
    return frames_dir, sheets_dir, audio_path, transcription_path, has_frames, has_sheets, has_audio, has_transcription

async def process_movie(movie_path: str, expert_profiles: list, review_system: ExpertReviewSystem):
    """Process a movie file and generate expert reviews."""
    try:
        movie_name = os.path.splitext(os.path.basename(movie_path))[0]
        print_step("[Start] Processing movie:", movie_path)
        
        # Check existing files
        frames_dir, sheets_dir, audio_path, transcription_path, has_frames, has_sheets, has_audio, has_transcription = check_files_exist(movie_name)
        
        # Extract frames if needed
        if has_frames:
            print_step("[Extraction] Using existing frames from", frames_dir)
        else:
            frame_extractor = FrameExtractor()
            frames_dir = frame_extractor.extract_frames(movie_path)
            if not frames_dir:
                raise Exception("Frame extraction failed")
            print_step("[Extraction] Frames extracted to", frames_dir)
        
        # Generate sheets if needed
        if has_sheets:
            print_step("[Sheet Creation] Using existing sheets from", sheets_dir)
        else:
            sheet_generator = SheetGenerator()
            sheets_dir = sheet_generator.generate_sheets(frames_dir)
            if not sheets_dir:
                raise Exception("Sheet generation failed")
            print_step("[Sheet Creation] Generated sheets")
        
        # Process audio if available and needed
        transcription = None
        if not args.no_audio:
            audio_extractor = AudioExtractor()
            
            # Extract audio if needed
            if has_audio:
                print_step("[Audio] Using existing audio from", audio_path)
            else:
                audio_path = audio_extractor.extract_audio(movie_path)
                if audio_path:
                    print_step("[Audio Extraction] Audio extracted to", audio_path)
            
            # Use existing transcription or create new one
            if has_transcription:
                print_step("[Transcription] Using existing transcription")
                with open(transcription_path, 'r', encoding='utf-8') as f:
                    transcription = {"text": f.read()}
            elif audio_path:
                # Split audio into chunks
                chunks = audio_extractor.split_audio(audio_path)
                if chunks:
                    print_step("[Audio Splitting] Split audio into", len(chunks), "chunks")
                    
                    # Transcribe audio
                    transcription = await audio_extractor.transcribe_audio(chunks)
                    if transcription:
                        print_step("[Audio Transcription] Successfully transcribed audio")
        
        # Generate expert reviews
        rating = await review_system.generate_expert_review(
            movie_name=movie_name,
            sheets_dir=sheets_dir,
            transcription=transcription
        )
        
        if rating:
            print_step("[Review] Expert review generated successfully")
            print("\nFinal Rating:", rating)
        
    except Exception as e:
        logger.error("Analysis failed", exc_info=True)
        print_error("Analysis failed:", str(e))

async def main():
    """Main entry point."""
    try:
        logger.info("Starting AIMDB movie analysis")
        logger.info(f"Processing movie: {args.movie_path}")
        
        # Generate expert profiles
        expert_gen = ExpertGenerator()
        expert_profiles = await expert_gen.generate_expert_profiles(args.experts)
        logger.info(f"Generated {len(expert_profiles)} expert profiles")
        
        # Initialize review system
        ranking_system = ExpertReviewSystem(expert_profiles)
        
        # Process the movie
        await process_movie(args.movie_path, expert_profiles, ranking_system)
        
    except Exception as e:
        logger.error("Analysis failed", exc_info=True)
        print_error("Analysis failed:", str(e))

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="AI Movie Database - Expert Review System")
    parser.add_argument("movie_path", help="Path to the movie file")
    parser.add_argument("-e", "--experts", type=int, default=10, help="Number of AI experts to employ")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio analysis")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Run main
    asyncio.run(main()) 