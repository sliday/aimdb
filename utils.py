import os
import logging
from colorama import Fore, Style
import emoji

def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('aimdb')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S,%f'
    )
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )
    
    # File handler
    log_file = f"logs/{get_timestamp()}_aimdb.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_timestamp() -> str:
    """Get current timestamp in YYYYMMDD_HHMMSS format."""
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def print_step(step: str, message: str = "", success: bool = True):
    """Print a step in the process with optional message."""
    icon = emoji.emojize(':check_mark_button:') if success else emoji.emojize(':cross_mark:')
    if message:
        print(f"{Fore.GREEN}{step}{Style.RESET_ALL} {message} {icon}")
    else:
        print(f"{Fore.GREEN}{step} {icon}{Style.RESET_ALL}")

def print_error(message: str, *args):
    """Print an error message in red."""
    print(f"{Fore.RED}Error: {message}{Style.RESET_ALL}", *args)

def create_directory(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)