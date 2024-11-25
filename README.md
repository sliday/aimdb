# AIMDB – AI-Powered Movie Analyzer 🎬

An intelligent system that analyzes movies through both visual and audio content, generating detailed expert reviews and ratings using GPT-4o-mini for visual analysis and Claude 3.5 Sonnet for screenplay (transcribed via whisper API) evaluation.

![CleanShot 2024-11-26 at 00 19 39@2x](https://github.com/user-attachments/assets/0f04918e-501e-4b45-86c0-288ca62d7efc)

## 🌟 Features

- **Frame Analysis**: Extracts keyframes and creates visual summaries
- **Audio Processing**: Transcribes movie dialogue using OpenAI's Whisper API
- **Expert Panel**: Generates diverse AI expert reviews with unique perspectives
- **Comprehensive Analysis**:
  - Visual aesthetics (GPT-4o-mini)
  - Screenplay quality (Claude 3.5 Sonnet)
  - Narrative structure
  - Technical proficiency
  - Cultural impact
  - And more!
- **IMDB-Style Reviews**: Includes punchy, memorable comments from each expert
- **Beautiful Output**: Colorful, emoji-rich formatting for engaging results

## 🚀 Installation

### Prerequisites

- Python 3.8+ (recommended: Python 3.9-3.11 for best compatibility)
- FFmpeg installed on your system
- OpenAI API key (for GPT-4o-mini and Whisper)
- Anthropic API key (for Claude 3.5 Sonnet)

### Step-by-step installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aimdb.git
cd aimdb
```

2. Create and activate a virtual environment (recommended):
```bash
# For Python 3.9-3.11 (recommended)
python -m venv venv
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
# For Python 3.9-3.11
pip install -r requirements.txt

# For Python 3.13+ (experimental)
pip install --break-system-packages -r requirements.txt
```

4. Download NLTK data (will be done automatically on first run, but you can do it manually):
```python
python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words
```

5. Set up your API keys:
```bash
# On Unix/macOS
export OPENAI_API_KEY='your-openai-api-key'
export ANTHROPIC_API_KEY='your-claude-api-key'

# On Windows
set OPENAI_API_KEY=your-openai-api-key
set ANTHROPIC_API_KEY=your-claude-api-key
```

### Troubleshooting

If you encounter installation issues:

1. Try using Python 3.9, 3.10, or 3.11 for best compatibility
2. Install system dependencies:
```bash
# On Ubuntu/Debian
sudo apt-get install python3-dev ffmpeg

# On macOS
brew install ffmpeg

# On Windows
choco install ffmpeg
```

3. If you're using Python 3.13+, you might need to install from source:
```bash
pip install --no-binary :all: -r requirements.txt
```

## 📋 Prerequisites

- Python 3.8+
- FFmpeg installed on your system
- OpenAI API key (for GPT-4o-mini and Whisper)
- Anthropic API key (for Claude 3.5 Sonnet)
- Sufficient disk space for frame extraction

## 🎮 Usage

Basic usage:
```bash
python main.py movie.mp4
```

Advanced options:
```bash
# Specify number of experts
python main.py movie.mp4 --experts 15

# Skip audio analysis (faster, visual-only review)
python main.py movie.mp4 --no-audio

# Full example with all options
python main.py path/to/movie.mp4 --experts 12 --no-audio
```

### Command-line Arguments

| Argument | Description |
|----------|-------------|
| `movie_path` | Path to the movie file (required) |
| `-e, --experts` | Number of AI experts to employ (default: 10) |
| `--no-audio` | Skip audio analysis for faster, visual-only review |

### Supported Formats
- Video: .mp4, .mkv, .avi, .mov
- Audio: Automatically extracted from video

The system will:
1. Extract keyframes from your movie (2 frames per second)
2. Create 10x10 grid sheets of frames
3. Extract and transcribe audio using Whisper (unless --no-audio is specified)
4. Generate expert reviews using both GPT-4o-mini and Claude 3.5 Sonnet
5. Display detailed analysis and IMDB-style comments

## 📊 Output Structure

The system creates the following directory structure:
```
moviename/
├── frames/         # Individual keyframes
├── sheets/        # 10x10 grid visualizations
└── audio/         # Extracted audio and transcriptions
    └── chunks/    # Audio chunks for processing
```

## 💡 Expert Analysis

Each expert provides:
- Score out of 100 points
- Confidence interval
- IMDB-style comment
- Detailed review
- Category-by-category breakdown:
  - Visual Aesthetics (15 points)
  - Screenplay Quality (15 points)
  - Narrative Structure (15 points)
  - Technical Proficiency (10 points)
  - Innovation (10 points)
  - Cultural Impact (10 points)
  - Audience Appeal (10 points)

## 🎯 Rating System

The final score is calculated by averaging expert ratings, with:
- Confidence intervals
- Genre bonuses
- Tier classification from "Critically Flawed" to "Timeless Masterpiece"

## 🔧 Customization

You can modify:
- `frame_extractor.py`: Adjust frame extraction rate
- `expert_generator.py`: Customize expert profiles
- `rating_system.py`: Modify rating categories and weights

## ⚠️ Notes

- Processing time depends on movie length and number of experts
- API costs will vary based on usage:
  - GPT-4o-mini for visual analysis
  - Claude 3.5 Sonnet for screenplay analysis
  - Whisper for audio transcription
- Ensure sufficient disk space for frame extraction
- Consider API rate limits when processing multiple movies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
```

The updated README provides:
1. Clear installation instructions
2. API key setup requirements
3. Detailed usage guide
4. Explanation of the dual-model approach (GPT-4o-mini + Claude)
5. Output structure and expectations
6. Cost considerations
7. Customization options

Would you like me to add any other specific details to the README?
