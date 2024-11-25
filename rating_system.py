from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from utils import print_step, print_error
import emoji
from colorama import Fore, Style, Back
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import logging
import ell
import os
import random
import re

@dataclass
class CategoryScore:
    score: float
    justification: str

@dataclass
class ExpertRating:
    # Visual aspects
    visual_aesthetics: CategoryScore
    technical_proficiency: CategoryScore
    
    # Narrative aspects
    screenplay_quality: CategoryScore
    narrative_structure: CategoryScore
    
    # Additional categories
    innovation: CategoryScore
    cultural_significance: CategoryScore
    audience_engagement: CategoryScore
    
    # Overall scores
    total_score: float
    confidence_interval: Tuple[float, float]
    genre_bonus: float
    tier: str
    
    # Reviews
    detailed_review: str
    imdb_comment: str

class MovieRatingSystem:
    TIERS = {
        (95, 100): "Timeless Masterpiece",
        (90, 94): "Exceptional",
        (85, 89): "Outstanding",
        (80, 84): "Excellent",
        (75, 79): "Very Good",
        (70, 74): "Good",
        (65, 69): "Above Average",
        (60, 64): "Average",
        (55, 59): "Below Average",
        (50, 54): "Mediocre",
        (40, 49): "Poor",
        (30, 39): "Very Poor",
        (0, 29): "Critically Flawed"
    }

    @staticmethod
    def get_tier(score: float) -> str:
        for (min_score, max_score), tier in MovieRatingSystem.TIERS.items():
            if min_score <= score <= max_score:
                return tier
        return "Unrated"

    @staticmethod
    def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        mean = np.mean(scores)
        std = np.std(scores)
        z = 1.96  # 95% confidence interval
        margin = z * (std / np.sqrt(len(scores)))
        return (mean - margin, mean + margin)

class TextAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger('aimdb.text')
        
    def analyze_text(self, transcription: dict) -> dict:
        """Analyze transcribed text using simple text analysis"""
        if not transcription or "text" not in transcription:
            self.logger.warning("No text provided for analysis")
            return self._get_empty_analysis()
            
        text = transcription["text"]
        
        # Simple text analysis without NLTK dependencies
        sentences = text.split('. ')
        words = text.split()
        
        # Simple dialogue detection
        dialogue_markers = self._analyze_dialogue_markers(text)
        
        # Basic text analysis
        analysis = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "dialogue_markers": dialogue_markers,
            "dialogue_ratio": dialogue_markers["dialogue_ratio"]
        }
        
        self.logger.debug("Text analysis completed successfully")
        return analysis
        
    def _analyze_dialogue_markers(self, text: str) -> dict:
        """Analyze dialogue-specific patterns using simple approach"""
        try:
            sentences = text.split('. ')
            dialogue_count = sum(1 for s in sentences if '"' in s or "'" in s)
            
            markers = {
                "dialogue_sentences": dialogue_count,
                "dialogue_ratio": dialogue_count / len(sentences) if sentences else 0
            }
            return markers
        except Exception as e:
            self.logger.error(f"Dialogue analysis failed: {str(e)}")
            return {
                "dialogue_sentences": 0,
                "dialogue_ratio": 0
            }
            
    def _get_empty_analysis(self) -> dict:
        """Return empty analysis structure"""
        return {
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "dialogue_markers": {"dialogue_sentences": 0, "dialogue_ratio": 0},
            "dialogue_ratio": 0
        }

@ell.simple(model="gpt-4o-mini", max_tokens=10)
def evaluate_category_prompt(category: str, max_points: int, expert_profile: dict, sheets_dir: str) -> str:
    """You are an expert film critic. Be concise."""
    # Get list of sheet files
    sheet_files = sorted([f for f in os.listdir(sheets_dir) if f.startswith('sheet_') and f.endswith('.jpg')])
    sheets_info = f"Analyzing {len(sheet_files)} grid sheets of movie frames"
    
    return f"""As {expert_profile['name']}, {expert_profile['background']}, 
    evaluate the {category} of the movie based on its visual representation in frame grids.

    Important Context:
    - You are looking at {len(sheet_files)} grid sheets, each containing 100 frames (10x10)
    - Each frame is a snapshot from the movie, taken every 0.5 seconds
    - Frames are arranged chronologically from top-left to bottom-right
    - These grids collectively represent the movie's visual flow and progression
    
    Consider your {expert_profile['specialization']} and {expert_profile['perspective']}.
    Analyze these frame grids to provide:
    1. A score out of {max_points} points
    2. A brief, focused justification based on what you observe in the frame grids"""

@ell.simple(model="gpt-4o-mini", max_tokens=500)
def analyze_screenplay_prompt(analysis: dict) -> str:
    """You are a professional screenplay analyst. Be concise."""
    return f"""Evaluate the following script elements concisely:

    Text Statistics:
    - Word count: {analysis.get('word_count', 0)}
    - Average sentence length: {analysis.get('avg_sentence_length', 0):.1f}
    - Dialogue percentage: {analysis.get('dialogue_ratio', 0) * 100:.1f}%
    
    Provide a brief analysis of:
    1. Dialogue quality and authenticity
    2. Character development through dialogue
    3. Plot progression and pacing
    4. Overall screenplay craftsmanship
    
    Score out of 15 points with brief justification."""

def format_expert_review(expert: dict, scores: Dict[str, float]) -> str:
    """Format individual expert review with stars."""
    # Calculate expert's score (out of 5 stars)
    total = sum(scores.values())
    max_total = len(scores) * 30  # Each category is out of 30
    star_score = (total / max_total) * 5
    
    # Generate star rating
    full_stars = int(star_score)
    half_star = (star_score - full_stars) >= 0.5
    stars = "⭐" * full_stars + ("✨" if half_star else "")
    
    # Format the review
    return f"""
{Fore.CYAN}👤 {expert['name']}{Style.RESET_ALL} ({expert['title']})
{Fore.YELLOW}{stars}{Style.RESET_ALL} ({star_score:.1f}/5)
{Fore.BLUE}Expertise:{Style.RESET_ALL} {', '.join(expert['expertise'])}
{Fore.GREEN}Style:{Style.RESET_ALL} {expert['style']}
"""

def format_review_output(review: dict, expert_profiles: list) -> str:
    """Format review output with colors and emoji."""
    # Get movie title
    title = review['movie_name']
    if len(title) > 50:  # Truncate long titles
        title = title[:47] + "..."
    
    # Parse scores
    visual_score = review['scores']['visual_quality']
    screenplay_score = review['scores']['screenplay_quality']
    total_score = (visual_score + screenplay_score) / 2  # Average of both scores
    
    # Convert to 100-point scale
    total_score_100 = (total_score / 30) * 100
    
    # Calculate confidence interval
    confidence_range = [total_score_100 - 3, total_score_100 + 3]  # Simple ±3 range
    
    # Generate star rating (out of 10)
    score_out_of_10 = total_score_100 / 10
    full_stars = int(score_out_of_10)
    half_star = (score_out_of_10 - full_stars) >= 0.5
    star_rating = "⭐" * full_stars + ("✨" if half_star else "")
    
    output = []
    
    # Header
    output.extend([
        "",
        f"{Back.BLACK}{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}",
        f"{Back.BLACK}{Fore.YELLOW}🎬 Film Analysis Report{Style.RESET_ALL}",
        f"{Back.BLACK}{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}",
        "",
        f"{Fore.CYAN}Title:{Style.RESET_ALL} {title}",
        ""
    ])
    
    # Technical Assessment
    output.extend([
        f"{Fore.CYAN}Technical Assessment:{Style.RESET_ALL}",
        f"{'─' * 40}",
        f"📊 Visual Quality:      {visual_score:>5.1f}/30 {get_score_emoji(visual_score, 30)}",
        f"📊 Screenplay Quality:  {screenplay_score:>5.1f}/30 {get_score_emoji(screenplay_score, 30)}",
        f"{'─' * 40}",
        f"📊 Overall Score:       {total_score_100:>5.1f}/100 {get_final_rating_emoji(total_score_100/100)}",
        f"🎯 Confidence Range:    [{confidence_range[0]:.1f} - {confidence_range[1]:.1f}]",
        f"🏆 Final Rating:        {get_rating_tier(total_score_100/100)}",
        ""
    ])
    
    # Expert Panel
    output.extend([
        f"{Fore.CYAN}Expert Panel:{Style.RESET_ALL}",
        f"{'─' * 40}"
    ])
    
    for expert in expert_profiles:
        output.extend([
            f"{Fore.YELLOW}👤 {expert['name']}{Style.RESET_ALL}",
            f"   {Fore.BLUE}Position:{Style.RESET_ALL} {expert['title']}",
            f"   {Fore.BLUE}Expertise:{Style.RESET_ALL} {', '.join(expert['expertise'])}",
            f"   {Fore.BLUE}Style:{Style.RESET_ALL} {expert['style']}",
            ""
        ])
    
    # Parse sections from the review text
    review_text = review['review']
    sections = parse_review_sections(review_text)
    
    # Detailed Analysis
    output.extend([
        f"{Fore.CYAN}Detailed Analysis:{Style.RESET_ALL}",
        f"{'─' * 40}"
    ])
    
    if "Key Strengths" in sections:
        output.extend([
            f"{Fore.GREEN}💪 Key Strengths:{Style.RESET_ALL}"
        ])
        strengths = parse_bullet_points(sections["Key Strengths"])
        for strength in strengths:
            output.append(f"   ✓ {strength}")
        output.append("")
    
    if "Areas for Improvement" in sections:
        output.extend([
            f"{Fore.YELLOW}⚠️ Critical Issues:{Style.RESET_ALL}"
        ])
        improvements = parse_bullet_points(sections["Areas for Improvement"])
        for improvement in improvements:
            output.append(f"   • {improvement}")
        output.append("")
    
    # Final Verdict
    if "Final Verdict" in sections:
        output.extend([
            f"{Fore.CYAN}Final Verdict:{Style.RESET_ALL}",
            f"{'─' * 40}"
        ])
        verdict = sections["Final Verdict"].strip()
        # Wrap long lines
        verdict_lines = [verdict[i:i+80] for i in range(0, len(verdict), 80)]
        output.extend([f"   {line}" for line in verdict_lines])
    
    # Footer
    output.extend([
        "",
        f"{Back.BLACK}{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}",
        f"{Back.BLACK}{Fore.YELLOW}End of Analysis{Style.RESET_ALL}",
        f"{Back.BLACK}{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}",
        ""
    ])
    
    return "\n".join(output)

def get_score_emoji(score: float, max_score: float) -> str:
    """Get appropriate emoji based on score percentage."""
    percentage = score / max_score
    if percentage >= 0.8:
        return "📷"  # Visual
    elif percentage >= 0.6:
        return "📝"  # Screenplay
    elif percentage >= 0.4:
        return "📚"  # Story
    elif percentage >= 0.2:
        return "⚙️"  # Technical
    else:
        return "⚠️"  # Warning

def get_final_rating_emoji(percentage: float) -> str:
    """Get final rating emoji based on score percentage."""
    if percentage >= 0.9:
        return "🏆🏆🏆"  # Masterpiece
    elif percentage >= 0.8:
        return "🏆🏆"    # Excellent
    elif percentage >= 0.7:
        return "🏆"      # Very Good
    elif percentage >= 0.6:
        return "⭐"      # Good
    elif percentage >= 0.5:
        return "✨"      # Average
    elif percentage >= 0.4:
        return "💫"      # Below Average
    else:
        return "⚠️"      # Poor

def get_rating_tier(percentage: float) -> str:
    """Get rating tier based on score percentage."""
    if percentage >= 0.9:
        return f"{Fore.YELLOW}Masterpiece{Style.RESET_ALL}"
    elif percentage >= 0.8:
        return f"{Fore.GREEN}Excellent{Style.RESET_ALL}"
    elif percentage >= 0.7:
        return f"{Fore.GREEN}Very Good{Style.RESET_ALL}"
    elif percentage >= 0.6:
        return f"{Fore.CYAN}Good{Style.RESET_ALL}"
    elif percentage >= 0.5:
        return f"{Fore.BLUE}Average{Style.RESET_ALL}"
    elif percentage >= 0.4:
        return f"{Fore.YELLOW}Below Average{Style.RESET_ALL}"
    else:
        return f"{Fore.RED}Poor{Style.RESET_ALL}"

class ExpertReviewSystem:
    def __init__(self, expert_profiles: list):
        self.logger = logging.getLogger('aimdb.rating')
        self.expert_profiles = expert_profiles
        self.text_analyzer = TextAnalyzer()
        
    @ell.simple(model="gpt-4o-mini", max_tokens=300)
    def analyze_visual_prompt(self, sheets_dir: str, expert_profile: dict) -> str:
        """Generate visual analysis prompt."""
        return f"""As {expert_profile['name']}, a {expert_profile['title']} specializing in {', '.join(expert_profile['expertise'])},
        critically analyze the visual elements of this movie based on the frame grids provided.
        
        Your Style: {expert_profile['style']}
        
        Context: You are examining 10x20 grids of frames from the movie, where each frame is a snapshot taken every 0.5 seconds,
        arranged chronologically. Each sheet shows 200 frames in a 1600x1800 pixel grid.
        
        Analyze with your usual critical eye:
        1. Visual Composition & Framing
        2. Technical Execution & Effects
        3. Production Design & Aesthetics
        4. Visual Storytelling & Continuity
        5. Innovation vs. Convention
        
        First, provide a numerical score out of 30 points in the format "Score: X/30".
        Then, provide your detailed critique, highlighting both strengths and weaknesses.
        Be honest and direct - your reputation depends on it."""
        
    def _parse_score(self, response: str, max_points: int) -> float:
        """Parse score from response text, fallback to 50% if parsing fails."""
        try:
            # First, look for explicit score format
            score_pattern = r"(?i)score:\s*(\d+(?:\.\d+)?)/30"
            match = re.search(score_pattern, response)
            if match:
                score = float(match.group(1))
                return min(max(0, score), max_points)
                
            # Try finding any number followed by /30
            score_pattern = r"(\d+(?:\.\d+)?)/30"
            match = re.search(score_pattern, response)
            if match:
                score = float(match.group(1))
                return min(max(0, score), max_points)
                
            # Try finding any number between 0 and 30 in the first few words
            first_line = response.split('\n')[0]
            number_pattern = r"(\d+(?:\.\d+)?)"
            matches = re.findall(number_pattern, first_line)
            for match in matches:
                score = float(match)
                if 0 <= score <= max_points:
                    return score
                    
            # If no valid score found, use 50%
            default_score = max_points * 0.5
            self.logger.warning(f"Could not find valid score in response, using default ({default_score}/{max_points})")
            return default_score
            
        except Exception as e:
            self.logger.warning(f"Score parsing failed: {str(e)}, using default")
            return max_points * 0.5
            
    async def analyze_visuals(self, sheets_dir: str) -> CategoryScore:
        """Analyze visual elements of the movie."""
        try:
            # Randomly select an expert for visual analysis
            expert = random.choice(self.expert_profiles)
            
            # Get visual analysis
            response = self.analyze_visual_prompt(sheets_dir, expert)
            
            # Parse score with max 30 points
            score = self._parse_score(response, max_points=30)
            
            return CategoryScore(score=score, justification=response)
            
        except Exception as e:
            self.logger.error(f"Visual analysis failed: {str(e)}")
            return CategoryScore(score=15, justification="Visual analysis failed")  # 50% of 30
            
    @ell.simple(model="gpt-4o-mini", max_tokens=300)
    def analyze_screenplay_prompt(self, transcription: dict) -> str:
        """Generate screenplay analysis prompt."""
        return f"""Analyze the following movie transcription with a critical eye.
        
        Focus your critique on:
        1. Dialogue Quality & Authenticity
        2. Narrative Structure & Coherence
        3. Character Development & Depth
        4. Thematic Exploration
        5. Overall Writing Craft
        
        First, provide a numerical score out of 30 points in the format "Score: X/30".
        Then, analyze the following text, being brutally honest about both merits and flaws:
        
        {transcription['text'][:1000]}..."""
        
    async def analyze_screenplay(self, transcription: dict) -> CategoryScore:
        """Analyze screenplay based on transcription."""
        try:
            if not transcription:
                return CategoryScore(score=0, justification="No transcription available")
                
            # Get screenplay analysis
            response = self.analyze_screenplay_prompt(transcription)
            
            # Parse score with max 30 points
            score = self._parse_score(response, max_points=30)
            
            return CategoryScore(score=score, justification=response)
            
        except Exception as e:
            self.logger.error(f"Screenplay analysis failed: {str(e)}")
            return CategoryScore(score=15, justification="Screenplay analysis failed")  # 50% of 30
            
    async def generate_expert_review(self, movie_name: str, sheets_dir: str, transcription: dict = None) -> dict:
        """Generate a comprehensive expert review."""
        try:
            # Initialize scores dictionary
            scores = {}
            
            # Analyze visual elements
            scores["visual_quality"] = await self.analyze_visuals(sheets_dir)
            
            # Analyze audio/dialogue if available
            if transcription:
                scores["screenplay_quality"] = await self.analyze_screenplay(transcription)
            
            # Generate final review
            review = self.generate_final_review_prompt(movie_name, scores)
            
            # Format the result
            result = {
                "movie_name": movie_name,
                "review": review,
                "scores": {
                    "visual_quality": scores["visual_quality"].score,
                    "screenplay_quality": scores.get("screenplay_quality", CategoryScore(0, "N/A")).score
                }
            }
            
            # Print formatted review
            print(format_review_output(result, self.expert_profiles))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate expert review: {str(e)}", exc_info=True)
            return None
            
    @ell.simple(model="gpt-4o-mini", max_tokens=500)
    def generate_final_review_prompt(self, movie_name: str, scores: Dict[str, CategoryScore]) -> str:
        """Generate final review prompt."""
        expert = random.choice(self.expert_profiles)
        
        return f"""As {expert['name']}, provide an uncompromising review of "{movie_name}".
        
        Your Background: {expert['background']}
        Your Style: {expert['style']}
        
        Scores:
        - Visual Quality: {scores['visual_quality'].score}/30
        - Screenplay: {scores.get('screenplay_quality', CategoryScore(0, 'N/A')).score}/30
        
        Structure your critique as follows:
        
        Overall Rating: [Score]/100
        
        Key Strengths:
        - [If any, be specific and justify]
        
        Critical Issues:
        - [Be direct and unsparing in your analysis]
        
        Final Verdict:
        [2-3 paragraphs of honest, professional criticism. Don't sugar-coat, but remain constructive.
        Your reputation as {expert['title']} is built on candid, insightful analysis.]"""
            
    def _parse_score(self, response: str, max_points: int) -> float:
        """Parse score from response text, fallback to 50% if parsing fails."""
        try:
            # First, look for explicit score format
            score_pattern = r"(?i)score:\s*(\d+(?:\.\d+)?)/30"
            match = re.search(score_pattern, response)
            if match:
                score = float(match.group(1))
                return min(max(0, score), max_points)
                
            # Try finding any number followed by /30
            score_pattern = r"(\d+(?:\.\d+)?)/30"
            match = re.search(score_pattern, response)
            if match:
                score = float(match.group(1))
                return min(max(0, score), max_points)
                
            # Try finding any number between 0 and 30 in the first few words
            first_line = response.split('\n')[0]
            number_pattern = r"(\d+(?:\.\d+)?)"
            matches = re.findall(number_pattern, first_line)
            for match in matches:
                score = float(match)
                if 0 <= score <= max_points:
                    return score
                    
            # If no valid score found, use 50%
            default_score = max_points * 0.5
            self.logger.warning(f"Could not find valid score in response, using default ({default_score}/{max_points})")
            return default_score
            
        except Exception as e:
            self.logger.warning(f"Score parsing failed: {str(e)}, using default")
            return max_points * 0.5

def parse_review_sections(review_text: str) -> dict:
    """Parse review text into sections."""
    sections = {}
    current_section = None
    current_text = []
    
    for line in review_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if line.startswith('**') and line.endswith(':**'):
            if current_section:
                sections[current_section] = '\n'.join(current_text)
            current_section = line.strip('*:')
            current_text = []
        else:
            current_text.append(line)
    
    if current_section:
        sections[current_section] = '\n'.join(current_text)
    
    return sections

def parse_bullet_points(text: str) -> list:
    """Parse bullet points from text."""
    points = []
    current_point = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for numbered points or bullet points
        if line[0].isdigit() and line[1] == '.' or line.startswith('•'):
            if current_point:
                points.append(' '.join(current_point))
            current_point = [line.split('.', 1)[-1].strip()]
        else:
            current_point.append(line)
    
    if current_point:
        points.append(' '.join(current_point))
    
    return points

def format_final_summary(expert_ratings: List[Tuple[str, ExpertRating]]) -> str:
    """Format the final summary with all expert scores"""
    total_score = np.mean([r.total_score for _, r in expert_ratings])
    
    summary = f"""
{Fore.YELLOW}╔══════════════════════════════════════════════════════════════╗{Style.RESET_ALL}
{Fore.YELLOW}║                     Final Movie Rating                       ║{Style.RESET_ALL}
{Fore.YELLOW}╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.CYAN}Overall Score: {total_score/10:.1f}/10 {emoji.emojize(':star:') * int(total_score/10)}{Style.RESET_ALL}

{Fore.GREEN}Expert Opinions:{Style.RESET_ALL}
"""
    
    for name, rating in expert_ratings:
        score_out_of_10 = rating.total_score / 10
        stars = emoji.emojize(':star:') * int(score_out_of_10)
        summary += f"{Fore.BLUE}{name}{Style.RESET_ALL}: {score_out_of_10:.1f}/10 {stars}\n"
        summary += f"{Fore.CYAN}└─{Style.RESET_ALL} \"{rating.imdb_comment}\"\n\n"
    
    return summary 