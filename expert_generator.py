import dspy
import random
import logging
from typing import List, Dict
from dataclasses import dataclass
import ell

@dataclass
class ExpertProfile:
    name: str
    title: str
    background: str
    style: str
    expertise: List[str]
    
class ExpertGenerator:
    def __init__(self):
        self.logger = logging.getLogger('aimdb.experts')
        
    @ell.simple(model="gpt-4o-mini", max_tokens=100)
    def generate_name(self, expertise: List[str]) -> str:
        """Generate a creative expert name based on their expertise."""
        return f"""Generate a realistic name for a seasoned film critic or academic with expertise in {', '.join(expertise)}.
        The name should sound natural and professional. Return ONLY the name, nothing else."""
        
    @ell.simple(model="gpt-4o-mini", max_tokens=150)
    def generate_title(self, name: str, expertise: List[str]) -> str:
        """Generate a professional title for the expert."""
        return f"""Create a professional title for {name}, who specializes in {', '.join(expertise)}.
        Examples:
        - Film Theory Professor at [University]
        - Senior Film Critic at [Publication]
        - Independent Cinema Researcher
        - Documentary Filmmaker and Critic
        Return ONLY the title, nothing else."""
        
    @ell.simple(model="gpt-4o-mini", max_tokens=200)
    def generate_background(self, name: str, title: str) -> str:
        """Generate a brief background story for the expert."""
        return f"""Create a brief, realistic background for {name}, {title}.
        Include:
        - Years of experience (10-40 years)
        - Notable achievements or publications
        - Known for being [critical trait]
        - Controversial stance on [film topic]
        Keep it to 2-3 sentences, focus on their critical perspective. Return ONLY the background."""
        
    @ell.simple(model="gpt-4o-mini", max_tokens=100)
    def generate_style(self, name: str) -> str:
        """Generate a unique reviewing style for the expert."""
        return f"""Describe {name}'s critical reviewing style in a short phrase.
        Examples:
        - Notoriously harsh on mainstream blockbusters
        - Champions experimental narratives, dismissive of conventional plots
        - Technical purist with strong opinions on digital effects
        - Advocates for traditional filmmaking, skeptical of AI
        Return ONLY the style description, nothing else."""
        
    def generate_expertise_list(self) -> List[str]:
        """Generate a list of film expertise areas."""
        expertise_areas = [
            "Experimental Cinema", "Film Theory", "Visual Effects",
            "Classical Narrative", "Documentary", "Avant-garde Film",
            "Digital Cinema", "Film History", "Screenwriting",
            "Production Design", "Film Technology", "Genre Studies",
            "Cultural Analysis", "Film Philosophy", "Cinema Verite",
            "Post-production", "Sound Design", "Cinematography",
            "Film Economics", "Independent Film", "Studio System",
            "Film Ethics", "Audience Psychology", "Film Education"
        ]
        # Select 2-3 areas of expertise (more focused)
        num_areas = random.randint(2, 3)
        return random.sample(expertise_areas, num_areas)
        
    async def generate_expert(self) -> ExpertProfile:
        """Generate a complete expert profile."""
        try:
            # Generate components
            expertise = self.generate_expertise_list()
            name = self.generate_name(expertise)
            title = self.generate_title(name, expertise)
            background = self.generate_background(name, title)
            style = self.generate_style(name)
            
            # Create profile
            profile = ExpertProfile(
                name=name,
                title=title,
                background=background,
                style=style,
                expertise=expertise
            )
            
            self.logger.info(f"Generated expert profile: {profile.name}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to generate expert profile: {str(e)}")
            # Return a fallback profile
            return ExpertProfile(
                name="A.J. Thompson",
                title="Veteran Film Critic",
                background="30-year veteran known for uncompromising analysis and controversial takes on modern cinema.",
                style="Brutally honest with a focus on technical excellence",
                expertise=["Film Criticism", "Technical Analysis"]
            )
            
    async def generate_expert_profiles(self, num_experts: int) -> List[Dict]:
        """Generate multiple expert profiles."""
        profiles = []
        for _ in range(num_experts):
            profile = await self.generate_expert()
            profiles.append({
                "name": profile.name,
                "title": profile.title,
                "background": profile.background,
                "style": profile.style,
                "expertise": profile.expertise
            })
        return profiles 