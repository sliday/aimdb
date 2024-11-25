import dspy
import random
from utils import print_step

class MovieReview(dspy.Signature):
    """Generate a movie review based on visual analysis."""
    sheets_description = dspy.InputField()
    expert_profile = dspy.InputField()
    review = dspy.OutputField(desc="Detailed movie review")
    score = dspy.OutputField(desc="Score out of 10")

class ExpertPanel:
    def __init__(self, lm):
        self.reviewer = dspy.ChainOfThought(MovieReview)
        dspy.configure(lm=lm)
        
    def generate_review(self, sheets_dir, expert_profile):
        """Generate a review from an expert"""
        sheets_description = f"Movie represented by frame sheets in {sheets_dir}"
        
        try:
            result = self.reviewer(
                sheets_description=sheets_description,
                expert_profile=expert_profile
            )
            return result.review, float(result.score)
        except Exception as e:
            print_error(f"Review generation failed: {e}")
            return None, 0.0

def run_expert_panel(sheets_dir, expert_profiles, lm):
    """Run the expert panel review process"""
    panel = ExpertPanel(lm)
    reviews = []
    scores = []
    
    for profile in expert_profiles:
        review, score = panel.generate_review(sheets_dir, profile)
        if review and score:
            reviews.append((profile["name"], review, score))
            scores.append(score)
            print_step("Review", f"Generated review from {profile['name']}")
    
    final_score = sum(scores) / len(scores) if scores else 0
    return reviews, final_score 