"""
EPPC Code Evaluator Module
Comprehensive evaluation framework for EPPC codes based on Khaled Saab's research approach
Provides multi-dimensional scoring (presence, intensity, quality) for each code
"""
from typing import Dict, List, Optional, Tuple
import re
from collections import Counter

class EPPCEvaluator:
    """Evaluates EPPC codes with multi-dimensional scoring"""
    
    def __init__(self):
        self.evaluation_criteria = self._initialize_evaluation_criteria()
    
    def _initialize_evaluation_criteria(self) -> Dict:
        """Initialize code-specific evaluation criteria"""
        return {
            "Partnership": {
                "indicators": [
                    "collaborative language", "expressing preferences", "active participation",
                    "mutual respect", "shared responsibility"
                ],
                "scoring_dimensions": {
                    "presence": 0.3, "intensity": 0.4, "quality": 0.3
                },
                "quality_markers": [
                    "we", "together", "let's", "collaborate", "partnership",
                    "prefer", "opinion", "value", "input", "involve"
                ]
            },
            "Emotional Support": {
                "indicators": [
                    "emotional expression", "empathy markers", "appreciation",
                    "concern acknowledgment", "reassurance"
                ],
                "scoring_dimensions": {
                    "presence": 0.25, "intensity": 0.35, "quality": 0.4
                },
                "quality_markers": [
                    "understand", "concern", "worry", "anxious", "appreciate",
                    "thank", "grateful", "support", "care", "feel"
                ]
            },
            "Information-Giving": {
                "indicators": [
                    "factual information", "explanations", "educational content",
                    "clarifications", "guidance"
                ],
                "scoring_dimensions": {
                    "presence": 0.4, "intensity": 0.3, "quality": 0.3
                },
                "quality_markers": [
                    "explain", "information", "about", "means", "because",
                    "reason", "how", "what", "when", "why"
                ]
            },
            "Information-Seeking": {
                "indicators": [
                    "questions", "inquiries", "clarification requests",
                    "information requests", "seeking guidance"
                ],
                "scoring_dimensions": {
                    "presence": 0.4, "intensity": 0.3, "quality": 0.3
                },
                "quality_markers": [
                    "question", "ask", "wonder", "curious", "want to know",
                    "can you", "what", "how", "why", "when"
                ]
            },
            "Shared Decision-Making": {
                "indicators": [
                    "discussing options", "weighing choices", "mutual agreement",
                    "collaborative decisions", "patient preferences"
                ],
                "scoring_dimensions": {
                    "presence": 0.3, "intensity": 0.35, "quality": 0.35
                },
                "quality_markers": [
                    "decide", "choice", "option", "prefer", "think",
                    "consider", "weigh", "together", "agree", "decision"
                ]
            }
        }
    
    def evaluate(self, text: str, detected_code: str, confidence: float) -> Dict:
        """
        Evaluate a detected EPPC code with multi-dimensional scoring
        
        Args:
            text: The text where the code was detected
            detected_code: The EPPC code category
            confidence: Initial detection confidence
            
        Returns:
            Dictionary with evaluation scores and metrics
        """
        if detected_code not in self.evaluation_criteria:
            return {
                "code": detected_code,
                "presence_score": confidence,
                "intensity_score": confidence,
                "quality_score": confidence,
                "overall_score": confidence,
                "evaluation_available": False
            }
        
        criteria = self.evaluation_criteria[detected_code]
        text_lower = text.lower()
        
        # Calculate presence score (how clearly the code is present)
        presence_score = self._calculate_presence(text_lower, detected_code, confidence)
        
        # Calculate intensity score (how strongly expressed)
        intensity_score = self._calculate_intensity(text_lower, detected_code, confidence)
        
        # Calculate quality score (how well executed)
        quality_score = self._calculate_quality(text_lower, criteria, confidence)
        
        # Weighted overall score
        weights = criteria["scoring_dimensions"]
        overall_score = (
            presence_score * weights["presence"] +
            intensity_score * weights["intensity"] +
            quality_score * weights["quality"]
        )
        
        return {
            "code": detected_code,
            "presence_score": round(presence_score, 3),
            "intensity_score": round(intensity_score, 3),
            "quality_score": round(quality_score, 3),
            "overall_score": round(overall_score, 3),
            "evaluation_available": True,
            "indicators_found": self._find_indicators(text_lower, criteria["indicators"])
        }
    
    def _calculate_presence(self, text: str, code: str, base_confidence: float) -> float:
        """Calculate how clearly the code is present in the text"""
        # Base presence on detection confidence, adjusted by text length and clarity
        text_length_factor = min(len(text.split()) / 50, 1.0)  # Normalize by expected length
        return min(base_confidence * (0.7 + 0.3 * text_length_factor), 1.0)
    
    def _calculate_intensity(self, text: str, code: str, base_confidence: float) -> float:
        """Calculate how intensely the code is expressed"""
        # Look for intensity markers (emphatic language, repetition, etc.)
        intensity_markers = ["very", "really", "extremely", "absolutely", "completely"]
        marker_count = sum(1 for marker in intensity_markers if marker in text)
        intensity_boost = min(marker_count * 0.1, 0.3)
        return min(base_confidence + intensity_boost, 1.0)
    
    def _calculate_quality(self, text: str, criteria: Dict, base_confidence: float) -> float:
        """Calculate quality of the code expression"""
        quality_markers = criteria.get("quality_markers", [])
        marker_matches = sum(1 for marker in quality_markers if marker in text)
        
        if not quality_markers:
            return base_confidence
        
        # Quality increases with more quality markers found
        marker_ratio = min(marker_matches / len(quality_markers), 1.0)
        quality_score = base_confidence * (0.5 + 0.5 * marker_ratio)
        return min(quality_score, 1.0)
    
    def _find_indicators(self, text: str, indicators: List[str]) -> List[str]:
        """Find which indicators are present in the text"""
        found = []
        text_lower = text.lower()
        for indicator in indicators:
            # Check if indicator keywords appear in text
            keywords = indicator.split()
            if any(keyword in text_lower for keyword in keywords):
                found.append(indicator)
        return found
    
    def evaluate_batch(self, results: List[Dict]) -> List[Dict]:
        """Evaluate multiple EPPC detection results"""
        evaluated = []
        for result in results:
            code = result.get('category', 'Unknown')
            text = result.get('matched_phrase', result.get('text', ''))
            confidence = float(result.get('confidence', 0.0))
            
            evaluation = self.evaluate(text, code, confidence)
            # Merge original result with evaluation
            merged = {**result, **evaluation}
            evaluated.append(merged)
        
        return evaluated
