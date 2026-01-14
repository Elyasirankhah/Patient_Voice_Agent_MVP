"""
Communication Coach Module
Provides personalized coaching tips based on communication analysis
"""
from typing import Dict, List, Optional
from collections import Counter

class CommunicationCoach:
    """Provides personalized communication coaching"""
    
    def __init__(self):
        """Initialize communication coach"""
        self.coaching_tips = self._initialize_coaching_tips()
    
    def _initialize_coaching_tips(self) -> Dict:
        """Initialize coaching tips for each EPPC code"""
        return {
            "Partnership": [
                "Use collaborative language like 'we' and 'together'",
                "Ask for patient preferences and opinions",
                "Frame decisions as shared responsibility",
                "Acknowledge patient expertise in their own experience"
            ],
            "Emotional Support": [
                "Acknowledge patient emotions explicitly",
                "Use empathetic language ('I understand', 'I hear you')",
                "Validate patient concerns",
                "Offer reassurance when appropriate"
            ],
            "Information-Giving": [
                "Use clear, jargon-free language",
                "Break down complex information into digestible parts",
                "Check for understanding",
                "Provide context and explanations"
            ],
            "Information-Seeking": [
                "Ask open-ended questions",
                "Encourage patient to share concerns",
                "Listen actively and follow up",
                "Create a safe space for questions"
            ],
            "Shared Decision-Making": [
                "Present multiple options when available",
                "Discuss pros and cons together",
                "Respect patient preferences",
                "Collaborate on treatment decisions"
            ]
        }
    
    def generate_coaching_summary(self, analysis_history: List[Dict]) -> Dict:
        """
        Generate coaching summary based on analysis history
        
        Args:
            analysis_history: List of analysis results
            
        Returns:
            Dictionary with coaching summary and recommendations
        """
        if not analysis_history:
            return {
                "summary": "No analysis history available yet.",
                "recommendations": [],
                "patterns": {}
            }
        
        # Aggregate patterns
        eppc_counter = Counter()
        sdoh_counter = Counter()
        
        for result in analysis_history:
            if result.get('type') == 'thread':
                thread_analysis = result.get('thread_analysis', {})
                patient_comm = thread_analysis.get('patient_communication', {})
                for code in patient_comm.get('eppc_codes', []):
                    eppc_counter[code.get('category', 'Unknown')] += 1
                for code in patient_comm.get('sdoh_codes', []):
                    sdoh_counter[code.get('category', 'Unknown')] += 1
            else:
                for code in result.get('eppc', []):
                    eppc_counter[code.get('category', 'Unknown')] += 1
                for code in result.get('sdoh', []):
                    sdoh_counter[code.get('category', 'Unknown')] += 1
        
        # Generate summary
        summary_parts = []
        summary_parts.append(f"Based on {len(analysis_history)} analysis{'es' if len(analysis_history) > 1 else ''}, ")
        
        if eppc_counter:
            top_patterns = eppc_counter.most_common(3)
            pattern_list = [f"{pattern} ({count}x)" for pattern, count in top_patterns]
            summary_parts.append(f"the most common communication patterns are: {', '.join(pattern_list)}. ")
        else:
            summary_parts.append("no consistent communication patterns have been detected yet. ")
        
        if sdoh_counter:
            top_sdoh = sdoh_counter.most_common(3)
            sdoh_list = [f"{cat} ({count}x)" for cat, count in top_sdoh]
            summary_parts.append(f"Social determinants of health concerns include: {', '.join(sdoh_list)}. ")
        
        # Generate recommendations
        recommendations = []
        for pattern, count in eppc_counter.most_common(3):
            if pattern in self.coaching_tips:
                tips = self.coaching_tips[pattern]
                recommendations.append({
                    "pattern": pattern,
                    "frequency": count,
                    "tips": tips[:2]  # Top 2 tips
                })
        
        return {
            "summary": "".join(summary_parts),
            "recommendations": recommendations,
            "patterns": {
                "eppc": dict(eppc_counter),
                "sdoh": dict(sdoh_counter)
            }
        }
