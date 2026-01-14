"""
Thread Analyzer Module
Analyzes full conversation threads and evaluates provider responsiveness
"""
from typing import Dict, List, Optional
from datetime import datetime
import re
from collections import defaultdict

class ThreadAnalyzer:
    """Analyzes conversation threads for patterns and provider responsiveness"""
    
    def __init__(self, text_analyzer, eppc_evaluator=None):
        """
        Initialize thread analyzer
        
        Args:
            text_analyzer: Function or analyzer instance to analyze individual messages
            eppc_evaluator: Optional EPPC evaluator for enhanced evaluation
        """
        self.text_analyzer = text_analyzer
        self.eppc_evaluator = eppc_evaluator
    
    def analyze_thread(self, thread_text: str) -> Dict:
        """
        Analyze a full conversation thread
        
        Args:
            thread_text: Full conversation thread text
            
        Returns:
            Dictionary with thread analysis results
        """
        # Parse thread into messages
        messages = self._parse_thread(thread_text)
        
        if not messages:
            return {
                "error": "Could not parse thread into messages",
                "messages": []
            }
        
        # Separate patient and provider messages
        patient_messages = [m for m in messages if m['role'] == 'patient']
        provider_messages = [m for m in messages if m['role'] == 'provider']
        
        # Analyze patient communication
        patient_communication = self._analyze_patient_communication(patient_messages)
        
        # Analyze provider communication
        provider_communication = self._analyze_provider_communication(provider_messages)
        
        # Evaluate provider responsiveness
        responsiveness = self._evaluate_responsiveness(patient_messages, provider_messages)
        
        # Calculate conversation metrics
        metrics = self._calculate_metrics(messages, patient_messages, provider_messages)
        
        return {
            "total_messages": len(messages),
            "patient_messages": len(patient_messages),
            "provider_messages": len(provider_messages),
            "patient_communication": patient_communication,
            "provider_communication": provider_communication,
            "responsiveness": responsiveness,
            "metrics": metrics,
            "messages": messages
        }
    
    def _parse_thread(self, thread_text: str) -> List[Dict]:
        """Parse thread text into individual messages"""
        messages = []
        lines = thread_text.strip().split('\n')
        
        current_role = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for role indicators
            if line.lower().startswith('patient:'):
                # Save previous message if exists
                if current_role and current_content:
                    messages.append({
                        'role': current_role,
                        'content': ' '.join(current_content),
                        'timestamp': None
                    })
                
                current_role = 'patient'
                current_content = [line.split(':', 1)[1].strip()] if ':' in line else []
            
            elif line.lower().startswith('provider:') or line.lower().startswith('doctor:') or line.lower().startswith('clinician:'):
                # Save previous message if exists
                if current_role and current_content:
                    messages.append({
                        'role': current_role,
                        'content': ' '.join(current_content),
                        'timestamp': None
                    })
                
                current_role = 'provider'
                current_content = [line.split(':', 1)[1].strip()] if ':' in line else []
            
            else:
                # Continuation of current message
                if current_role:
                    current_content.append(line)
        
        # Save last message
        if current_role and current_content:
            messages.append({
                'role': current_role,
                'content': ' '.join(current_content),
                'timestamp': None
            })
        
        return messages
    
    def _analyze_patient_communication(self, patient_messages: List[Dict]) -> Dict:
        """Analyze patient communication patterns"""
        all_eppc_codes = []
        all_sdoh_codes = []
        
        for msg in patient_messages:
            try:
                eppc_results, sdoh_results = self.text_analyzer(msg['content'])
                all_eppc_codes.extend(eppc_results)
                all_sdoh_codes.extend(sdoh_results)
            except Exception as e:
                print(f"Error analyzing patient message: {e}")
                continue
        
        # Evaluate codes if evaluator available
        if self.eppc_evaluator and all_eppc_codes:
            all_eppc_codes = self.eppc_evaluator.evaluate_batch(all_eppc_codes)
        
        return {
            "eppc_codes": all_eppc_codes,
            "sdoh_codes": all_sdoh_codes,
            "total_patterns": len(all_eppc_codes),
            "total_sdoh": len(all_sdoh_codes)
        }
    
    def _analyze_provider_communication(self, provider_messages: List[Dict]) -> Dict:
        """Analyze provider communication patterns"""
        all_eppc_codes = []
        
        for msg in provider_messages:
            try:
                eppc_results, _ = self.text_analyzer(msg['content'])
                all_eppc_codes.extend(eppc_results)
            except Exception as e:
                print(f"Error analyzing provider message: {e}")
                continue
        
        return {
            "eppc_codes": all_eppc_codes,
            "total_patterns": len(all_eppc_codes)
        }
    
    def _evaluate_responsiveness(self, patient_messages: List[Dict], provider_messages: List[Dict]) -> Dict:
        """Evaluate provider responsiveness to patient messages"""
        if not patient_messages:
            return {
                "score": 0.0,
                "response_rate": 0.0,
                "average_response_time": None,
                "issues": ["No patient messages found"]
            }
        
        # Simple responsiveness: ratio of provider responses to patient messages
        response_ratio = len(provider_messages) / len(patient_messages) if patient_messages else 0
        
        # Check if provider addresses patient concerns
        issues = []
        if response_ratio < 0.5:
            issues.append("Low response rate - provider may not be responding to all patient messages")
        elif response_ratio > 2.0:
            issues.append("High response rate - provider may be over-responding")
        
        return {
            "score": min(response_ratio / 1.0, 1.0),  # Normalize to 0-1
            "response_rate": response_ratio,
            "average_response_time": None,  # Would need timestamps
            "issues": issues
        }
    
    def _calculate_metrics(self, all_messages: List[Dict], patient_messages: List[Dict], provider_messages: List[Dict]) -> Dict:
        """Calculate conversation metrics"""
        total_words = sum(len(msg['content'].split()) for msg in all_messages)
        patient_words = sum(len(msg['content'].split()) for msg in patient_messages)
        provider_words = sum(len(msg['content'].split()) for msg in provider_messages)
        
        return {
            "total_words": total_words,
            "patient_words": patient_words,
            "provider_words": provider_words,
            "patient_word_ratio": patient_words / total_words if total_words > 0 else 0,
            "provider_word_ratio": provider_words / total_words if total_words > 0 else 0
        }
