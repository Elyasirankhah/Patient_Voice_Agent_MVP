"""
Response Generation Module for EPPC Codes
Generates empathetic, context-aware responses based on detected EPPC codes
Inspired by Khaled Saab's AMIE research on empathetic clinician-patient interactions
"""
from typing import Dict, List, Optional
import os

class ResponseGenerator:
    """Generates empathetic responses based on EPPC code analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize response generator with OpenAI API"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', '')
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                pass
    
    def generate_response(
        self,
        patient_message: str,
        detected_eppc_codes: List[Dict],
        detected_sdoh_codes: List[Dict],
        context: Optional[str] = None
    ) -> Dict:
        """
        Generate an empathetic response based on detected codes
        
        Args:
            patient_message: The patient's message
            detected_eppc_codes: List of detected EPPC codes
            detected_sdoh_codes: List of detected SDoH codes
            context: Optional additional context
            
        Returns:
            Dictionary with generated response and metadata
        """
        if not self.client:
            return {
                "response": "I understand your concern. Let me help you with that.",
                "error": "OpenAI API key not configured",
                "codes_addressed": []
            }
        
        try:
            # Build prompt for response generation
            prompt = self._build_prompt(patient_message, detected_eppc_codes, detected_sdoh_codes, context)
            
            # Generate response using new OpenAI API (v1.0.0+)
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            return {
                "response": generated_text,
                "codes_addressed": [code.get('category', 'Unknown') for code in detected_eppc_codes],
                "sdoh_addressed": [code.get('category', 'Unknown') for code in detected_sdoh_codes],
                "model": "gpt-4o-mini"
            }
        
        except Exception as e:
            return {
                "response": self._get_fallback_response(detected_eppc_codes, detected_sdoh_codes),
                "error": str(e),
                "codes_addressed": []
            }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for response generation"""
        return """You are an empathetic healthcare provider assistant. Generate responses that:
1. Acknowledge patient concerns and emotions
2. Address detected communication patterns (EPPC codes)
3. Show awareness of social determinants of health (SDoH) when relevant
4. Use warm, professional, and supportive language
5. Keep responses concise and actionable
6. Avoid medical advice - focus on communication and support"""
    
    def _build_prompt(
        self,
        patient_message: str,
        eppc_codes: List[Dict],
        sdoh_codes: List[Dict],
        context: Optional[str]
    ) -> str:
        """Build the prompt for response generation"""
        prompt_parts = [
            f"Patient message: {patient_message}\n\n"
        ]
        
        if eppc_codes:
            eppc_list = [f"- {code.get('category', 'Unknown')} (confidence: {code.get('confidence', 0):.0%})" 
                         for code in eppc_codes[:5]]
            prompt_parts.append(f"Detected communication patterns:\n" + "\n".join(eppc_list) + "\n\n")
        
        if sdoh_codes:
            sdoh_list = [f"- {code.get('category', 'Unknown')} (confidence: {code.get('confidence', 0):.0%})" 
                         for code in sdoh_codes[:5]]
            prompt_parts.append(f"Detected social needs:\n" + "\n".join(sdoh_list) + "\n\n")
        
        if context:
            prompt_parts.append(f"Additional context: {context}\n\n")
        
        prompt_parts.append("Generate an empathetic, appropriate response to the patient's message that addresses the detected patterns and needs.")
        
        return "".join(prompt_parts)
    
    def _get_fallback_response(self, eppc_codes: List[Dict], sdoh_codes: List[Dict]) -> str:
        """Get a fallback response when API fails"""
        if sdoh_codes:
            return "I understand you're facing some challenges. Let's work together to find solutions that work for you."
        elif eppc_codes:
            primary_code = eppc_codes[0].get('category', '') if eppc_codes else ''
            if primary_code == "Emotional Support":
                return "I hear your concerns and want to make sure we address them together."
            elif primary_code == "Information-Seeking":
                return "That's a great question. Let me help clarify that for you."
            else:
                return "Thank you for sharing that with me. Let's work together on this."
        else:
            return "I understand your concern. How can I help you today?"
