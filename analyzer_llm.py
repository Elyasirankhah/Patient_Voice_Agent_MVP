"""
LLM-based analyzer using OpenAI API for few-shot learning
Most accurate but requires API key
"""
import json
import os
from typing import List, Dict, Tuple

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI not installed. Install with: pip install openai")

class LLMAnalyzer:
    def __init__(self, eppc_codebook_path: str, sdoh_codebook_path: str, api_key: str = None):
        """Initialize LLM analyzer"""
        # Load codebooks
        with open(eppc_codebook_path, 'r', encoding='utf-8') as f:
            self.eppc_codebook = json.load(f)
        with open(sdoh_codebook_path, 'r', encoding='utf-8') as f:
            self.sdoh_codebook = json.load(f)
        
        # Initialize OpenAI client
        self.client = None
        if HAS_OPENAI:
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("⚠️  No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
    
    def _build_few_shot_prompt(self, text: str) -> str:
        """Build few-shot learning prompt with better examples"""
        # Get diverse examples from each category (use more examples since we have 1,496!)
        eppc_examples_text = []
        for cat, data in self.eppc_codebook.items():
            examples = data.get('examples', [])[:5]  # Use 5 examples per category
            if examples:
                examples_str = '\n  - ' + '\n  - '.join(examples[:5])
                eppc_examples_text.append(f"{cat}:{examples_str}")
        
        sdoh_examples_text = []
        for cat, data in self.sdoh_codebook.items():
            examples = data.get('examples', [])[:5]  # Use 5 examples per category
            if examples:
                examples_str = '\n  - ' + '\n  - '.join(examples[:5])
                sdoh_examples_text.append(f"{cat}:{examples_str}")
        
        prompt = f"""You are an expert medical communication analyzer. Analyze the patient message and identify:

1. EPPC (Electronic Patient-Provider Communication) codes
2. SDoH (Social Determinants of Health) categories

EPPC Categories and Examples:
{chr(10).join(eppc_examples_text)}

SDoH Categories and Examples:
{chr(10).join(sdoh_examples_text)}

Patient message to analyze: "{text}"

Instructions:
- Identify ALL relevant EPPC codes and SDoH categories
- Provide confidence scores (0.0-1.0) based on how clearly the message matches
- Only include categories with confidence >= 0.5
- Be sensitive to variations in wording (e.g., "can't afford" = Financial Insecurity)

Respond ONLY with valid JSON (no other text):
{{
  "eppc": [{{"category": "CategoryName", "confidence": 0.85, "reason": "Brief explanation"}}],
  "sdoh": [{{"category": "CategoryName", "confidence": 0.90, "reason": "Brief explanation"}}]
}}
"""
        return prompt
    
    def analyze(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Analyze text using LLM"""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")
        
        prompt = self._build_few_shot_prompt(text)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap
                messages=[
                    {"role": "system", "content": "You are a medical communication analyzer. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                eppc_results = result.get('eppc', [])
                sdoh_results = result.get('sdoh', [])
                return eppc_results, sdoh_results
            else:
                return [], []
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return [], []

