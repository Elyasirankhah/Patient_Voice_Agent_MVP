"""
Chat Agent Module
Provides interactive AI chat functionality for coaching and response generation
"""
from typing import Dict, List, Optional
import os

class ChatAgent:
    """Interactive chat agent for coaching and consultation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize chat agent"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', '')
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                pass
    
    def chat(
        self,
        user_message: str,
        conversation_history: List[Dict],
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate a chat response
        
        Args:
            user_message: User's message
            conversation_history: Previous conversation messages
            context: Optional context dictionary
            
        Returns:
            Agent's response text
        """
        if not self.client:
            return "I'm here to help! However, OpenAI API is not configured. Please set your API key."
        
        try:
            messages = []
            
            # Add system prompt
            system_prompt = self._get_system_prompt(context)
            messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history
            for msg in conversation_history[-10:]:  # Last 10 messages
                role = "assistant" if msg.get('role') == 'agent' else "user"
                messages.append({"role": role, "content": msg.get('message', '')[:1000]})
            
            # Add current message
            messages.append({"role": "user", "content": user_message})
            
            # Generate response using new OpenAI API (v1.0.0+)
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"I apologize, I encountered an error: {str(e)}. Please try again."
    
    def _get_system_prompt(self, context: Optional[Dict] = None) -> str:
        """Get system prompt based on context"""
        base_prompt = """You are a friendly, supportive communication coaching assistant for healthcare providers. 
You help clinicians improve their patient communication by analyzing their communication patterns (EPPC codes) 
and social determinants of health (SDoH) detections. 

Be conversational, warm, and encouraging. Use a friendly tone like you're chatting with a colleague. 
Provide specific, actionable advice based on the analysis history. Keep responses concise but helpful.
Use emojis occasionally to make it more engaging. Be natural and personable.

IMPORTANT RULES:
1. You MUST base your responses on the specific patterns detected in the user's analysis history.
2. Do NOT give generic advice. Always reference the specific EPPC codes and SDoH categories that were detected.
3. You know about INDIVIDUAL conversations. Each conversation is numbered (#1, #2, etc.).
4. If the user asks about a specific conversation, reference that one.
5. If the user asks a question and you're not sure WHICH conversation they mean, ASK them: "Which conversation are you referring to? (#1, #2, etc.)"
6. When discussing patterns, mention which conversation(s) they appeared in."""
        
        if context:
            base_prompt += "\n\n=== USER'S ANALYSIS DATA ==="
            
            if context.get('total_analyses'):
                base_prompt += f"\nTotal conversations analyzed: {context['total_analyses']}"
            
            # Add individual conversation details
            if context.get('conversations'):
                base_prompt += "\n\n--- INDIVIDUAL CONVERSATIONS ---"
                for conv in context['conversations']:
                    base_prompt += f"\n\n**Conversation #{conv['number']}** (analyzed: {conv['timestamp']})"
                    base_prompt += f"\nPreview: {conv['text_preview'][:200]}..."
                    if conv['eppc']:
                        base_prompt += f"\nEPPC patterns: {', '.join(conv['eppc'])}"
                    if conv['sdoh']:
                        base_prompt += f"\nSDoH detected: {', '.join(conv['sdoh'])}"
                base_prompt += "\n\n--- END INDIVIDUAL CONVERSATIONS ---"
            
            if context.get('eppc_codes'):
                # Count frequency of each code
                from collections import Counter
                code_counts = Counter(context['eppc_codes'])
                codes_with_counts = [f"{code} ({count}x)" for code, count in code_counts.most_common()]
                base_prompt += f"\n\nAGGREGATE communication patterns:\n- " + "\n- ".join(codes_with_counts)
                
                # Add specific coaching context for each pattern
                base_prompt += "\n\nCoaching tips for these patterns:"
                if 'Emotional Support' in context['eppc_codes']:
                    base_prompt += "\n- Emotional Support: Patients are expressing emotions/concerns - coach on empathetic responses"
                if 'Information-Seeking' in context['eppc_codes']:
                    base_prompt += "\n- Information-Seeking: Patients are asking questions - coach on clear explanations"
                if 'Partnership' in context['eppc_codes']:
                    base_prompt += "\n- Partnership: Patients want collaboration - coach on shared decision-making"
                if 'Shared Decision-Making' in context['eppc_codes']:
                    base_prompt += "\n- Shared Decision-Making: Patients want input in decisions - coach on presenting options"
                if 'Information-Giving' in context['eppc_codes']:
                    base_prompt += "\n- Information-Giving: Provider is educating - coach on clarity and checking understanding"
            
            if context.get('sdoh_codes'):
                from collections import Counter
                sdoh_counts = Counter(context['sdoh_codes'])
                sdoh_with_counts = [f"{code} ({count}x)" for code, count in sdoh_counts.most_common()]
                base_prompt += f"\n\nAGGREGATE SDoH detected:\n- " + "\n- ".join(sdoh_with_counts)
                base_prompt += "\n\nThese SDoH factors indicate patients may face barriers. Coach on addressing these sensitively."
            
            base_prompt += "\n\n=== END OF DATA ==="
            base_prompt += "\n\nREMEMBER: Reference specific conversation numbers when relevant. Ask 'which conversation?' if unclear."
        
        return base_prompt
