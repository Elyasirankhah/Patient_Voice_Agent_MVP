"""
Analyzer using the fine-tuned transformer model

This analyzer uses the model trained by train_model.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TrainedAnalyzer:
    def __init__(self, model_path: str = './trained_models/final_model'):
        """
        Initialize analyzer with trained model
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading trained model from {model_path}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mappings
        with open(self.model_path / 'label_mappings.json', 'r') as f:
            mappings = json.load(f)
            self.label2id = mappings['label2id']
            self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
            self.num_labels = mappings['num_labels']
        
        print(f"✅ Model loaded ({self.num_labels} categories)")
    
    def analyze(self, text: str, top_k: int = 3, confidence_threshold: float = 0.5) -> Tuple[List[Dict], List[Dict]]:
        """
        Analyze text and return EPPC and SDoH predictions
        
        Args:
            text: Input text to analyze
            top_k: Return top-k predictions per category type
            confidence_threshold: Minimum confidence to include prediction
        
        Returns:
            Tuple of (eppc_results, sdoh_results)
        """
        # Tokenize input
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Get top predictions
        probs = probs.cpu().numpy()[0]
        top_indices = np.argsort(probs)[::-1]
        
        eppc_results = []
        sdoh_results = []
        
        for idx in top_indices:
            prob = float(probs[idx])
            if prob < confidence_threshold:
                break
            
            label = self.id2label[idx]
            
            result = {
                'category': label.replace('EPPC_', '').replace('SDoH_', ''),
                'confidence': prob,
                'full_label': label
            }
            
            if label.startswith('EPPC_'):
                if len(eppc_results) < top_k:
                    eppc_results.append(result)
            elif label.startswith('SDoH_'):
                if len(sdoh_results) < top_k:
                    sdoh_results.append(result)
        
        return eppc_results, sdoh_results
    
    def analyze_batch(self, texts: List[str], confidence_threshold: float = 0.5) -> List[Tuple[List[Dict], List[Dict]]]:
        """
        Analyze multiple texts at once (more efficient)
        
        Args:
            texts: List of texts to analyze
            confidence_threshold: Minimum confidence threshold
        
        Returns:
            List of (eppc_results, sdoh_results) tuples
        """
        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Process each result
        results = []
        probs_np = probs.cpu().numpy()
        
        for i in range(len(texts)):
            prob = probs_np[i]
            top_indices = np.argsort(prob)[::-1]
            
            eppc_results = []
            sdoh_results = []
            
            for idx in top_indices:
                if prob[idx] < confidence_threshold:
                    break
                
                label = self.id2label[idx]
                
                result = {
                    'category': label.replace('EPPC_', '').replace('SDoH_', ''),
                    'confidence': float(prob[idx]),
                    'full_label': label
                }
                
                if label.startswith('EPPC_'):
                    if len(eppc_results) < 3:
                        eppc_results.append(result)
                elif label.startswith('SDoH_'):
                    if len(sdoh_results) < 3:
                        sdoh_results.append(result)
            
            results.append((eppc_results, sdoh_results))
        
        return results


# Example usage
if __name__ == "__main__":
    print("Testing trained analyzer...")
    
    analyzer = TrainedAnalyzer()
    
    test_texts = [
        "I can't afford my medication anymore",
        "I don't have a place to live",
        "Can you help me understand my treatment options?"
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        eppc, sdoh = analyzer.analyze(text)
        print(f"EPPC: {eppc}")
        print(f"SDoH: {sdoh}")

