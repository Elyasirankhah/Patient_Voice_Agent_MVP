"""
ML-based analyzer using ClinicalBERT embeddings and similarity matching

ClinicalBERT Model:
- Trained on: Clinical notes from MIMIC-III database (intensive care unit records)
- Best for: Patient-provider communication, clinical documentation, patient narratives
- Better than BioBERT for our use case (patient messages vs. research papers)
- Size: ~110M parameters (larger than MiniLM, but more accurate for medical text)
- Embedding dimension: 768 (vs. 384 for MiniLM)
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Try to use sentence transformers for better accuracy
try:
    from sentence_transformers import SentenceTransformer
    USE_EMBEDDINGS = True
except ImportError:
    USE_EMBEDDINGS = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")
    print("   Falling back to improved keyword matching...")

class MLAnalyzer:
    def __init__(self, eppc_codebook_path: str, sdoh_codebook_path: str):
        """Initialize analyzer with codebooks"""
        # Load codebooks
        with open(eppc_codebook_path, 'r', encoding='utf-8') as f:
            self.eppc_codebook = json.load(f)
        with open(sdoh_codebook_path, 'r', encoding='utf-8') as f:
            self.sdoh_codebook = json.load(f)
        
        # Initialize embedding model if available
        self.model = None
        self.eppc_embeddings = {}
        self.sdoh_embeddings = {}
        
        if USE_EMBEDDINGS:
            print("Loading embedding model for semantic analysis...")
            # Note: ClinicalBERT produces too high similarities (0.85-0.99) making discrimination difficult
            # Using all-MiniLM-L6-v2: better discrimination between categories
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model_type = 'general'  # Better for this use case
                print("✅ all-MiniLM-L6-v2 loaded (optimized for discrimination)")
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
                self.model = None
            if self.model:
                self._precompute_embeddings()
        else:
            print("Using improved keyword matching (install sentence-transformers for better accuracy)")
    
    def _precompute_embeddings(self):
        """Precompute embeddings for all examples - optimized for large datasets"""
        if not self.model:
            return
        
        print("Precomputing embeddings for examples...")
        
        # EPPC embeddings - use sample of examples for speed (top 50 per category)
        for category, data in self.eppc_codebook.items():
            examples = data.get('examples', [])
            if examples:
                # Use top 50 examples per category for speed (still very accurate)
                sample_examples = examples[:50] if len(examples) > 50 else examples
                try:
                    embeddings = self.model.encode(sample_examples, convert_to_numpy=True, show_progress_bar=False)
                    self.eppc_embeddings[category] = {
                        'examples': sample_examples,
                        'embeddings': embeddings,
                        'avg_embedding': np.mean(embeddings, axis=0)  # Average embedding
                    }
                except Exception as e:
                    print(f"Error encoding EPPC category {category}: {e}")
        
        # SDoH embeddings - use sample of examples
        for category, data in self.sdoh_codebook.items():
            examples = data.get('examples', [])
            if examples:
                # Use top 50 examples per category for speed
                sample_examples = examples[:50] if len(examples) > 50 else examples
                try:
                    embeddings = self.model.encode(sample_examples, convert_to_numpy=True, show_progress_bar=False)
                    self.sdoh_embeddings[category] = {
                        'examples': sample_examples,
                        'embeddings': embeddings,
                        'avg_embedding': np.mean(embeddings, axis=0)
                    }
                except Exception as e:
                    print(f"Error encoding SDoH category {category}: {e}")
        
        print(f"✅ Precomputed embeddings for {len(self.eppc_embeddings)} EPPC and {len(self.sdoh_embeddings)} SDoH categories")
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def analyze_with_embeddings(self, text: str, threshold: float = 0.60) -> Tuple[List[Dict], List[Dict]]:
        """
        Analyze text using sentence embeddings
        
        Args:
            threshold: Similarity threshold (0.60 default for all-MiniLM-L6-v2)
        """
        if not self.model:
            return self.analyze_improved_keywords(text)
        
        text_embedding = self.model.encode([text], convert_to_numpy=True)[0]
        
        detected_eppc = []
        detected_sdoh = []
        
        # Check EPPC categories
        for category, data in self.eppc_embeddings.items():
            similarities = [self._cosine_similarity(text_embedding, emb) for emb in data['embeddings']]
            max_sim = max(similarities)
            
            if max_sim >= threshold:
                best_match_idx = np.argmax(similarities)
                best_example = data['examples'][best_match_idx]
                
                detected_eppc.append({
                    'category': category,
                    'confidence': float(max_sim),
                    'matched_phrase': best_example,
                    'similarity_score': float(max_sim)
                })
        
        # Check SDoH categories
        for category, data in self.sdoh_embeddings.items():
            similarities = [self._cosine_similarity(text_embedding, emb) for emb in data['embeddings']]
            max_sim = max(similarities)
            
            if max_sim >= threshold:
                best_match_idx = np.argmax(similarities)
                best_example = data['examples'][best_match_idx]
                
                detected_sdoh.append({
                    'category': category,
                    'confidence': float(max_sim),
                    'matched_phrase': best_example,
                    'similarity_score': float(max_sim)
                })
        
        # Sort by confidence
        detected_eppc.sort(key=lambda x: x['confidence'], reverse=True)
        detected_sdoh.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detected_eppc, detected_sdoh
    
    def _select_top_matches(self, scores: List[Dict], min_gap: float = 0.02, max_items: int = 3) -> List[Dict]:
        """
        Select top matches using gap-based approach
        
        Args:
            scores: List of scored items (sorted by confidence)
            min_gap: Minimum gap between top score and others to include
            max_items: Maximum items to return
        
        Returns:
            List of top matches with significant scores
        """
        if not scores:
            return []
        
        # Always include top match if it's high enough
        if scores[0]['confidence'] < 0.85:  # Absolute minimum for ClinicalBERT
            return []
        
        selected = [scores[0]]
        
        # Add more if they're close to the top score
        for i in range(1, min(max_items, len(scores))):
            gap = scores[0]['confidence'] - scores[i]['confidence']
            if gap <= min_gap:  # Close enough to top score
                selected.append(scores[i])
            else:
                break  # Gap is too large, stop
        
        return selected
    
    def analyze_improved_keywords(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Improved keyword matching with fuzzy matching"""
        text_lower = text.lower()
        detected_eppc = []
        detected_sdoh = []
        
        # EPPC - check for partial matches and word overlap
        for category, data in self.eppc_codebook.items():
            examples = data.get('examples', [])
            matches = []
            
            for example in examples:
                example_lower = example.lower()
                # Exact substring match
                if example_lower in text_lower:
                    matches.append((example, 0.9))
                # Word overlap
                example_words = set(example_lower.split())
                text_words = set(text_lower.split())
                common_words = example_words.intersection(text_words)
                if len(common_words) >= 2:  # At least 2 words match
                    overlap_score = len(common_words) / max(len(example_words), 1)
                    if overlap_score > 0.3:
                        matches.append((example, overlap_score))
            
            if matches:
                best_match = max(matches, key=lambda x: x[1])
                detected_eppc.append({
                    'category': category,
                    'confidence': best_match[1],
                    'matched_phrase': best_match[0]
                })
        
        # SDoH - same approach
        for category, data in self.sdoh_codebook.items():
            examples = data.get('examples', [])
            matches = []
            
            for example in examples:
                example_lower = example.lower()
                if example_lower in text_lower:
                    matches.append((example, 0.9))
                else:
                    example_words = set(example_lower.split())
                    text_words = set(text_lower.split())
                    common_words = example_words.intersection(text_words)
                    if len(common_words) >= 2:
                        overlap_score = len(common_words) / max(len(example_words), 1)
                        if overlap_score > 0.3:
                            matches.append((example, overlap_score))
            
            if matches:
                best_match = max(matches, key=lambda x: x[1])
                detected_sdoh.append({
                    'category': category,
                    'confidence': best_match[1],
                    'matched_phrase': best_match[0]
                })
        
        detected_eppc.sort(key=lambda x: x['confidence'], reverse=True)
        detected_sdoh.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detected_eppc[:5], detected_sdoh[:5]
    
    def analyze(self, text: str, threshold: float = 0.60) -> Tuple[List[Dict], List[Dict]]:
        """
        Main analysis method
        
        Args:
            text: Input text to analyze
            threshold: Similarity threshold (default 0.60 for all-MiniLM-L6-v2)
                      - 0.55-0.65: Balanced (recommended)
                      - 0.65-0.70: Conservative (high precision)
                      - 0.50-0.55: Sensitive (higher recall)
        """
        if USE_EMBEDDINGS and self.model:
            return self.analyze_with_embeddings(text, threshold)
        else:
            return self.analyze_improved_keywords(text)

