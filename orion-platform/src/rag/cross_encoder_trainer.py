"""
ORION Cross-Encoder Training for RAG Reranking
=============================================

Trains cross-encoder models for improved reranking in the RAG pipeline.
"""

from sentence_transformers import CrossEncoder, SentencesDataset, losses, evaluation
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import time
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CrossEncoderTrainer:
    """Training pipeline for cross-encoder reranking models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.model = None
        self.training_history = []
        
    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'model_name': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
            'num_labels': 1,
            'max_length': 512,
            'batch_size': 16,
            'epochs': 3,
            'evaluation_steps': 1000,
            'warmup_steps': 200,
            'learning_rate': 2e-5,
            'use_amp': True,  # Mixed precision training
            'save_dir': 'models/cross_encoder_reranker',
            'validation_split': 0.2,
            'negative_sampling_ratio': 4,  # Negative to positive ratio
            'hard_negative_mining': True,
            'temperature': 1.0  # For distillation from larger models
        }
    
    def prepare_training_data(self, query_doc_pairs: List[Tuple[str, str, float]]) -> List[InputExample]:
        """
        Prepare training data from query-document pairs with relevance scores
        
        Args:
            query_doc_pairs: List of (query, document, relevance_score) tuples
        
        Returns:
            List of InputExample objects for training
        """
        train_examples = []
        
        for query, document, score in query_doc_pairs:
            # Create InputExample with query and document as texts, score as label
            example = InputExample(
                texts=[query, document],
                label=float(score)
            )
            train_examples.append(example)
        
        # Add hard negatives if enabled
        if self.config['hard_negative_mining']:
            hard_negatives = self._mine_hard_negatives(query_doc_pairs)
            train_examples.extend(hard_negatives)
        
        logger.info(f"Prepared {len(train_examples)} training examples")
        return train_examples
    
    def _mine_hard_negatives(self, query_doc_pairs: List[Tuple[str, str, float]]) -> List[InputExample]:
        """Mine hard negative examples for training"""
        hard_negatives = []
        
        # Group by query
        query_groups = {}
        for query, doc, score in query_doc_pairs:
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append((doc, score))
        
        # For each query, create hard negatives
        for query, doc_scores in query_groups.items():
            # Sort by score
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top documents as positives
            positives = [doc for doc, score in doc_scores if score > 0.5]
            
            # Take bottom documents as hard negatives
            negatives = [doc for doc, score in doc_scores if score <= 0.5]
            
            # Create negative examples
            for neg_doc in negatives[:self.config['negative_sampling_ratio']]:
                hard_negatives.append(
                    InputExample(texts=[query, neg_doc], label=0.0)
                )
        
        return hard_negatives
    
    def prepare_materials_science_data(self, materials_queries: List[Dict[str, Any]]) -> List[InputExample]:
        """
        Prepare training data specific to materials science domain
        
        Args:
            materials_queries: List of dicts with 'query', 'positive_docs', 'negative_docs'
        
        Returns:
            List of InputExample objects
        """
        train_examples = []
        
        for item in materials_queries:
            query = item['query']
            
            # Add positive examples
            for pos_doc in item.get('positive_docs', []):
                train_examples.append(
                    InputExample(
                        texts=[query, pos_doc['text']],
                        label=pos_doc.get('relevance', 1.0)
                    )
                )
            
            # Add negative examples
            for neg_doc in item.get('negative_docs', []):
                train_examples.append(
                    InputExample(
                        texts=[query, neg_doc['text']],
                        label=neg_doc.get('relevance', 0.0)
                    )
                )
        
        # Add domain-specific augmentations
        augmented_examples = self._augment_materials_queries(train_examples)
        train_examples.extend(augmented_examples)
        
        return train_examples
    
    def _augment_materials_queries(self, examples: List[InputExample]) -> List[InputExample]:
        """Augment training data with materials science specific variations"""
        augmented = []
        
        materials_synonyms = {
            'bandgap': ['band gap', 'energy gap', 'Eg'],
            'formation energy': ['heat of formation', 'formation enthalpy', 'ΔHf'],
            'bulk modulus': ['compression modulus', 'K', 'B'],
            'density': ['mass density', 'ρ', 'specific gravity'],
            'conductivity': ['electrical conductivity', 'σ', 'conductance'],
            'TiO2': ['titanium dioxide', 'titania', 'titanium(IV) oxide'],
            'synthesis': ['preparation', 'fabrication', 'processing'],
            'DFT': ['density functional theory', 'first principles', 'ab initio']
        }
        
        for example in examples[:100]:  # Limit augmentation to prevent explosion
            query, doc = example.texts
            
            # Replace with synonyms
            for term, synonyms in materials_synonyms.items():
                if term.lower() in query.lower():
                    for synonym in synonyms:
                        new_query = query.replace(term, synonym)
                        augmented.append(
                            InputExample(
                                texts=[new_query, doc],
                                label=example.label * 0.9  # Slightly lower confidence
                            )
                        )
                        break  # One augmentation per term
        
        return augmented
    
    def train(self, train_examples: List[InputExample], 
              val_examples: Optional[List[InputExample]] = None) -> CrossEncoder:
        """
        Train the cross-encoder model
        
        Args:
            train_examples: Training data
            val_examples: Validation data (optional)
        
        Returns:
            Trained CrossEncoder model
        """
        # Initialize model
        self.model = CrossEncoder(
            self.config['model_name'],
            num_labels=self.config['num_labels'],
            max_length=self.config['max_length']
        )
        
        # Create datasets
        train_dataset = SentencesDataset(train_examples, self.model)
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.config['batch_size']
        )
        
        # Define loss function
        if self.config['num_labels'] == 1:
            train_loss = losses.RankingLoss(self.model)
        else:
            train_loss = losses.SoftmaxLoss(self.model)
        
        # Setup evaluation
        evaluator = None
        if val_examples:
            val_dataset = SentencesDataset(val_examples, self.model)
            val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=self.config['batch_size']
            )
            evaluator = evaluation.RerankingEvaluator(
                val_dataloader,
                name='cross-encoder-eval'
            )
        
        # Training
        logger.info(f"Starting cross-encoder training for {self.config['epochs']} epochs")
        
        self.model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=self.config['epochs'],
            evaluation_steps=self.config['evaluation_steps'],
            output_path=self.config['save_dir'],
            warmup_steps=self.config['warmup_steps'],
            use_amp=self.config['use_amp'],
            callback=self._training_callback,
            show_progress_bar=True
        )
        
        logger.info("Cross-encoder training completed")
        return self.model
    
    def _training_callback(self, score, epoch, steps):
        """Callback to track training progress"""
        self.training_history.append({
            'score': score,
            'epoch': epoch,
            'steps': steps,
            'timestamp': time.time()
        })
        
        if steps % 1000 == 0:
            logger.info(f"Step {steps}: Score = {score:.4f}")
    
    def train_from_materials_dataset(self, dataset_path: str) -> CrossEncoder:
        """
        Train from a materials science specific dataset
        
        Args:
            dataset_path: Path to dataset file (JSON or CSV)
        
        Returns:
            Trained model
        """
        # Load dataset
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        elif dataset_path.endswith('.csv'):
            data = pd.read_csv(dataset_path).to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        # Prepare examples
        all_examples = []
        
        for item in data:
            # Expected format: query, positive_chunk, negative_chunks, relevance_score
            query = item['query']
            
            # Positive example
            all_examples.append(
                InputExample(
                    texts=[query, item['positive_chunk']],
                    label=item.get('relevance_score', 1.0)
                )
            )
            
            # Negative examples
            if 'negative_chunks' in item:
                for neg_chunk in item['negative_chunks']:
                    all_examples.append(
                        InputExample(
                            texts=[query, neg_chunk],
                            label=0.0
                        )
                    )
        
        # Split into train/val
        train_examples, val_examples = train_test_split(
            all_examples,
            test_size=self.config['validation_split'],
            random_state=42
        )
        
        # Train model
        return self.train(train_examples, val_examples)
    
    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Rerank candidates for a given query
        
        Args:
            query: Search query
            candidates: List of candidate text chunks
        
        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Create input pairs
        inputs = [[query, chunk] for chunk in candidates]
        
        # Get scores
        scores = self.model.predict(inputs)
        
        # Sort by score
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def evaluate_on_materials_benchmark(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate model on materials science specific benchmark
        
        Args:
            test_data: List of test queries with ground truth rankings
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        metrics = {
            'mrr': [],  # Mean Reciprocal Rank
            'map': [],  # Mean Average Precision
            'ndcg@5': [],
            'ndcg@10': [],
            'precision@1': [],
            'precision@5': []
        }
        
        for item in test_data:
            query = item['query']
            relevant_docs = set(item['relevant_docs'])
            candidates = item['candidates']
            
            # Rerank
            ranked = self.rerank(query, candidates)
            
            # Calculate metrics
            # MRR
            for i, (doc, _) in enumerate(ranked):
                if doc in relevant_docs:
                    metrics['mrr'].append(1.0 / (i + 1))
                    break
            else:
                metrics['mrr'].append(0.0)
            
            # Precision@k
            p_at_1 = 1.0 if ranked[0][0] in relevant_docs else 0.0
            metrics['precision@1'].append(p_at_1)
            
            p_at_5 = sum(1 for doc, _ in ranked[:5] if doc in relevant_docs) / 5.0
            metrics['precision@5'].append(p_at_5)
            
            # NDCG
            relevance_scores = [1.0 if doc in relevant_docs else 0.0 for doc, _ in ranked]
            metrics['ndcg@5'].append(self._calculate_ndcg(relevance_scores[:5]))
            metrics['ndcg@10'].append(self._calculate_ndcg(relevance_scores[:10]))
        
        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        
        logger.info(f"Evaluation results: {avg_metrics}")
        return avg_metrics
    
    def _calculate_ndcg(self, relevance_scores: List[float]) -> float:
        """Calculate NDCG for a ranked list"""
        if not relevance_scores or sum(relevance_scores) == 0:
            return 0.0
        
        # DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        
        # IDCG
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def save_model(self, save_path: Optional[str] = None):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        if save_path is None:
            save_path = self.config['save_dir']
        
        self.model.save(save_path)
        
        # Save training history and config
        metadata = {
            'config': self.config,
            'training_history': self.training_history
        }
        
        with open(Path(save_path) / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a trained model"""
        self.model = CrossEncoder(load_path)
        
        # Load metadata if available
        metadata_path = Path(load_path) / 'training_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.config = metadata.get('config', self.config)
                self.training_history = metadata.get('training_history', [])
        
        logger.info(f"Model loaded from {load_path}")
    
    def create_synthetic_training_data(self, num_examples: int = 10000) -> List[InputExample]:
        """Create synthetic training data for materials science domain"""
        examples = []
        
        # Materials science query templates
        query_templates = [
            "What is the {property} of {material}?",
            "Find materials with {property} greater than {value} {unit}",
            "How to synthesize {material} using {method}?",
            "Compare {property} between {material1} and {material2}",
            "What are the applications of {material} in {field}?",
            "{material} crystal structure and properties",
            "DFT calculation of {property} for {material}",
            "Experimental measurement of {property} in {material}"
        ]
        
        # Sample values
        properties = ['bandgap', 'formation energy', 'bulk modulus', 'density', 
                     'melting point', 'thermal conductivity', 'electrical conductivity']
        materials = ['TiO2', 'Si', 'GaN', 'graphene', 'MoS2', 'perovskite', 
                    'ZnO', 'Cu2O', 'BaTiO3', 'SrTiO3']
        methods = ['sol-gel', 'CVD', 'ALD', 'hydrothermal', 'solid state']
        
        # Generate examples
        for _ in range(num_examples):
            # Random query
            template = np.random.choice(query_templates)
            query = template.format(
                property=np.random.choice(properties),
                material=np.random.choice(materials),
                material1=np.random.choice(materials),
                material2=np.random.choice(materials),
                method=np.random.choice(methods),
                value=np.random.randint(1, 100),
                unit=np.random.choice(['eV', 'GPa', 'K', 'g/cm³']),
                field=np.random.choice(['electronics', 'photovoltaics', 'catalysis'])
            )
            
            # Generate positive document
            positive_doc = f"The {np.random.choice(properties)} of {np.random.choice(materials)} " \
                          f"has been measured to be {np.random.uniform(0.1, 10):.2f} " \
                          f"{np.random.choice(['eV', 'GPa', 'K'])}. " \
                          f"This was determined using {np.random.choice(['DFT', 'experimental', 'ML'])} methods."
            
            # Generate negative document (different topic)
            negative_doc = f"The synthesis of {np.random.choice(materials)} can be achieved through " \
                          f"{np.random.choice(methods)} method at {np.random.randint(300, 1500)}K."
            
            # Add examples
            examples.append(InputExample(texts=[query, positive_doc], label=1.0))
            examples.append(InputExample(texts=[query, negative_doc], label=0.0))
        
        return examples