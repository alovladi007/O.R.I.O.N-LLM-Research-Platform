"""
ORION RAG System Implementation
==============================

Advanced Retrieval-Augmented Generation with hybrid search and reranking.
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
import faiss
import redis
import time
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from elasticsearch import AsyncElasticsearch
from .cross_encoder_trainer import CrossEncoderTrainer

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Advanced RAG System implementation with:
    - Hybrid sparse-dense retrieval
    - FAISS vector indexing
    - Elasticsearch integration
    - Cross-encoder reranking
    - Context-aware generation
    - Redis caching
    """
    
    def __init__(self, config, knowledge_graph=None):
        self.config = config
        self.knowledge_graph = knowledge_graph
        self._initialized = False
        
        # RAG components
        self.embedding_model = None
        self.cross_encoder = None
        self.faiss_index = None
        self.es_client = None
        self.redis_client = None
        
        # Configuration
        self.rag_config = config.get('rag', {})
        self.alpha = self.rag_config.get('retrieval', {}).get('alpha', 0.6)
        self.top_k = self.rag_config.get('retrieval', {}).get('top_k', 8)
        self.rerank_n = self.rag_config.get('retrieval', {}).get('rerank_n', 12)
        self.score_threshold = self.rag_config.get('retrieval', {}).get('score_threshold', 0.2)
        self.chunk_size = self.rag_config.get('retrieval', {}).get('chunk_size', 512)
        self.chunk_overlap = self.rag_config.get('retrieval', {}).get('chunk_overlap', 20)
        
        logger.info("Advanced RAG System created")
    
    async def initialize(self):
        """Initialize RAG system components"""
        # Initialize embedding model
        embedding_model_name = self.rag_config.get('embedding', {}).get('model', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize FAISS index
        self._initialize_faiss()
        
        # Initialize Elasticsearch
        es_config = self.config.get('database', {}).get('elasticsearch', {})
        if es_config.get('enabled', True):
            self.es_client = AsyncElasticsearch(
                [f"{es_config.get('host', 'localhost')}:{es_config.get('port', 9200)}"],
                basic_auth=(es_config.get('username'), es_config.get('password'))
            )
        
        # Initialize Redis cache
        redis_config = self.config.get('database', {}).get('redis', {})
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            password=redis_config.get('password'),
            decode_responses=True
        )
        
        # Load cross-encoder if available
        cross_encoder_path = self.rag_config.get('cross_encoder_path')
        if cross_encoder_path and Path(cross_encoder_path).exists():
            trainer = CrossEncoderTrainer()
            trainer.load_model(cross_encoder_path)
            self.cross_encoder = trainer
        
        self._initialized = True
        logger.info("RAG System initialized")
    
    def _initialize_faiss(self):
        """Initialize FAISS index with configuration"""
        faiss_config = self.rag_config.get('faiss', {})
        index_type = faiss_config.get('index_type', 'HNSWFlat')
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        if index_type == 'HNSWFlat':
            M = faiss_config.get('M', 32)
            ef_construction = faiss_config.get('ef_construction', 200)
            ef_search = faiss_config.get('ef_search', 64)
            
            # Create HNSW index
            self.faiss_index = faiss.IndexHNSWFlat(dimension, M)
            self.faiss_index.hnsw.efConstruction = ef_construction
            self.faiss_index.hnsw.efSearch = ef_search
            
        elif index_type == 'IVF_PQ':
            nlist = faiss_config.get('nlist', 100)
            m = faiss_config.get('m', 8)
            
            # Create IVF-PQ index
            quantizer = faiss.IndexFlatL2(dimension)
            self.faiss_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)
            
        else:
            # Default to flat index
            self.faiss_index = faiss.IndexFlatL2(dimension)
        
        # Storage for document metadata
        self.document_store = {}
        self.doc_id_counter = 0
    
    async def shutdown(self):
        """Shutdown RAG system"""
        if self.es_client:
            await self.es_client.close()
        
        if self.redis_client:
            self.redis_client.close()
        
        self._initialized = False
        logger.info("RAG System shutdown")
    
    async def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze query intent and entities"""
        # Extract entities and intent
        entities = self._extract_entities(query)
        intent = self._classify_intent(query)
        
        # Check cache
        cache_key = f"rag:query_analysis:{hash(query)}"
        cached = self.redis_client.get(cache_key)
        if cached:
            import json
            return json.loads(cached)
        
        result = {
            "query": query,
            "entities": entities,
            "intent": intent,
            "context": context or {},
            "timestamp": time.time()
        }
        
        # Cache result
        self.redis_client.setex(
            cache_key, 
            self.rag_config.get('cache', {}).get('ttl', 21600),  # 6 hours
            json.dumps(result)
        )
        
        return result
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract material science entities from query"""
        entities = {
            "materials": [],
            "properties": [],
            "methods": [],
            "values": []
        }
        
        # Simple pattern matching (would use NER model in production)
        material_patterns = ['TiO2', 'Si', 'GaN', 'graphene', 'perovskite']
        property_patterns = ['bandgap', 'formation energy', 'bulk modulus', 'density']
        method_patterns = ['DFT', 'synthesis', 'CVD', 'ALD', 'sol-gel']
        
        query_lower = query.lower()
        
        for material in material_patterns:
            if material.lower() in query_lower:
                entities["materials"].append(material)
        
        for prop in property_patterns:
            if prop.lower() in query_lower:
                entities["properties"].append(prop)
        
        for method in method_patterns:
            if method.lower() in query_lower:
                entities["methods"].append(method)
        
        return entities
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'explain']):
            return 'definition'
        elif any(word in query_lower for word in ['find', 'search', 'list']):
            return 'search'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            return 'comparison'
        elif any(word in query_lower for word in ['synthesize', 'make', 'prepare']):
            return 'synthesis'
        elif any(word in query_lower for word in ['predict', 'calculate', 'estimate']):
            return 'prediction'
        else:
            return 'general'
    
    async def search(self, query: str, num_results: int = None, 
                    filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform hybrid search with reranking"""
        if num_results is None:
            num_results = self.top_k
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        
        # Dense retrieval with FAISS
        dense_results = self._faiss_search(query_embedding, num_results * 2)
        
        # Sparse retrieval with Elasticsearch
        sparse_results = []
        if self.es_client:
            sparse_results = await self._elasticsearch_search(query, num_results * 2, filters)
        
        # Hybrid fusion
        fused_results = self._hybrid_fusion(dense_results, sparse_results)
        
        # Reranking with cross-encoder
        if self.cross_encoder and len(fused_results) > 0:
            reranked_results = self._rerank_results(query, fused_results[:self.rerank_n])
        else:
            reranked_results = fused_results
        
        # Filter by score threshold
        filtered_results = [
            r for r in reranked_results 
            if r.get('score', 0) >= self.score_threshold
        ][:num_results]
        
        return {
            "query": query,
            "results": filtered_results,
            "num_results": len(filtered_results),
            "search_type": "hybrid",
            "filters_applied": filters
        }
    
    def _faiss_search(self, query_embedding: torch.Tensor, k: int) -> List[Dict[str, Any]]:
        """Search FAISS index"""
        if self.faiss_index.ntotal == 0:
            return []
        
        # Convert to numpy
        query_vec = query_embedding.cpu().numpy().reshape(1, -1)
        
        # Search
        distances, indices = self.faiss_index.search(query_vec, min(k, self.faiss_index.ntotal))
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx in self.document_store:
                doc = self.document_store[idx]
                results.append({
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": float(1 / (1 + dist)),  # Convert distance to similarity
                    "source": "dense"
                })
        
        return results
    
    async def _elasticsearch_search(self, query: str, size: int, 
                                  filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search Elasticsearch index"""
        # Build query
        es_query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["text^2", "title", "abstract"],
                            "type": "best_fields"
                        }
                    }
                ]
            }
        }
        
        # Add filters
        if filters:
            filter_clauses = []
            for field, value in filters.items():
                filter_clauses.append({"term": {field: value}})
            es_query["bool"]["filter"] = filter_clauses
        
        # Execute search
        try:
            response = await self.es_client.search(
                index="orion_materials",
                body={
                    "query": es_query,
                    "size": size,
                    "_source": ["text", "title", "metadata"]
                }
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "text": hit["_source"].get("text", ""),
                    "metadata": hit["_source"].get("metadata", {}),
                    "score": hit["_score"],
                    "source": "sparse"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []
    
    def _hybrid_fusion(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """Fuse dense and sparse results with weighted scores"""
        # Create score dictionaries
        dense_scores = {r["text"]: r["score"] for r in dense_results}
        sparse_scores = {r["text"]: r["score"] for r in sparse_results}
        
        # Get all unique texts
        all_texts = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        # Compute hybrid scores
        fused_results = []
        for text in all_texts:
            dense_score = dense_scores.get(text, 0)
            sparse_score = sparse_scores.get(text, 0)
            
            # Normalize scores
            max_dense = max(dense_scores.values()) if dense_scores else 1
            max_sparse = max(sparse_scores.values()) if sparse_scores else 1
            
            norm_dense = dense_score / max_dense if max_dense > 0 else 0
            norm_sparse = sparse_score / max_sparse if max_sparse > 0 else 0
            
            # Weighted combination
            hybrid_score = self.alpha * norm_dense + (1 - self.alpha) * norm_sparse
            
            # Get metadata from either source
            metadata = {}
            for r in dense_results + sparse_results:
                if r["text"] == text:
                    metadata = r.get("metadata", {})
                    break
            
            fused_results.append({
                "text": text,
                "metadata": metadata,
                "score": hybrid_score,
                "dense_score": norm_dense,
                "sparse_score": norm_sparse
            })
        
        # Sort by hybrid score
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        
        return fused_results
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using cross-encoder"""
        if not self.cross_encoder:
            return results
        
        # Extract texts
        texts = [r["text"] for r in results]
        
        # Rerank
        reranked = self.cross_encoder.rerank(query, texts)
        
        # Update scores
        reranked_results = []
        for text, score in reranked:
            for r in results:
                if r["text"] == text:
                    r_copy = r.copy()
                    r_copy["rerank_score"] = score
                    r_copy["score"] = score  # Use rerank score as final score
                    reranked_results.append(r_copy)
                    break
        
        return reranked_results
    
    async def generate_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate response using RAG"""
        # Analyze query
        query_analysis = await self.analyze_query(query, context)
        
        # Search for relevant context
        search_results = await self.search(query, num_results=self.top_k)
        
        # Build prompt with context
        prompt = self._build_prompt(query, search_results["results"], query_analysis)
        
        # Generate response (placeholder - would use LLM)
        response_text = f"Based on the search results for '{query}', here is the response..."
        
        # Extract references
        references = []
        for i, result in enumerate(search_results["results"][:5]):
            references.append({
                "id": i + 1,
                "text": result["text"][:200] + "...",
                "metadata": result.get("metadata", {}),
                "score": result.get("score", 0)
            })
        
        return {
            "query": query,
            "response": response_text,
            "references": references,
            "confidence": self._calculate_confidence(search_results["results"]),
            "search_results": search_results
        }
    
    def _build_prompt(self, query: str, results: List[Dict], query_analysis: Dict) -> str:
        """Build prompt for LLM with retrieved context"""
        # System prompt
        system_prompt = """You are ORION's expert retrieval assistant. Use the following context to answer user queries on materials and processes. Provide citations with [source_id]."""
        
        # Context injection
        context_parts = []
        for i, result in enumerate(results[:5]):
            context_parts.append(f"[{i+1}]: {result['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Build full prompt
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"
        
        return prompt
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence score based on search results"""
        if not results:
            return 0.0
        
        # Factors: top score, score distribution, number of results
        top_score = results[0].get("score", 0) if results else 0
        avg_score = np.mean([r.get("score", 0) for r in results[:5]]) if len(results) >= 5 else top_score
        
        # Confidence based on top score and consistency
        confidence = top_score * 0.6 + avg_score * 0.4
        
        return min(confidence, 1.0)
    
    async def update_indices(self, documents: List[Dict[str, Any]]):
        """Update RAG indices with new documents"""
        # Chunk documents
        chunks = []
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
        
        # Update FAISS index
        if chunks:
            embeddings = self.embedding_model.encode(
                [c["text"] for c in chunks],
                convert_to_tensor=True,
                show_progress_bar=True
            )
            
            # Add to FAISS
            embeddings_np = embeddings.cpu().numpy()
            start_idx = self.doc_id_counter
            
            self.faiss_index.add(embeddings_np)
            
            # Update document store
            for i, chunk in enumerate(chunks):
                self.document_store[start_idx + i] = chunk
            
            self.doc_id_counter += len(chunks)
        
        # Update Elasticsearch
        if self.es_client and chunks:
            await self._update_elasticsearch(chunks)
        
        logger.info(f"Updated indices with {len(chunks)} chunks from {len(documents)} documents")
    
    def _chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk document into smaller pieces"""
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        
        # Simple chunking (would use more sophisticated method in production)
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": i // (self.chunk_size - self.chunk_overlap),
                    "document_id": document.get("id", "unknown")
                }
            })
        
        return chunks
    
    async def _update_elasticsearch(self, chunks: List[Dict[str, Any]]):
        """Update Elasticsearch index"""
        try:
            # Bulk index
            actions = []
            for chunk in chunks:
                actions.append({
                    "index": {
                        "_index": "orion_materials",
                        "_id": f"{chunk['metadata'].get('document_id')}_{chunk['metadata'].get('chunk_id')}"
                    }
                })
                actions.append({
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                })
            
            if actions:
                await self.es_client.bulk(body=actions)
                
        except Exception as e:
            logger.error(f"Failed to update Elasticsearch: {e}")
    
    async def enhance_search_results(self, query: str, kg_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance knowledge graph results with RAG context"""
        enhanced_results = []
        
        for kg_result in kg_results:
            # Search for additional context
            material_name = kg_result.get("name", "")
            if material_name:
                context_search = await self.search(
                    f"{material_name} properties synthesis applications",
                    num_results=3
                )
                
                # Add context to result
                kg_result["rag_context"] = {
                    "additional_info": [r["text"] for r in context_search["results"]],
                    "confidence": context_search.get("confidence", 0)
                }
            
            enhanced_results.append(kg_result)
        
        return enhanced_results