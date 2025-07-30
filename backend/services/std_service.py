import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StdService:
    """
    基于FAISS的术语标准化服务
    """
    def __init__(self,
                 model="../BAAI/bge-m3",
                 db_path="db/snomed_bge_m3_faiss.db",
                 idmap_path="db/snomed_bge_m3_faiss.db.idmap.txt",
                 top_k=5):
        self.db_path = db_path
        self.idmap_path = idmap_path
        self.top_k = top_k
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model, trust_remote_code=True, device=self.device)
        self.index = faiss.read_index(self.db_path)
        self.id_map = self._load_id_map(self.idmap_path)

    def _load_id_map(self, idmap_path):
        id_map = []
        with open(idmap_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    id_map.append({'concept_name': parts[0], 'concept_class_id': parts[1]})
        return id_map

    def search_similar_terms(self, query: str, limit: int = 5) -> List[Dict]:
        query_vec = self.model.encode([query]).astype('float32')
        D, I = self.index.search(query_vec, limit)
        results = []
        for rank, idx in enumerate(I[0]):
            if idx < len(self.id_map):
                entry = self.id_map[idx]
                results.append({
                    'concept_name': entry['concept_name'],
                    'concept_class_id': entry['concept_class_id'],
                    'score': float(D[0][rank])
                })
        return results