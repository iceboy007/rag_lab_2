import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer
import os

# 日志
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='将SNOMED数据向量化并插入FAISS数据库')
    parser.add_argument('--input', type=str, default='backend/data/SNOMED_5000.csv', help='输入CSV文件路径')
    parser.add_argument('--faiss_index', type=str, default='IndexFlatIP', help='FAISS索引类型，如IndexFlatL2, IndexFlatIP, IndexHNSWFlat')
    parser.add_argument('--output', type=str, default='backend/db/snomed_bge_m3_faiss.index', help='FAISS索引保存路径')
    parser.add_argument('--batch_size', type=int, default=1024, help='批处理大小')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-m3', help='用于向量化的模型名')
    return parser.parse_args()

def get_faiss_index(index_type, dim):
    if index_type == 'IndexFlatL2':
        return faiss.IndexFlatL2(dim)
    elif index_type == 'IndexFlatIP':
        return faiss.IndexFlatIP(dim)
    elif index_type == 'IndexHNSWFlat':
        return faiss.IndexHNSWFlat(dim, 32)
    else:
        raise ValueError(f'不支持的FAISS索引类型: {index_type}')

def main():
    args = parse_args()
    file_path = args.input
    index_type = args.faiss_index
    output_path = args.output
    batch_size = args.batch_size
    model_name = args.model_name

    # 加载数据，只保留 concept_name 和 concept_class_id 两列
    logging.info(f'Loading data from {file_path}')
    df = pd.read_csv(file_path, dtype=str, usecols=['concept_name', 'concept_class_id']).fillna('NA')

    # 加载模型
    logging.info(f'Loading embedding model: {model_name}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    # 取样本获取向量维度
    sample_text = df.iloc[0]['concept_name']
    sample_vec = model.encode([sample_text])[0]
    dim = len(sample_vec)
    logging.info(f'Embedding dimension: {dim}')

    # 创建FAISS索引
    index = get_faiss_index(index_type, dim)
    id_map = []  # 记录 concept_name, concept_class_id

    # 批量处理
    for start_idx in tqdm(range(0, len(df), batch_size), desc='Processing batches'):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        docs = batch_df['concept_name'].tolist()
        try:
            embeddings = model.encode(docs, show_progress_bar=False, batch_size=64, device=device)
        except Exception as e:
            logging.error(f'Error generating embeddings for batch {start_idx // batch_size + 1}: {e}')
            continue
        embeddings = np.array(embeddings).astype('float32')
        index.add(embeddings)
        id_map.extend(zip(batch_df['concept_name'], batch_df['concept_class_id']))

    # 保存索引
    faiss.write_index(index, output_path)
    logging.info(f'Saved FAISS index to {output_path}')

    # 保存id_map，每行 concept_name,concept_class_id
    id_map_path = output_path + '.idmap.txt'
    with open(id_map_path, 'w', encoding='utf-8') as f:
        for cname, cclass in id_map:
            f.write(f'{cname},{cclass}\n')
    logging.info(f'Saved id map to {id_map_path}')

    # 示例查询
    query = df.iloc[0]['concept_name']
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, 5)
    logging.info(f'Query result for "{query}":')
    for rank, idx in enumerate(I[0]):
        if idx < len(id_map):
            cname, cclass = id_map[idx]
            logging.info(f'Rank {rank+1}: concept_name={cname}, concept_class_id={cclass}, score={D[0][rank]}')
        else:
            logging.info(f'Rank {rank+1}: idx={idx}, score={D[0][rank]}')

if __name__ == '__main__':
    main() 