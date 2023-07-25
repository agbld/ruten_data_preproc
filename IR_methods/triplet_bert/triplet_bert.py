#%%
# import packages
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast
import os

from utils.sentence_transformer_custom import SentenceTransformerCustom

save_model_path = 'E:/Models/ICL/cw/exp11_ECom-BERT_xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# writer_path = os.path.join(save_model_path, 'eval')
tokenizer_pretrain_model_path = 'E:/Models/ICL/cw/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch'

#%%
# initialize model
model = SentenceTransformerCustom(model_name_or_path=save_model_path, 
                                  tokenizer_pretrain_model_path=tokenizer_pretrain_model_path, 
                                  device='cuda:0')

#%%
# get query embeddings
selected_queries_filter = pd.read_csv('../../outputs/selected_queries_filter.csv')

query_embeddings = model.encode(
    sentences=selected_queries_filter['query'].tolist(),
    batch_size=512,
    normalize_embeddings=True,
    show_progress_bar=True,
)

#%%
# read items file, get embeddings, and return top-k items by dot product.
# def get_top_k_items(query_embeddings, items_file, k=10):
items_file = '../../dataset/item/activate_item/part-00006-bf5967a7-8415-4f74-85a8-1a4661ff6f2d-c000.snappy.parquet'
k = 10

items = pd.read_parquet(items_file, columns=['G_NAME'])

with autocast():
    item_embeddings = model.encode(
        sentences=items['G_NAME'].tolist(),
        batch_size=2048,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

item_scores = query_embeddings.dot(item_embeddings.T)
top_k_idx = item_scores.argsort(axis=1)[:, -k:]
top_k_scores = item_scores[np.arange(item_scores.shape[0])[:, None], top_k_idx]
top_k_items = items.iloc[top_k_idx.flatten()].reset_index(drop=True)
top_k_items['score'] = top_k_scores.flatten()
# return top_k_items

#%%
