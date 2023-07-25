#%%
# import packages
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast
import os
import html

from utils.sentence_transformer_custom import SentenceTransformerCustom

save_model_path = '/mnt/E/Models/ICL/cw/exp11_ECom-BERT_xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# writer_path = os.path.join(save_model_path, 'eval')
tokenizer_pretrain_model_path = '/mnt/E/Models/ICL/cw/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch'

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

print('query_embeddings.shape', query_embeddings.shape)

#%%
# read items file, get embeddings.
items_file = '/mnt/E/Datasets/Ruten/item/activate_item/part-00006-bf5967a7-8415-4f74-85a8-1a4661ff6f2d-c000.snappy.parquet'
k = 10

# read items file, get embeddings.
items = pd.read_parquet(items_file)[:]
items['G_NAME'] = items['G_NAME'].map(html.unescape)

#%%
with autocast():
    item_embeddings = model.encode(
        sentences=items['G_NAME'].tolist(),
        batch_size=2048,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

print('item_embeddings.shape', item_embeddings.shape)

#%%
# INPUT: selected_queries_filter['query'].values, query_embeddings, items['G_NAME'].values, item_embeddings
# RETURN: a dataframe with columns:['query', 'G_NAME', 'score'] from given query. 
# def get_topk(queries, query_embeddings, items, item_embeddings, k=10):

scores = np.dot(query_embeddings, item_embeddings.T)
topk_idx = np.argsort(scores, axis=1)[:, ::-1][:, :k]
topk_scores = np.take_along_axis(scores, topk_idx, axis=1)

results = []
for i in range(len(selected_queries_filter['query'].values)):
    for j in range(k):
        results.append({'query': selected_queries_filter['query'].values[i], 'G_NAME': items['G_NAME'].values[topk_idx[i][j]], 'score': topk_scores[i][j]})

results = pd.DataFrame(results)

#%%
