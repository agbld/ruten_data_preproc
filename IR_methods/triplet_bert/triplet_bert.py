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
path_to_activate_item_folder = '/mnt/E/Datasets/Ruten/item/activate_item'
K = 50
# test args
TOP_N = None
N_ROWS = None

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
# declare get topk fucntion
def get_topk(queries, query_embeddings, path_to_items_file, k=10, model=model, batch_size=2048, normalize_embeddings=True, show_progress_bar=True, rows=None):
    if rows is not None:
        items = pd.read_parquet(path_to_items_file, columns=['G_NAME'])[:rows]
    else:
        items = pd.read_parquet(path_to_items_file, columns=['G_NAME'])
    items['G_NAME'] = items['G_NAME'].map(html.unescape)

    with autocast():
        item_embeddings = model.encode(
            sentences=items['G_NAME'].tolist(),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

    # print('item_embeddings.shape', item_embeddings.shape)

    scores = np.dot(query_embeddings, item_embeddings.T)
    topk_idx = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    topk_scores = np.take_along_axis(scores, topk_idx, axis=1)

    results = []
    for i in range(len(queries)):
        for j in range(k):
            results.append({'query': queries[i], 'G_NAME': items['G_NAME'].values[topk_idx[i][j]], 'score': topk_scores[i][j]})

    return pd.DataFrame(results)

#%%
# get all path of item files. check if the path is .parquet file.
path_to_item_files = [os.path.join(path_to_activate_item_folder, file) for file in os.listdir(path_to_activate_item_folder) if file.endswith('.parquet')]
path_to_item_files = path_to_item_files[:TOP_N]
print(f'\nnumber of item files: {len(path_to_item_files)}')

#%%
total_results = []
for i in range(len(path_to_item_files)):
    print(f'\nprocessing file {i+1}/{len(path_to_item_files)} ({path_to_item_files[i].split("/")[-1]})')
    total_results.append(get_topk(
        queries=selected_queries_filter['query'].tolist(),
        query_embeddings=query_embeddings,
        path_to_items_file=path_to_item_files[i],
        k=K,
        model=model,
        rows=N_ROWS,
    ))

total_results = pd.concat(total_results) 
total_results = total_results.sort_values(['query', 'score'], ascending=False).groupby('query').head(50)
total_results.to_csv(f'../../outputs/query_results_{save_model_path.split("/")[-1]}.csv', index=False)

#%%
