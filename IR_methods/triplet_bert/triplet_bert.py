#%%
# import packages
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast
import os
import html
from argparse import ArgumentParser

from utils.sentence_transformer_custom import SentenceTransformerCustom

#%%
# arg parse
try:
    # parse args
    parser = ArgumentParser()
    parser.add_argument('--save_model_path', type=str, default=None)
    parser.add_argument('--tokenizer_pretrain_model_path', type=str, default='/mnt/E/Models/ICL/cw/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch')
    parser.add_argument('--path_to_activate_item_folder', type=str, default='/mnt/E/Datasets/Ruten/item/activate_item')
    parser.add_argument('--output_folder', type=str, default='./outputs/query_results')
    parser.add_argument('--path_to_queries_file', type=str, default='./outputs/picked_queries_filtered.csv')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--TOP_N', type=int, default=None)
    parser.add_argument('--N_ROWS', type=int, default=None)
    args = parser.parse_args()
    
    # assign args
    save_model_path = args.save_model_path
    tokenizer_pretrain_model_path = args.tokenizer_pretrain_model_path
    path_to_activate_item_folder = args.path_to_activate_item_folder
    output_folder = args.output_folder
    path_to_queries_file = args.path_to_queries_file
    batch_size = args.batch_size
    K = args.K
    TOP_N = args.TOP_N
    N_ROWS = args.N_ROWS

except:
    save_model_path = '/mnt/E/Models/ICL/cw/exp11_ECom-BERT_xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
    # save_model_path = '/mnt/E/Models/ICL/cw/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch'
    tokenizer_pretrain_model_path = '/mnt/E/Models/ICL/cw/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch'
    path_to_activate_item_folder = '/mnt/E/Datasets/Ruten/item/activate_item'
    output_folder = './outputs/query_results'
    path_to_queries_file = './outputs/picked_queries_filtered.csv'
    batch_size = 2048
    K = 50
    # test args
    TOP_N = 10
    N_ROWS = 1000

#%%
# initialize model
model = SentenceTransformerCustom(model_name_or_path=save_model_path, 
                                  tokenizer_pretrain_model_path=tokenizer_pretrain_model_path, 
                                  device='cuda:0')

#%%
# get query embeddings
selected_queries_filter = pd.read_csv(path_to_queries_file)

query_embeddings = model.encode(
    sentences=selected_queries_filter['query'].tolist(),
    batch_size=batch_size,
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
    tmp_output_path = output_folder + f'/tmp/query_results__{save_model_path.split("/")[-1]}__{path_to_item_files[i].split("/")[-1].split(".")[0]}.parquet'
    
    # check if the ith file is processed. if yes, load the processed file from tmp folder.
    if os.path.exists(tmp_output_path):
        tmp_df = pd.read_parquet(tmp_output_path)
        total_results.append(tmp_df)
        continue
        
    tmp_df = get_topk(
        queries=selected_queries_filter['query'].tolist(),
        query_embeddings=query_embeddings,
        path_to_items_file=path_to_item_files[i],
        k=K,
        model=model,
        rows=N_ROWS,
        batch_size=batch_size,
    )
    tmp_df.to_parquet(tmp_output_path, index=False)
    total_results.append(tmp_df)

total_results = pd.concat(total_results) 
total_results = total_results.sort_values(['query', 'score'], ascending=False).groupby('query').head(50)
total_results.to_csv(os.path.join(output_folder, f'query_results_{save_model_path.split("/")[-1]}.csv'), index=False)

#%%
