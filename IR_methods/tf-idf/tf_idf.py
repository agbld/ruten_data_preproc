#%%
from tqdm import tqdm
import numpy as np
import pandas as pd
import requests
import jieba

#%%
# # load test query
# test_query_df = pd.read_csv('../../../../../PChome_datasets/search/pchome_test_collection/round1/test_query/test_query_250.csv')

# load valid query
test_query_df = pd.read_csv('../../../../../PChome_datasets/search/pchome_test_collection/round1/valid_query/valid_query_200.csv')

# product item
item_df = pd.read_parquet('../../../../../PChome_datasets/search/pchome_test_collection/round0/product_collection/product_collection_lg.parquet')

#%%
# class_name = ['brand','name','type','p-other']
class_name = ['type','brand','p-other']
for each_name in class_name:
    jieba.load_userdict('./Lexicon_merge/{}.txt'.format(each_name))

#%%
# 1. jieba tokenizer
def jieba_tokenizer(x):
    tokens = jieba.lcut(x, cut_all=False)
    stop_words = ['【','】','/','~','＊','、','（','）','+','‧',' ','']
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

#%%
jieba_tokenizer('3090')

#%%
#Import TfIdfVectorizer from the scikit-learn library
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stopwords
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", tokenizer=jieba_tokenizer, ngram_range=(1,2))

#Replace NaN with an empty string
#item_tokens_df['tokens'] = item_tokens_df['tokens'].fillna('')
item_df['name'] = item_df['name'].fillna('')

#%%
#Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
tfidf_matrix = tfidf.fit_transform(item_df['name'])
#Output the shape of tfidf_matrix
tfidf_matrix.shape

#%%
# #Define a TF-IDF Vectorizer Object. Remove all english stopwords
# tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", tokenizer=jieba_tokenizer, ngram_range=(1,2))

#Define a TF-IDF Vectorizer Object. Remove all english stopwords
tfidf_char = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", analyzer='char')

#%%
#Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
tfidf_matrix_char = tfidf_char.fit_transform(item_df['name'])
#Output the shape of tfidf_matrix
tfidf_matrix_char.shape

#%%
from sklearn.metrics.pairwise import cosine_similarity
def search(query):
    score = np.zeros(tfidf_matrix.shape[1])
    que_tfidf = tfidf.transform([query]) # sparse array
    scores = cosine_similarity(que_tfidf,tfidf_matrix)
    top_50_indices = np.argsort(-scores[0])[:50]
    sum_of_score = sum(scores[0])
    # print("sum_of_score: ", sum_of_score)
    
    # if sum_of_score < 15 then using model two search again
    if sum_of_score < 10 : 
    # if True : 
        # print('using model 2')
        que_tfidf = tfidf_char.transform([query]) # sparse array
        scores = cosine_similarity(que_tfidf,tfidf_matrix_char)
        top_50_indices = np.argsort(-scores[0])[:50]
        sum_of_score = sum(scores[0])
        
        return sum_of_score, top_50_indices
    
    return sum_of_score, top_50_indices

#%%
# query = test_query_df.iloc[112]['query']
# query = test_query_df.iloc[225]['query']
# query = test_query_df.iloc[225]['query']
query = '禮物'

#%%
sum_of_score, top_50_indices = search(query)
print("query: ", query)
print("sum_of_score: ", sum_of_score)
item_df.loc[top_50_indices]

#%%
for i, r in tqdm(enumerate(test_query_df.iloc)):
    query = r['query']
    sum_of_score, top_50_indices = search(query)
    results = item_df.loc[top_50_indices]
    # results = results.drop('sign_id', axis=1).reset_index(drop=True)

    # save result files for pooling step(id start from 250)
    results.to_parquet('./results/results_round1_valid_query/result_'+str(r['query_id']).zfill(3)+'.parquet')
    
#%%
N = 0
print("query: ",test_query_df.iloc[N]['query'])
top_50 = pd.read_parquet('./results/results_round1_valid_query/result_'+str(N).zfill(3)+'.parquet')
top_50

#%%
for i in range(50):
    query = jieba_tokenizer(test_query_df.iloc[i]['query'])
    sum_of_score, top_50_indices = search(query)
    print(i, sum_of_score)
    if sum_of_score == 0:
        print(i," Document Not Found Error")
        
#%%
merge = []
for i in tqdm(item_df['name']):
    merge = list(set(merge + list(i)))
    
#%%
for i in range(10):
    print(test_query_df.iloc[i]['query'])