#%%
import pandas as pd
import os

#%%
path_to_query_results_folder = '../outputs/query_results'

#%%
path_to_query_results_files = [os.path.join(path_to_query_results_folder, file) for file in os.listdir(path_to_query_results_folder) if file.endswith('.csv')]

# read all query results files into a list of dataframes
query_results = []
for file in path_to_query_results_files:
    query_results.append(pd.read_csv(file, usecols=['query', 'G_NAME']))

# merge all dataframes into one. keep only the rows that apears in all dataframes
query_results = pd.concat(query_results).drop_duplicates(['query', 'G_NAME'])

query_results.to_csv('../outputs/query_results.csv', index=False)

#%%