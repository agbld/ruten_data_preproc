#%%
# import packages
import pandas as pd
from tqdm import tqdm
from random import choice
from argparse import ArgumentParser
import numpy as np

#%%
# arg parse
try:
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='./outputs/merged_result.parquet', help='path to merged_result.parquet')
    parser.add_argument('--output', type=str, default='./outputs/picked_queries.csv', help='path to output csv file')
    parser.add_argument('--num_of_parts', type=int, default=10, help='number of parts to split queries')
    parser.add_argument('--num_of_pairs_per_part', type=int, default=100, help='number of pairs to draw per part')
    args = parser.parse_args()

    path_to_query_item_pairs = args.path
    path_to_output = args.output
    NUM_OF_PARTS = args.num_of_parts
    NUM_OF_PAIRS_PER_PART = args.num_of_pairs_per_part
except:
    path_to_query_item_pairs = './outputs/query_item_pairs.parquet'
    path_to_output = './outputs/picked_queries.csv'
    NUM_OF_PARTS = 10
    NUM_OF_PAIRS_PER_PART = 20
    
#%%
# read query item pairs as df
query_item_pairs_df = pd.read_parquet(path_to_query_item_pairs, columns=['query', 'query_counts', 'G_NAME'])[:]
query_item_pairs_df = query_item_pairs_df.drop_duplicates()

#%%
# prepare queries_df for splitting queries into parts

# create queries_df
queries_df = query_item_pairs_df.drop(columns='G_NAME').drop_duplicates()

# Calculate the total sum of query_counts
total_sum = queries_df['query_counts'].sum()

# Determine the desired sum for each part
desired_sum = total_sum // NUM_OF_PARTS

# Sort the DataFrame by counts in descending order
queries_df = queries_df.sort_values(by='query_counts')


#%%
# Initialize a dictionary to store the separated parts
parts = {}

# Iterate over the DataFrame to create the parts
def convert_to_suffix(num):
    suffixes = ['', 'k', 'm', 'b', 't']  # 可自行擴展更大的後綴
    magnitude = 0

    while abs(num) >= 1000 and magnitude < len(suffixes) - 1:
        magnitude += 1
        num //= 1000

    return f"{num}{suffixes[magnitude]}"

current_part = []
current_sum = 0
cumulative_sum = 0
part_key_start = convert_to_suffix(int(queries_df.iloc[0]['query_counts']))
part_key_end = 0
part_key = 0
parts_info = []

with tqdm(total=queries_df.shape[0]) as pbar:
    for index, row in queries_df.iterrows():
        if (current_sum + row['query_counts'] <= desired_sum) or (len(parts) == NUM_OF_PARTS - 1):
            current_part.append(row['query'])
            current_sum += row['query_counts']
            cumulative_sum += row['query_counts']
        else:
            # add current_part to parts
            part_key_end = convert_to_suffix(int(row['query_counts']))
            part_key = f'{part_key_start}-{part_key_end}'
            parts[part_key] = current_part
            part_key_start = part_key_end
            parts_info.append({'part_key': part_key, 
                               '# unique queries': len(current_part), 
                               '# queries': current_sum, 
                               '# cumulated queries': cumulative_sum, 
                               '# sampled queries': NUM_OF_PAIRS_PER_PART})
            
            # create a new part
            current_part = []
            current_part.append(row['query'])
            current_sum = row['query_counts']
            cumulative_sum += row['query_counts']

        pbar.update(1)

    # add current_part to parts (last part)
    part_key_end = convert_to_suffix(int(row['query_counts']))
    part_key = f'{part_key_start}-{part_key_end}'
    parts[part_key] = current_part
    part_key_start = part_key_end
    parts_info.append({'part_key': part_key, 
                       '# unique queries': len(current_part), 
                       '# queries': current_sum, 
                       '# cumulated queries': cumulative_sum,
                       '# sampled queries': NUM_OF_PAIRS_PER_PART})

    parts_info_df = pd.DataFrame(parts_info)
    parts_info_df.to_csv('./outputs/parts_info.csv', index=False)
            
#%%
selected_queries = []
with tqdm(total=len(parts)) as pbar:
    for part_key, queries in parts.items():
        part_queries = np.random.choice(queries, size=NUM_OF_PAIRS_PER_PART, replace=False)
        part_queries = pd.DataFrame(part_queries, columns=['query'])
        part_queries['part_key'] = part_key
        selected_queries.append(part_queries)
        pbar.update(1)
selected_queries = pd.concat(selected_queries)

# save selected_pairs
selected_queries.to_csv(path_to_output, index=False)

# %%
