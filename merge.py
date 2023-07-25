#%%
# import packages
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
from math import ceil
from argparse import ArgumentParser

#%%
# declare functions

def map_item_name(path_to_item_files: list, queries_df: pd.DataFrame, verbose: bool = False, print_error: bool = False) -> pd.DataFrame: 
    queries_df['G_NAME'] = None
    num_of_unique_gno = queries_df['gno'].nunique()
    error_files = []

    avg_ram_usage = 0
    with tqdm(total=len(path_to_item_files), desc='Mapping G_NAME: ', disable=not verbose) as pbar:
        for file in path_to_item_files:
            try:
                item_file_df = pd.read_parquet(file, columns=['G_NAME', 'G_NO'])
            except:
                error_files.append(file)        
                pbar.update(1)
                continue
            
            # find all gno (G_NO) in queries_df that is also in item_file_df and save into a list called gno_list.
            gno_list = queries_df.loc[queries_df['gno'].isin(item_file_df['G_NO']), 'gno'].values

            # map G_NAME from item_file_df to queries_df using the column "gno" from queries_df and "G_NO" from item_file_df as the key. use gno_list for faster speed.
            queries_df.loc[queries_df['gno'].isin(gno_list), 'G_NAME'] = queries_df.loc[queries_df['gno'].isin(gno_list), 'gno'].map(item_file_df.set_index('G_NO')['G_NAME'])
            
            # update the total memory usage of queries_df plus item_file_df with tqdm.set_postfix.
            ram_usage = queries_df.memory_usage(deep=True).sum() / 1024 / 1024 + item_file_df.memory_usage(deep=True).sum() / 1024 / 1024
            ratio_of_mapped_G_NAME = (queries_df['G_NAME'].nunique() - 1) / num_of_unique_gno * 100 # TODO: incorrect ratio.
            avg_ram_usage += ram_usage
            
            pbar.set_postfix_str({'mapped_ratio': f'{ratio_of_mapped_G_NAME:.2f}%', 'ram_usage': f'{ram_usage:.2f}'})
            pbar.update(1)
    
    if print_error:
        print(error_files)
        
    queries_df = queries_df[queries_df['G_NAME'].notnull()]
    
    # queries_df.drop(columns=['gno'], inplace=True)
    
    queries_df = queries_df.sort_values('G_NAME').drop_duplicates()
    
    return queries_df, avg_ram_usage / len(path_to_item_files)

def map_item_name_multi(args):
    return map_item_name(*args)

#%%
# arg parse
if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        parser.add_argument('--activate_item_folder', 
                            type=str, 
                            default='./dataset/item/activate_item', 
                            help='path to activate_item folder')
        parser.add_argument('--concat_query_file', 
                            type=str, 
                            default='./dataset/trace/one-month-search-trace_v2.parquet', 
                            help='path to concat_query_file')
        parser.add_argument('--output_folder',
                            type=str,
                            default='./outputs',
                            help='path to output folder')

        # number os processes to run mapping.
        # use 1 for single-processing. 
        # RAM usage will increase linearly with the number of processes.
        parser.add_argument('--num_of_processes', type=int, default=1, help='number of processes to run mapping')
        parser.add_argument('--chunk_size', type=int, default=15, help='number of files to run mapping per process')
        parser.add_argument('--top_n', type=int, default=None, help='for testing. set to None to run all files')
        args = parser.parse_args()

        path_to_activate_item_folder = args.activate_item_folder
        path_to_concat_query_file = args.concat_query_file
        path_to_output_folder = args.output_folder
        NUM_OF_PROCESSES = args.num_of_processes # int(ceil(cpu_count() / 2)) 
        TOP_N = args.top_n
        CHUNK_SIZE = args.chunk_size
    except:
        print('\nnot execute from CLI, using default arguments...')
        path_to_activate_item_folder = './dataset/item/activate_item'
        path_to_concat_query_file = './dataset/trace/one-month-search-trace_v2.parquet'
        path_to_output_folder = './outputs'
        NUM_OF_PROCESSES = 4
        TOP_N = 20
        CHUNK_SIZE = 5

    start = time.time()

    # read/preproc queries_df
    print('reading queries_df...')
    queries_df = pd.read_parquet(path_to_concat_query_file, columns=['gno', 'query'])
    query_counts = queries_df['query'].value_counts()
    queries_df['query_counts'] = queries_df['query'].map(query_counts)
    queries_df = queries_df.drop_duplicates()
    queries_df = queries_df[~(queries_df['query'] == '')]   # remove rows with empty query

    # get all path of item files. check if the path is .parquet file.
    path_to_item_files = [os.path.join(path_to_activate_item_folder, file) for file in os.listdir(path_to_activate_item_folder) if file.endswith('.parquet')]
    if TOP_N is not None:   # for testing
        path_to_item_files = path_to_item_files[:TOP_N]
    print(f'\nnumber of item files: {len(path_to_item_files)}')
    
    # map item names
    if NUM_OF_PROCESSES > 1:
        # split the path_to_item_files into chunks
        path_to_item_files_chunks = [path_to_item_files[i:i+CHUNK_SIZE] for i in range(0, len(path_to_item_files), CHUNK_SIZE)]
        print(f'\nsplit into {len(path_to_item_files_chunks)} chunks for multi-processing...')
        
        # build arguments for map_item_name
        args = [(path_to_item_files_chunk, queries_df, False, False) for path_to_item_files_chunk in path_to_item_files_chunks]

        # run map_item_name with multi-processing
        num_of_parts = int(ceil(len(args) / NUM_OF_PROCESSES)) # do it in parts to avoid memory error
        query_item_pairs_df = None
        for i in range(num_of_parts):
            print(f'\nmapping item names with multi-processing ({i+1}/{num_of_parts})...')
            
            with Pool(processes=NUM_OF_PROCESSES) as pool:
                args_part = args[i*NUM_OF_PROCESSES:(i+1)*NUM_OF_PROCESSES]
                results = list(tqdm(pool.imap(map_item_name_multi, args_part), total=len(args_part)))
        
            # calculate/print total ram usage of results
            total_ram_usage = 0
            for result in results:
                # total_ram_usage += result.memory_usage(deep=True).sum() / 1024 / 1024
                total_ram_usage += result[1]
            print(f'avg_ram_usage: {total_ram_usage:.2f} MB')
        
            # reduce results from multi-processing
            print('reducing results...')
            if query_item_pairs_df is None:
                query_item_pairs_df = pd.concat(list(r[0] for r in results)).sort_values('G_NAME', ascending=False)
                query_item_pairs_df = query_item_pairs_df.drop_duplicates(keep='first')
            else:
                query_item_pairs_df = pd.concat([query_item_pairs_df] + list(r[0] for r in results)).sort_values('G_NAME', ascending=False)
                query_item_pairs_df = query_item_pairs_df.drop_duplicates(keep='first')
        
    else:
        # run map_item_name with single-processing
        print('mapping G_NAME with single-processing...')
        query_item_pairs_df, avg_ram_usage = map_item_name(path_to_item_files, queries_df.copy(deep=True), True, True)

    ratio_of_mapped_items = (query_item_pairs_df['G_NAME'].nunique()) / queries_df['gno'].nunique() * 100
    print(f'\nratio_of_mapped_items: {ratio_of_mapped_items:.2f}%') # TODO: incorrect ratio.
    
    # save query_item_pairs_df
    print('\nsaving query_item_pairs_df...')
    query_item_pairs_df.to_parquet(f'{path_to_output_folder}/merged_result.parquet', index=False)
    
    end = time.time()
    print(f'\nTime elapsed: {end - start:.2f} seconds')
    print('done')

#%%