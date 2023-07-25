import os 
os.chdir(r'/home/ee303/Desktop/CABRSS_chun_wei/semantic_search')
# -------------------------------------------------------
#   environment setup
# -------------------------------------------------------
# transformers==4.24.0
# sentence-transformers==2.2.0
# huggingface-hub==0.10.1
# gensim==3.8.3

# -------------------------------------------------------
#   Global
# -------------------------------------------------------
device = 'cuda:0'

# # -------------------------------------------------------
# #   Trainset
# # -------------------------------------------------------
intent_pos_sm_df_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm_pos.parquet'
intent_neg_sm_df_path = '../PChome_datasets/search/pchome_search_click_dataset/train/negative/round1_train_sm_neg.parquet'

# -------------------------------------------------------
#   Test collection
# -------------------------------------------------------
round0_test_query_path = '../PChome_datasets/search/pchome_test_collection/round0/test_query/test_query_250.csv' # round0
round1_test_query_path = '../PChome_datasets/search/pchome_test_collection/round1/test_query/test_query_250.csv' # round1

product_collection_lg_path = '../PChome_datasets/search/pchome_test_collection/round0/product_collection/product_collection_lg.parquet'
round0_product_collection_sm_path = '../PChome_datasets/search/pchome_test_collection/round0/product_collection/product_collection_sm.parquet'
round0_plus_product_collection_sm_path = '../PChome_datasets/search/pchome_test_collection/round1/product_collection/round0_product_collection_sm.parquet' # round0-plus collection_sm
round1_product_collection_sm_path = '../PChome_datasets/search/pchome_test_collection/round1/product_collection/round1_product_collection_sm.parquet' # round1

round0_qrels_path = '../PChome_datasets/search/pchome_test_collection/round0/qrels_corrected_20220412/qrels.parquet' # round0 qrels
round0_plus_qrels_path = '../PChome_datasets/search/pchome_test_collection/round1/qrels/round0_qrels.parquet' # round0-plus qrels
round1_qrels_path = '../PChome_datasets/search/pchome_test_collection/round1/qrels/round1_qrels.parquet' # round1 qrels
round0_plus_qrels_only_r0_id = '../PChome_datasets/search/pchome_test_collection/round1/qrels/round0_plus_qrels_only_r0_id.parquet'
# # -------------------------------------------------------
# #  001. bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2
# # -------------------------------------------------------
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  002. bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  002. bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round1
# # -------------------------------------------------------
# # testset
# current_test_query_path = round1_test_query_path
# current_product_collection_path = round1_product_collection_sm_path
# current_qrels_path = round1_qrels_path
# # exㄥp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round1'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #   002. bert_soft-margin-triplet-loss_train-lg_intent-based-neg-2 (THE BEST)
# # -------------------------------------------------------
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-lg_intent-based-neg-2'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/train_lg.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  0. bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2_chunwei
# # -------------------------------------------------------
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2_chunwei'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False,
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  1. bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2_ner44k
# # -------------------------------------------------------
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2_ner44k'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': True,
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  1. bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2_fine-tune-ner44k
# # -------------------------------------------------------
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2_fine-tune-ner44k'
# # network
# network = 'ner-finetune'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': True,
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = './experiments/bert_soft-margin-triplet-loss_train-sm_intent-based-neg-2'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #   2. bert_soft-margin-triplet-loss_train-sm_clear-neg-2
# # -------------------------------------------------------
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_clear-neg-2'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'clear',
#     'neg-num': 2,
#     'add_ner_trainset': False,
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #   3. bert_soft-margin-triplet-loss_train-sm_clear-neg-2_ner190k
# # -------------------------------------------------------
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_clear-neg-2_ner190k'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'clear',
#     'neg-num': 2,
#     'add_ner_trainset': True,
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #   4. bert_soft-margin-triplet-loss_train-sm_clear-neg-2_ner44k
# # -------------------------------------------------------
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_clear-neg-2_ner44k'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'clear',
#     'neg-num': 2,
#     'add_ner_trainset': True,
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  002. bert_soft-margin-triplet-loss_train-sm_ner-neg_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_ner-neg_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'ner-neg'
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  003. bert_soft-margin-triplet-loss_train-sm_ner-neg-add-random_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_ner-neg-add-random_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'ner-neg'
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  004. bert_soft-margin-triplet-loss_train-sm_ner-neg-3-without-store-add-random_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_soft-margin-triplet-loss_train-sm_ner-neg-3-without-store-add-random_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'ner-neg'
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  005. bert_dynamic-margin-triplet-loss_train-sm_ner-neg-3-add-random_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_dynamic-soft-margin-triplet-loss_train-sm_ner-neg-3-add-random_valid-on-round0-plus'
# # network
# network = 'triplet-dynamic-margin'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'ner-neg'
# }
# # loss
# loss = 'dynamic-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  006. bert_dynamic-margin-triplet-loss_train-sm_ner-neg-3-add-random_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_dynamic-soft-margin-triplet-loss_train-sm_ner-neg-3-add-random_valid-on-round0-plus'
# # network
# network = 'triplet-dynamic-margin'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'ner-neg'
# }
# # loss
# loss = 'dynamic-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  004. lce-loss
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'lce-loss_train-sm_ner-neg-without-store-add-random_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'ner-neg'
# }
# # loss
# loss = 'lce-loss'
# LCE_sample_size = 4
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  010. bert_batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  010. bert_in-batch-softmax-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_in-batch-softmax-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'in-batch-softmax-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# # pretrained_model_path = '/home/ee303/Desktop/eCom-Iris_chun_wei/semantic_search/experiments/bert_in-batch-pro-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# -------------------------------------------------------
#  010. bert_in-batch-pro-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# -------------------------------------------------------
# eval 的設定要另外改
# testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_in-batch-pro-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-triplet-loss-pro'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# -------------------------------------------------------
#  010. Kedong_bert_batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# -------------------------------------------------------
# eval 的設定要另外改
# testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# exp
# exp_name = 'Kedong_bert_batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'Kedong_train_df',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# exp_name = 'kd_bert_in-batch-soft-margin-triplet-loss_train-sm-clean_intent-based-neg-2'
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  011. bert_batch-soft-margin-triplet-loss_train-sm_valid-on-round0-plus (without intent-based-neg-2)
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_batch-soft-margin-triplet-loss_train-sm_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  011. bert_batch-soft-margin-triplet-loss_train-sm_naive-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_batch-soft-margin-triplet-loss_train-sm_naive-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm_pos.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'naive',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  011. bert_wating-mlm_batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_wating-mlm_batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  011. bert_wating-mlm_batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus_20-epoch
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'bert_wating-mlm_batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus_20-epoch'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  012. wating-mlm_bert_in-batch-soft-margin-triplet-loss_train-sm_ner-neg-2_valid-on-round0-plus (neg-2: target_store_no_df+target_rg_no_df)
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'wating-mlm_bert_in-batch-soft-margin-triplet-loss_train-sm_ner-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'ner-neg',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  012. wating-mlm_bert_in-batch-soft-margin-triplet-loss_train-sm_ner-neg-2_valid-on-round0-plus (neg-2: target_rg_no_group_df+target_rg_no_df)
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'wating-mlm_bert_in-batch-soft-margin-triplet-loss_train-sm_ner-neg-2-rg_no_group+rg_no_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'ner-neg',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  011. desc-pre-train_batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'desc-pre-train_batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'in-batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_desc_pre-train/epoch-9/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  011. desc-pre-train_in-batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'desc-pre-train_in-batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'in-batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_desc_pre-train/best_model/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  013. (未執行) wating-mlm_bert_soft-margin-triplet-loss_train-sm_ner-neg-3_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'wating-mlm_bert_soft-margin-triplet-loss_train-sm_ner-neg-3_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'ner-neg',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  010. bert_in-batch-softmax-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ttttttt'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-triplet-loss'
# # training config
# pretrained_model_path = '/home/ee303/Desktop/eCom-Iris_chun_wei/semantic_search/experiments/bert_batch-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # pretrained_model_path = '/home/ee303/Desktop/eCom-Iris_chun_wei/semantic_search/experiments/bert_in-batch-pro-soft-margin-triplet-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  010. ckip_in-batch-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ckip_in-batch-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'in-batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  010. ckip_in-batch-loss-size128_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ckip_in-batch-loss-size128_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'in-batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 128
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # ======================================================================
# #   batch size exp
# # ======================================================================

# # -------------------------------------------------------
# #  425. ckip_batch-all-loss-size4_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ckip_batch-all-loss-size4_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 4
# evaluation_steps = 200*8
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  425. ckip_batch-all-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ckip_batch-all-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  425. ckip_batch-all-loss-size128_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ckip_batch-all-loss-size128_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 128
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # ======================================================================
# #   batch size exp end
# # ======================================================================

# # -------------------------------------------------------
# #  010. ckip_xbm_batch-all-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ckip_xbm_batch-all-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# # pretrained_model_path = '/home/ee303/Desktop/eCom-Iris_chun_wei/semantic_search/experiments/ckip_batch-all-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  010. ckip_xbm_batch-all-hard-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ckip_xbm_batch-all-hard-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# # pretrained_model_path = '/home/ee303/Desktop/eCom-Iris_chun_wei/semantic_search/experiments/ckip_batch-all-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  010. ckip_xbm_batch-all-hard-current-anchor-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ckip_xbm_batch-all-hard-current-anchor-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# # pretrained_model_path = '/home/ee303/Desktop/eCom-Iris_chun_wei/semantic_search/experiments/ckip_batch-all-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  010. ckip_in-batch-softmax-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ckip_in-batch-softmax-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'intent-based',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'in-batch-softmax-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # ======================================================================
# #   ablation study
# # ======================================================================

# # -------------------------------------------------------
# #  best_model. exp0.5_ECom-BERT_xbm-epoch0_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp0.5_ECom-BERT_xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  exp1. exp1_ckip-BERT_xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp1_ckip-BERT_xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  exp2. exp2_ECom-BERT_xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp2_ECom-BERT_xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 3
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  exp3. exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  exp4. exp4_ECom-BERT_xbm_soft-margin-triplet-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp4_ECom-BERT_xbm_soft-margin-triplet-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  exp5. exp5_ECom-BERT_xbm_in-batch-triplet-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp5_ECom-BERT_xbm_in-batch-triplet-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-in-batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  exp6. exp6_ECom-BERT_xbm_in-batch-softmax-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp6_ECom-BERT_xbm_in-batch-softmax-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-in-batch-softmax-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  exp7. exp7_ECom-BERT_xbm-epoch0_batch-all-loss_train-sm_intent-based-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp7_ECom-BERT_xbm_batch-all-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  exp8. exp8_ECom-BERT_xbm-epoch0_batch-all-loss_train-lg_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp8_ECom-BERT_xbm_batch-all-loss_train-lg_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-lg-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  exp9. exp9_ECom-BERT_xbm_batch-all-loss_train-sm_naive-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp9_ECom-BERT_xbm_batch-all-loss_train-sm_naive-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'naive',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  exp9-2. exp9-2_ECom-BERT_xbm_in-batch-loss_train-sm_naive-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp9-2_ECom-BERT_xbm_in-batch-loss_train-sm_naive-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'naive',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-in-batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  exp9-2. exp9-3_ECom-BERT_xbm_batch-hard-loss_train-sm_naive-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp9-3_ECom-BERT_xbm_batch-hard-loss_train-sm_naive-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'naive',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-hard-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  exp10. exp10_ECom-BERT_wo-xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp10_ECom-BERT_wo-xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-hard-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  exp11. exp11_ECom-BERT_xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp11_ECom-BERT_xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-hard-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  exp12. exp12_ECom-BERT_xbm_batch-all-batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp12_ECom-BERT_xbm_batch-all-batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-all-batch-hard-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  exp13. ECom-BERT_batch-hard-loss-koleo_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------

# koleo_lapha = 0.3
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp13_ECom-BERT_batch-hard-loss-koleo_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-hard-triplet-loss+koleo'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  exp14. exp14_ECom-BERT_batch-hard-loss-koleo0.1_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------

# koleo_lapha = 0.1
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp14_ECom-BERT_batch-hard-loss-koleo0.1_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-hard-triplet-loss+koleo'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  exp15. exp15_ECom-BERT_contrastive-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp15_ECom-BERT_contrastive-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'soft-margin-contrastive-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  exp16. exp16_ECom-BERT_contrastive-loss_top10_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp16_ECom-BERT_contrastive-loss_top10_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'soft-margin-contrastive-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  =baseline=ckip_soft-margin-triplet-loss_train-sm_intent-based-neg-2_fix-seed (fix-random seed 2022)
# # -------------------------------------------------------
# # validset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = '=baseline=ckip_soft-margin-triplet-loss_train-sm_intent-based-neg-2_fix-seed'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  =baseline=ckip_soft-margin-triplet-loss_train-lg_intent-based-neg-2_fix-seed (fix-random seed 2022)
# # -------------------------------------------------------
# # validset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = '=baseline=ckip_soft-margin-triplet-loss_train-lg_intent-based-neg-2_fix-seed'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-lg-clean_intent-based-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # ======================================================================
# #   ablation study end
# # ======================================================================

# # # -------------------------------------------------------
# # #   5. reranking
# # # -------------------------------------------------------
# exp_name = 'bi-sm_lce-intent_candidate-200_only-rerank'
# save_model_path = './experiments/' + exp_name

# # # -------------------------------------------------------
# # #   5. psudo relevance feedback
# # # -------------------------------------------------------
# exp_name = 'PRF_bert_in-batch-softmax-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# save_model_path = './experiments/' + exp_name

# # # -------------------------------------------------------
# # #   5. test
# # # -------------------------------------------------------
# exp_name = 'mlm_pre_train_cvc/pc+momo_title+desc/5_epoch'
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  011. ttt
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ttt'
# # network
# network = 'triplet'
# # dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-neg-2',
#     'neg-num': 2,
#     'add_ner_trainset': False
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# # pretrained_model_path = '/home/ee303/Desktop/eCom-Iris_chun_wei/semantic_search/experiments/ckip_batch-all-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# exp_name = 'dssm'。


# # -------------------------------------------------------
# #  exp1. exp1_ckip-BERT_xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus-redo1
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp1_ckip-BERT_xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus-redo1'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # ======================================================================
# #   incorrect neg fine tune
# # ======================================================================
# evaluate incorrect neg fine tune 模型時, 請使用本區塊最下方的設定（帶入模型名稱）


# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test01
# # -------------------------------------------------------
# # 訓練正樣本 -> 驗證集中正確的樣本
# # 訓練負樣本 -> 驗證集中錯誤的樣本
# neg_strategy == 'only-incorrect-neg'
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test01'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test03
# # -------------------------------------------------------
# # 訓練正樣本 -> 驗證集中正確的樣本 + click_log_隨機正樣本
# # 訓練負樣本 -> 驗證集中錯誤的樣本 + click_log_隨機負樣本
# neg_strategy = 'add-random-train-sample'
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test03'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test04
# # -------------------------------------------------------
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本

# neg_strategy = 'only-incorrect-neg'
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test04'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test04--2
# # -------------------------------------------------------
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本

# neg_strategy = 'only-incorrect-neg'
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test04--2'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test05
# # -------------------------------------------------------
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本 + 3倍click_log_隨機正樣本

# neg_strategy = 'add-random-train-sample'
# N_times_neg = 3
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test05'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test05--2
# # -------------------------------------------------------
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本 + 3倍click_log_隨機正樣本

# neg_strategy = 'add-random-train-sample'
# N_times_neg = 3
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test05--2'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test05--3
# # -------------------------------------------------------
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本 + 3倍click_log_隨機正樣本

# neg_strategy = 'add-random-train-sample'
# N_times_neg = 3
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test05--3'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test06 (同exp05)
# # -------------------------------------------------------
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本 + 3倍click_log_隨機正樣本

# neg_strategy = 'add-random-train-sample'
# N_times_neg = 3
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test06'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test07
# # -------------------------------------------------------
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本 + 10倍click_log_隨機正樣本

# neg_strategy = 'add-random-train-sample'
# N_times_neg = 10

# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test07'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test08
# # -------------------------------------------------------
# # 測試只有1000筆訓練資料的訓練速度
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本 + 3倍click_log_隨機正樣本

# neg_strategy = 'add-random-train-sample'
# N_times_neg = 3

# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test08'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 1
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test09
# # -------------------------------------------------------
# # 測試是否能在1個epoch內提升round1準確度
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本 + 'train-sm-clean_intent-based-book-neg-2'

# neg_strategy = 'train-sm-clean_intent-based-book-neg-2'
# N_times_neg = 3

# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test09'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test10
# # -------------------------------------------------------
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本 + 3倍click_log_隨機正樣本

# neg_strategy = 'add-random-train-sample'
# N_times_neg = 3

# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test10'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test11(同test10)
# # -------------------------------------------------------
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本 + 3倍click_log_隨機正樣本

# neg_strategy = 'add-random-train-sample'
# N_times_neg = 3

# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test11'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test12(同test09)
# # -------------------------------------------------------
# # 測試是否能在1個epoch內提升round1準確度
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本 + 'train-sm-clean_intent-based-book-neg-2'

# neg_strategy = 'train-sm-clean_intent-based-book-neg-2'
# N_times_neg = 3

# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test12'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  incorrect-neg-fine-tune_exp3_test13(同test12)
# # -------------------------------------------------------
# # 測試是否能在1個epoch內提升round1準確度
# # Triplet 訓練樣本 -> 驗證集中正確的樣本 + 驗證集中錯誤的樣本 + 'train-sm-clean_intent-based-book-neg-2'

# neg_strategy = 'train-sm-clean_intent-based-book-neg-2'
# N_times_neg = 3

# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'incorrect-neg-fine-tune_exp3_test13'
# # network
# network = 'triplet'
# # dataset
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = './experiments/exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# test_eval_result_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = False
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # # -------------------------------------------------------
# # #  evaluate incorrect neg fine tune 模型專用
# # # -------------------------------------------------------
# exp_name = 'incorrect-neg-fine-tune_exp3_test12'
# save_model_path = './experiments/' + exp_name + "/best_model/"

# # ======================================================================
# #   incorrect neg fine tune end
# # ======================================================================



# # ======================================================================
# #   psudo batch-all start
# # ======================================================================

# # -------------------------------------------------------
# #  exp3. exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus-redo
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp3_ECom-BERT_wo-xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus-redo'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # -------------------------------------------------------
# #  psudo-batch-all-testing03
# # -------------------------------------------------------
# drop out psudo anchor and positive
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'psudo-batch-all-testing03'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'psudo-batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name


# # ======================================================================
# #   psudo batch-all end
# # ======================================================================



# # ======================================================================
# #  new ablation study (2023/06/25)
# # ======================================================================

# # -------------------------------------------------------
# #  ablation[ABRSS-] exp11. exp11_ECom-BERT_xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus (舊的 2023/06/25之前做的)
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp11_ECom-BERT_xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-hard-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  ablation[exp1]_CKIP-BERT_xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ablation[exp1]_CKIP-BERT_xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-hard-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/ckiplab-bert-base-chinese/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # -------------------------------------------------------
# #  ablation[exp2]_ECom-BERT_xbm_soft-margin-triplet-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ablation[exp2]_ECom-BERT_xbm_soft-margin-triplet-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-soft-margin-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  ablation[exp3]_ECom-BERT_xbm_in-batch-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ablation[exp3]_ECom-BERT_xbm_in-batch-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-in-batch-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  ablation[exp4]_ECom-BERT_xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ablation[exp4]_ECom-BERT_xbm_batch-all-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-all-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3


# # -------------------------------------------------------
# #  ablation[exp5] exp10. exp10_ECom-BERT_wo-xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus (舊的 2023/06/25之前做的)
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'exp10_ECom-BERT_wo-xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'batch-hard-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name

# # -------------------------------------------------------
# #  ablation[exp6]_ECom-BERT_xbm_batch-hard-loss_train-sm_random-neg-2_valid-on-round0-plus 
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ablation[exp6]_ECom-BERT_xbm_batch-hard-loss_train-sm_random-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'naive',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-hard-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# -------------------------------------------------------
#  ablation[exp7]_ECom-BERT_xbm_batch-hard-loss_train-sm_intent-based-neg-2_valid-on-round0-plus 
# -------------------------------------------------------
# eval 的設定要另外改
# testset
# current_test_query_path = round0_test_query_path
# current_product_collection_path = round0_plus_product_collection_sm_path
# current_qrels_path = round0_plus_qrels_path
# # exp
# exp_name = 'ablation[exp7]_ECom-BERT_xbm_batch-hard-loss_train-sm_intent-based-neg-2_valid-on-round0-plus'
# # network
# network = 'triplet'
# # dataset
# # train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
# offline_mining_strategy = {
#     'mine-neg-strategy': 'train-sm-clean_intent-based-neg-2',
#     'neg-num': 2,
# }
# # loss
# loss = 'xbm-batch-hard-triplet-loss'
# # training config
# pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
# epochs = 10
# batch_size = 32
# evaluation_steps = 200
# save_model_path = './experiments/' + exp_name
# # xbm setting
# xbm_enable = True
# xbm_start_iteration = 0
# xbm_size = batch_size*3

# # ======================================================================
# #  new ablation study end (2023/06/25)
# # ======================================================================






# # -------------------------------------------------------
# #  $$$$$exp11. exp11_ECom-BERT_xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus
# # -------------------------------------------------------
# # eval 的設定要另外改
# # testset
current_test_query_path = round0_test_query_path
current_product_collection_path = round0_plus_product_collection_sm_path
current_qrels_path = round0_plus_qrels_path
# exp
exp_name = 'exp11_ECom-BERT_xbm_batch-hard-loss_train-sm_intent-based-book-neg-2_valid-on-round0-plus'
# network
network = 'triplet'
# dataset
# train_pos_path = '../PChome_datasets/search/pchome_search_click_dataset/train/positive/round1_train_sm.parquet'
offline_mining_strategy = {
    'mine-neg-strategy': 'train-sm-clean_intent-based-book-neg-2',
    'neg-num': 2,
}
# loss
loss = 'xbm-batch-hard-triplet-loss'
# training config
pretrained_model_path = '../pretrained_models/mlm_pre_train_cvc/pc+momo_title+desc/5_epoch/'
epochs = 10
batch_size = 32
evaluation_steps = 200
save_model_path = './experiments/' + exp_name
# xbm setting
xbm_enable = True
xbm_start_iteration = 0
xbm_size = batch_size*3