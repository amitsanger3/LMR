import torch
import numpy as np
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size=400002  # # size of total vocabulary only for glove
embed_size=769  # # tensor len
hidden_size = 512
rnn_layers = 1
num_layers=1
num_output=64
batch_size=16


lr = 1e-3   # [1e-3, 1e-5, 1e-7]

weight_decay = 0.01
warmup_steps=10000

betas=[0.8, 0.9]
clip=1.0
verbose=True
sup_labels=None
t_total=-1
warmup=0.1
weight_decay=0.01
validate_every=1
schedule="warmup_linear"
e=1e-6

logs_txt = "GateIdLogs/RNN_CONLL_N2.txt"

base_model = """ Due to IPR we are not disclosing this part of code."""
bert_base_model = "bert-base-multilingual-cased"
pos_tagger = "pos_tagger/pytorch_model.bin"
text_len = 64
tag_name = 'O'

best_loss = np.inf
best_accuracy = 0.0
start_epoch = 0
num_epochs = 3
title = "NLP model without RNN with MultiAttention"

model_path = "TRAINED_MODEL/Conell/"
report_path = "REPORT/Conell/"
logs_path = "LOGS/Conell/"

pos_dict_path = "data/pos/pos.json"
geo_path = "/geoai/input.jsonl"
faulty_op_path = "faulty/lmr_fault.txt"


test_data_stage = 2
distance = False
