from tqdm import tqdm
import random
import torch
import numpy as np
from transformers import pipeline
from utils import *
from sacremoses import MosesTokenizer, MosesDetokenizer
from rank_bm25 import BM25Okapi
import time
from carbontracker.tracker import CarbonTracker
import csv

src_lang = 'de'
tgt_lang = 'en'
domain = 'it'
split = 'test'
data_dir = '../datasets'
top_k = 100
k = 100
prompts = {}
predictions = []

# Create index
my_dict = {src_lang: read_file(f"{data_dir}/{domain}/train.{src_lang}"),
           tgt_lang: read_file(f"{data_dir}/{domain}/train.{tgt_lang}")}

for lang in [src_lang, tgt_lang]:
    mt_tok = MosesTokenizer(lang=lang) 
    corpus = my_dict[lang]
    tokenized_corpus = []
    for i in range(len(corpus)):
        tokenized_corpus.append(mt_tok.tokenize(corpus[i]))
    # create BM25 for source and target corpora
    if lang == src_lang:
        globals()['BM25_' + src_lang] = BM25Okapi(tokenized_corpus)
    if lang == tgt_lang:
        globals()['BM25_' + tgt_lang] = BM25Okapi(tokenized_corpus)

# Search using a key (test samples) among the source sentences of the train set
src = read_file(f"{data_dir}/{domain}/{split}.{src_lang}")
tgt = read_file(f"{data_dir}/{domain}/{split}.{tgt_lang}")

lengths = [len(x)*2 for x in src]  

mt_tok = MosesTokenizer(lang=src_lang)
similar_outs = {}
top_examples = []

for ind in tqdm(range(len(src))):
    tokenized_query = mt_tok.tokenize(src[ind]) 
    doc_scores = BM25_de.get_scores(tokenized_query)
    top_n_indices = np.argsort(doc_scores)[::-1][:top_k]
    similar_outs[ind] = top_n_indices
    
    # Create a list to store top examples and their scores for this instance
    instance_examples = [ind]  # Add index as the first element
    instance_examples.append(src[ind])
    for i in range(k):
        instance_examples.extend([
            my_dict[src_lang][similar_outs[ind][i]],
            my_dict[tgt_lang][similar_outs[ind][i]],
            doc_scores[top_n_indices[i]]
        ])
    
    # Append the top examples and scores for this instance to the list
    top_examples.append(instance_examples)

# Save the top examples and their scores to a CSV file
csv_file = f"../datasets/it_ext/Similar{k}.BM25.csv"
csv_columns = ["", "Query"]
for i in range(1, k + 1):
    csv_columns.extend([f"top{i}", f"top{i}_trg{i}", f"top{i}_score{i}"])

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(csv_columns)
    writer.writerows(top_examples)
