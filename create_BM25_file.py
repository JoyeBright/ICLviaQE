from tqdm import tqdm
import random
from transformers import XGLMTokenizer, XGLMForCausalLM
import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
from utils import *
from sacremoses import MosesTokenizer, MosesDetokenizer
from rank_bm25 import BM25Okapi
import time
from carbontracker.tracker import CarbonTracker

src_lang = 'de'
tgt_lang = 'en'
domain = 'it'
split = 'test'
data_dir = '../datasets'
top_k = 100
k = 16
prompts = {}
predictions = []


# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-7.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-7.5B", torch_dtype=torch.float16)
model.half()
model.to(device)
model.eval()

# Create index
my_dict = {src_lang: read_file(f"{data_dir}/{domain}/train.{src_lang}"),
           tgt_lang: read_file(f"{data_dir}/{domain}/train.{tgt_lang}")}

# Carbon tracker
tracker = CarbonTracker(epochs=len(my_dict[src_lang]), monitor_epochs=-1, epochs_before_pred = -1, verbose=2, components="gpu", log_dir=f'decoding/{domain},')

# Record the start time
start_time = time.time()

# tracker starts
tracker.epoch_start()

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
for ind in tqdm(range(len(src))):
    tokenized_query = mt_tok.tokenize(src[ind]) 
    doc_scores = BM25_de.get_scores(tokenized_query)
    top_n_indices = np.argsort(doc_scores)[::-1][:top_k]
    similar_outs[ind] = top_n_indices
    # print(my_dict['de'][similar_outs[ind][0]])
    if k==1:
        prompts[ind] = my_dict[src_lang][similar_outs[ind][0]] + " = " + my_dict[tgt_lang][similar_outs[ind][0]] + " </s> " + src[ind] + " = "
    if k>1:
        prompts[ind] = (" </s> ").join([f"{my_dict[src_lang][item]} = {my_dict[tgt_lang][item]}" for item in similar_outs[ind][:k]])
        prompts[ind] = prompts[ind] + " </s> domain: IT </s> " + src[ind] + " = "
    with open(f"{domain}{'/BM25-Domain-XGLM-7.5b-'}{'0+'}{k}.ICE", "a") as f:
        f.write(prompts[ind] + "\n")
    input_ids = tokenizer.encode(prompts[ind], return_tensors='pt').to(device)
    output = model.generate(input_ids, max_new_tokens=200, do_sample=False)
    final_output = tokenizer.decode(output[0, input_ids.shape[1]: ], skip_special_tokens=True)
    predictions.append(final_output)
    with open(f"{domain}{'/BM25-Domain-XGLM-7.5b-'}{'0+'}{k}", "a") as f:
         f.write(final_output + "\n")
    print("----")  
    print(prompts[ind])
    print("iterator: ", ind)
    print("model ouput: ", final_output)
    print("actual label: ", tgt[ind])
    outputs = get_outputs(predictions, truncate=True, max_length=lengths[:ind+1])
    print("bleu score so far: ", score(outputs, tgt[:ind+1]))


with open(f"{domain}{'/BM25-Domain-XGLM-7.5b-'}{'0+'}{k}.new.{'bleu'}", "w") as f:
    f.write(str(score(outputs, tgt)))

# tracker ends
tracker.epoch_end()

# Record the end time
end_time = time.time()

# Calculate the total elapsed time
total_elapsed_time = end_time - start_time

# Save the total elapsed time
print(f"Total Elapsed Time: {total_elapsed_time} seconds")
output_file_path = f"{domain}{'/BM25-Domain-XGLM-7.5b-'}{'0+'}{k}.new.TT"
with open(output_file_path, "a") as output_file:
    output_file.write(f"{total_elapsed_time}\n")

# tracker stops
tracker.stop()
