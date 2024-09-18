import os
import re
import time
import random
import torch
import pandas as pd
import nltk
from tqdm import tqdm
from transformers import XGLMTokenizer, XGLMForCausalLM, AutoTokenizer, XLMRobertaForSequenceClassification
from sacrebleu import sentence_bleu
from carbontracker.tracker import CarbonTracker
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams, FreqDist
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder

# Configuration Variables
src_lang = 'de'
tgt_lang = 'en'
domain = 'it'
split = 'test'
data_dir = '../datasets'
k = 16
beam_size = 16
max_prompts = 16
random_seed = 777
early_stop_patience = 16
iteration = 16

# Set the random seed for reproducibility
random.seed(random_seed)

# Initialize required lists
qes = []
reals = []
predictions = []

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Utility functions
def compute_ngrams(sentence, n):
    words = word_tokenize(sentence)
    return set(ngrams(words, n))

def find_jaccard_similarity(sentence1, sentence2, n):
    ngrams1 = compute_ngrams(sentence1, n)
    ngrams2 = compute_ngrams(sentence2, n)

    finder1 = BigramCollocationFinder.from_documents([ngrams1])
    finder2 = BigramCollocationFinder.from_documents([ngrams2])

    bigrams1 = finder1.nbest(BigramAssocMeasures.pmi, len(ngrams1))
    bigrams2 = finder2.nbest(BigramAssocMeasures.pmi, len(ngrams2))

    if bigrams1 and bigrams2:
        common_bigrams = set(bigrams1).intersection(bigrams2)
        return len(common_bigrams) / min(len(bigrams1), len(bigrams2))
    return 0.0

def preprocess_mixed_language_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + stopwords.words('german'))
    return [token for token in tokens if token.isalnum() and token not in stop_words]

def calculate_ttr(tokens):
    return len(set(tokens)) / len(tokens) if tokens else 0

def calculate_length(tokens):
    return len(tokens)

def calculate_mtld(tokens, threshold=30):
    mtld, chunk, types = 0, [], set()
    for token in tokens:
        types.add(token)
        chunk.append(token)
        if len(types) > threshold:
            mtld += 1
            chunk, types = [], set()
    return mtld + (1 if chunk else 0)

def calculate_yules_k(tokens):
    freq_dist = FreqDist(tokens)
    m1 = sum(freq_dist.values())
    if m1 == 0: return 0
    m2 = sum([freq ** 2 for freq in freq_dist.values()])
    return 10000 * (m2 - m1) / (m1 ** 2)

# Search Functions
def analyze_line(line, src, ind):
    tokens = preprocess_mixed_language_text(line)
    return {
        'ttr': calculate_ttr(tokens),
        'mtld': calculate_mtld(tokens),
        'yules_k': calculate_yules_k(tokens),
        'length': calculate_length(tokens),
        'unigram': find_jaccard_similarity(line, src[ind], 1),
        'bigram': find_jaccard_similarity(line, src[ind], 2)
    }

def tokenize_QE_function(input, target, tokenizer_QE, device):
    return tokenizer_QE(input, target, padding='max_length', truncation='longest_first', max_length=250, return_tensors='pt').to(device)

def search(device, model, tokenizer, tokenizer_QE, model_QE, src, target, cols, ind):
    beam, prompt, itr, patience_counter = [("", 0.0, "")], "", 0, 0
    best_bleu_score, selected_prompt_set = 0.0, set()

    while itr < iteration and patience_counter < early_stop_patience:
        available_prompts = [f"{my_dict['Sim'][cols[p]][ind]} = {my_dict['Sim'][cols[p + 1]][ind]} </s> "
                             for p in range(0, k * 2, 2) if p not in selected_prompt_set]
        results_list = [analyze_line(prompt + new_prompt, src, ind) for new_prompt in available_prompts]

        df = pd.DataFrame(results_list)
        normalized_df = (df - df.min()) / (df.max() - df.min())
        normalized_df.fillna(df.mean(), inplace=True)

        weights = {'TTR': 0, 'MTLD': 0, "Yule's K": 0, 'Length': 0, 'Similarity': 1, 'unigram': 1, 'bigram': 0}
        df['Combined_Score'] = sum(weights[col] * normalized_df[col] for col in weights)

        best_prompt_index = df['Combined_Score'].idxmax()
        selected_prompt_set.add(best_prompt_index * 2)
        selected_prompt = prompt + available_prompts[best_prompt_index]
        prompt = selected_prompt

        input_ids = tokenizer.encode(prompt + src[ind] + " = ", return_tensors='pt').to(device)
        output = model.generate(input_ids, max_new_tokens=200, do_sample=False)
        final_output = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)

        QE_input = tokenize_QE_function(src[ind], final_output, tokenizer_QE, device)
        with torch.no_grad():
            current_bleu_score = model_QE(**QE_input).logits.item()

        real_bleu = sentence_bleu(final_output, [target[ind]]).score
        qes.append(current_bleu_score)
        reals.append(real_bleu)

        beam.append((prompt, current_bleu_score, final_output))
        if current_bleu_score >= 100.0: return beam

        best_bleu_score = max(best_bleu_score, beam[0][1])
        patience_counter += 1 if best_bleu_score <= beam[0][1] else 0
        itr += 1

    return beam

# Initialize Models and Tokenizers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-7.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-7.5B", torch_dtype=torch.float16).half().to(device).eval()

tokenizer_QE = AutoTokenizer.from_pretrained("QE/IT_BLEU_3/checkpoint-30000", do_lower_case=False, normalization=True)
model_QE = XLMRobertaForSequenceClassification.from_pretrained("QE/IT_BLEU_3/checkpoint-30000", num_labels=1, hidden_dropout_prob=0.1).to(device).eval()

# Load Data
my_dict = {"Sim": read_csv_to_dict(f"{data_dir}/{domain}_ext/Similar{k}_1.csv"),
           "Dis": read_csv_to_dict(f"{data_dir}/{domain}_ext/Dissimilar8_1.csv")}

src = read_file(f"{data_dir}/{domain}/{split}.{src_lang}")
tgt = read_file(f"{data_dir}/{domain}/{split}.{tgt_lang}")
cols = [f'top{j+1}' if j % 2 == 0 else f'top{j+1}_trg' for j in range(k * 2)]
lengths = [len(x) * 2 for x in src]

# Carbon Tracker
tracker = CarbonTracker(epochs=len(src), monitor_epochs=-1, epochs_before_pred=-1, verbose=2, components="gpu", log_dir=f'decoding/{domain}')

# Main Loop
start_time = time.time()

for ind in range(len(src)):
    tracker.epoch_start()

    iteration_start_time = time.time()
    beam = search(device, model, tokenizer, tokenizer_QE, model_QE, src, tgt, cols, ind)
    iteration_elapsed_time = time.time() - iteration_start_time

    best_prompt, best_score, best_output = max(beam, key=lambda x: x[1])
    predictions.append(best_output)
    outputs = get_outputs(predictions[:ind + 1], truncate=True, max_length=lengths[:ind + 1])
    bleu_sofar = score_sacrebleu(outputs, tgt[:ind + 1]).score

    # Save results
    save_results(domain, early_stop_patience, iteration, best_prompt, best_output, iteration_elapsed_time, bleu_sofar)

    tracker.epoch_end()

# Save total elapsed time
total_elapsed_time = time.time() - start_time
save_total_time(domain, early_stop_patience, iteration, total_elapsed_time)

tracker.stop()
