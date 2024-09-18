from tqdm import tqdm
import random
from transformers import XGLMTokenizer, XGLMForCausalLM, AutoTokenizer, XLMRobertaForSequenceClassification
import torch
from sacrebleu import sentence_bleu
from utils import *
import time
from carbontracker.tracker import CarbonTracker
import nltk
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import re
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder

src_lang = 'de'
tgt_lang = 'en'
domain = 'it'
split = 'test'
data_dir = '../datasets'
k = 16
prompts = {}
predictions = []
random_seed = 777
iteration = 16
max_prompts = 16 # max_prompts = iteration
early_stop_patience = 3
random.seed(random_seed)

qes = []
reals = []


nltk.download('punkt')
nltk.download('stopwords')


def tokenize_QE_function(input, target, tokenizer_QE, device):
    return tokenizer_QE(input, target, padding='max_length', truncation='longest_first', max_length=250, return_tensors='pt').to(device)


def beam_search(device, model, tokenizer, tokenizer_QE, model_QE, src, target, cols, ind, max_prompts, early_stop_patience=early_stop_patience):
    beam = [("", 0.0, "")]
    prompt = ""
    itr = 0
    best_bleu_score = 0.0
    patience_counter = 0
    # selected_prompt_indices = set()

    while itr < iteration and patience_counter < early_stop_patience:

        available_prompts = [f"{my_dict['Sim'][cols[p]][ind]} = {my_dict['Sim'][cols[p + 1]][ind]} </s> " for p in range(0, k * 2, 2)]

        if available_prompts:
            selected_prompt_index = itr % k  # Change index after every iteration
            # selected_prompt_index = random.randint(0, k-1)
            # print(selected_prompt_index)

            selected_prompt = available_prompts[selected_prompt_index]

            # Construct the full prompt
            selected_prompt = prompt + selected_prompt
            prompt = selected_prompt

            input_ids = tokenizer.encode(prompt + src[ind] + " = ", return_tensors='pt').to(device)

            print(prompt + src[ind] + " = ")

            max_length = model.config.max_position_embeddings
            if len(input_ids[0]) > max_length:
                return beam
                
            output = model.generate(input_ids, max_new_tokens=200, do_sample=False)
            final_output = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
            print(final_output)
            print(target[ind])

            QE_input = tokenize_QE_function(src[ind], final_output, tokenizer_QE, device)
            with torch.no_grad():
                outputs = model_QE(**QE_input)

            current_bleu_score = outputs.logits.item()
            print("QE:", current_bleu_score)

            real_bleu = sentence_sacrebleu(final_output, [target[ind]])
            print("Real BLEU:", real_bleu)
            
            beam.append((prompt, current_bleu_score, final_output))

            if current_bleu_score >= 100:
                return beam
            
            beam = sorted(beam, key=lambda x: x[1], reverse=True)[:early_stop_patience]

            if current_bleu_score <= best_bleu_score:
                patience_counter += 1
            else:
                patience_counter = 0

            best_bleu_score = beam[0][1]

        itr += 1

        print(50 * "+")
    
    return beam




# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-7.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-7.5B", torch_dtype=torch.float16)
model.half()
model.to(device)
model.eval()

# QE Model
#QE/IT_BLEU_3/checkpoint-30000

tokenizer_QE = AutoTokenizer.from_pretrained("QE/IT_BLEU_3/checkpoint-30000", 
                                             do_lower_case=False, normalization=True)
model_QE = XLMRobertaForSequenceClassification.from_pretrained("QE/IT_BLEU_3/checkpoint-30000",
                                                                num_labels=1, hidden_dropout_prob=0.1).to(device)

model_QE.eval()

# Create Similar and Dissimilar index
my_dict = {"Sim": read_csv_to_dict(f"{data_dir}/{domain}_ext/Similar{k}_BM25.csv"),
           "Dis": read_csv_to_dict(f"{data_dir}/{domain}_ext/Dissimilar8_1.csv")}

# Search using a key (test samples) among the source sentences of the train set
src = read_file(f"{data_dir}/{domain}/{split}.{src_lang}")
tgt = read_file(f"{data_dir}/{domain}/{split}.{tgt_lang}")
lengths = [len(x) * 2 for x in src]

# Assume you have cols defined elsewhere in your code
cols = []
for j in range(0, k):
    cols.append('top' + str(j + 1))
    cols.append('top' + str(j + 1) + '_trg' + str(j + 1))


# Carbon tracker
tracker = CarbonTracker(epochs=len(src), monitor_epochs=-1, epochs_before_pred = -1, verbose=2, components="gpu", log_dir=f'decoding/{domain},')

# Record the start time
start_time = time.time()

for ind in range(0, len(src)):
     #tracker starts
    tracker.epoch_start()

    # Record the start time for each iteration
    iteration_start_time = time.time()

    # Beam Search
    beam = beam_search(device, model, tokenizer, tokenizer_QE, model_QE, src, tgt, cols, ind, max_prompts=max_prompts, \
                      early_stop_patience=early_stop_patience)
    
    # Record the end time for each iteration
    iteration_end_time = time.time()

    # Calculate the elapsed time for each iteration
    iteration_elapsed_time = iteration_end_time - iteration_start_time

    # Select the top beam_size candidates
    beam = sorted(beam, key=lambda x: x[1], reverse=True)[:early_stop_patience]

    print(beam)

    # Select the best prompt from the beam
    best_prompt, best_score, best_output = max(beam, key=lambda x: x[1])

    print('---****FOUND****---')
    print("Best Prompt:", best_prompt)
    print("Best Blue Score:", best_score)
    print("Best output:", best_output)
    predictions.append(best_output)
    outputs = get_outputs(predictions[:ind+1], truncate=True, max_length=lengths[:ind+1])
    bleu_sofar, X  = score_sacrebleu(outputs, tgt[:ind+1])
    print("BLEU So far: ", bleu_sofar.score)
    print("Actual Target:", tgt[ind])
    print(f"Iteration Elapsed Time: {iteration_elapsed_time} seconds")

    # Save only the best prompt to the file (append mode)
    output_file_path = f"decoding/{domain}/test.bp"
    with open(output_file_path, "a") as output_file:
        output_file.write(f"{best_prompt}\n")
    
    output_file_path = f"decoding/{domain}/test.out"
    with open(output_file_path, "a") as output_file:
        output_file.write(f"{best_output}\n")
    
    output_file_path = f"decoding/{domain}/test.time"
    with open(output_file_path, "a") as output_file:
        output_file.write(f"{iteration_elapsed_time}\n")

    # tracker ends
    tracker.epoch_end()

# Record the end time
end_time = time.time()

# Calculate the total elapsed time
total_elapsed_time = end_time - start_time

# Save the total elapsed time
print(f"Total Elapsed Time: {total_elapsed_time} seconds")
output_file_path = f"decoding/{domain}/test.TT"
with open(output_file_path, "a") as output_file:
    output_file.write(f"{total_elapsed_time}\n")

# tracker stops
tracker.stop()


    
    
