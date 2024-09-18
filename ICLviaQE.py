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
beam_size = 16
greedy_beam_size = 16
predictions = []
random_seed = 777
iteration = 16
max_prompts = 16 # max_prompts = iteration
early_stop_patience = 16
random.seed(random_seed)

qes = []
reals = []


nltk.download('punkt')
nltk.download('stopwords')

def compute_ngrams(sentence, n):
    words = word_tokenize(sentence)
    n_grams = set(ngrams(words, n))
    return n_grams

def find_jaccard_similarity(sentence1, sentence2, ind, n):
    ngrams1 = compute_ngrams(sentence1, n)
    ngrams2 = compute_ngrams(sentence2[ind], n)

    finder1 = BigramCollocationFinder.from_documents([ngrams1])
    finder2 = BigramCollocationFinder.from_documents([ngrams2])
    

    bigrams1 = finder1.nbest(BigramAssocMeasures.pmi, len(ngrams1))
    bigrams2 = finder2.nbest(BigramAssocMeasures.pmi, len(ngrams2))

    # Check if both denominators are non-zero before division
    if len(bigrams1) > 0 and len(bigrams2) > 0:
        common_bigrams = set(bigrams1).intersection(bigrams2)
        return len(common_bigrams) / min(len(bigrams1), len(bigrams2))
    else:
        # Handle the case where at least one denominator is zero
        return 0.0  # You can adjust this default value as needed

def preprocess_mixed_language_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english') + stopwords.words('german'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

def calculate_ttr(tokens):
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0
    return ttr

def calculate_length(tokens):
    return len(tokens) 

def calculate_mtld(tokens, threshold=30):
    mtld = 0
    chunk = []
    types = set()

    for token in tokens:
        types.add(token)
        chunk.append(token)

        if len(types) > threshold:
            mtld += 1
            chunk = []
            types = set()

    mtld += 1 if len(chunk) > 0 else 0

    return mtld

def calculate_yules_k(tokens):
    freq_dist = FreqDist(tokens)
    m1 = sum(freq_dist.values())
    
    # Check for division by zero
    if m1 == 0:
        return 0

    m2 = sum([freq ** 2 for freq in freq_dist.values()])
    k = 10000 * (m2 - m1) / (m1 ** 2)
    return k

def analyze_lines(line, ind, source):
    tokens = preprocess_mixed_language_text(line)
    ttr = calculate_ttr(tokens)
    mtld = calculate_mtld(tokens)
    yules_k = calculate_yules_k(tokens)
    length = calculate_length(tokens)
    similarity = calculate_similarity('../datasets/it_ext/Similar16_1.csv', line, ind)
    unigram = find_jaccard_similarity(line, source, ind, 1)
    bigram = find_jaccard_similarity(line, source, ind, 2)

    # Print results line by line
    print(f'Line: {line}\nTTR={ttr}, MTLD={mtld}, Yule\'s K={yules_k}, Length={length}, Similarity={similarity}, unigram={unigram}, bigram={bigram}')

    return ttr, mtld, yules_k, length, similarity, unigram, bigram

def calculate_similarity(similarity_csv_path, line, ind):
    sum_values = []
    df = pd.read_csv(similarity_csv_path, index_col=False)
    # print(line)
    sum = 0
    sentences = line.strip().split('</s>')
    #print(sentences)
    selected = []
    for sentence in sentences:
        parts = sentence.strip().split('=')
        if parts not in selected:
            if len(parts) == 2:
                source, target = parts[0].strip(), parts[1].strip()
                print(f"\nSource: {source}\nTarget: {target}\n")
                for j, column_value in enumerate(df.iloc[ind]):
                    if j not in [0, 1]:
                        column_name = df.columns[j]
                        # print(f"Row: {j}, Column: {column_name}, Value: {column_value}")
                        if column_value==source and j%2==1:
                            #print("YES "*10)
                            # print("Column: ", df.columns[j])
                            score = df.loc[ind, df.columns[j+2]]
                            #print(score)
                            match = re.search(r'\d+\.\d+', score)
                            if match:
                                score_value = float(match.group())
                                # print(f"Extracted Score: {score_value}")
                                sum+=score_value
            selected.append(parts)
    # sum_values.append(sum)
    
    return sum

def tokenize_QE_function(input, target, tokenizer_QE, device):
    return tokenizer_QE(input, target, padding='max_length', truncation='longest_first', max_length=250, return_tensors='pt').to(device)


def beam_search (device, model, tokenizer, tokenizer_QE, model_QE, src, target, cols, ind, beam_size, max_prompts, \
                early_stop_patience = early_stop_patience):
    beam = [("", 0.0, "")]  # Initial beam with empty prompt, zero BLEU score, and empty output
    prompt = ""  # Initial beam with an empty prompt
    itr = 0

    best_bleu_score = 0.0
    patience_counter = 0  # Counter to track how many iterations BLEU score hasn't improved

    selected_prompt_set = set()

    while itr < iteration and patience_counter < early_stop_patience:

        print(10*"--")
        print("itr: ", itr)

        results_list = []

        print("selected_prompt_set: ", selected_prompt_set)

        # Create a list of prompts excluding the selected ones
        available_prompts = [f"{my_dict['Sim'][cols[p]][ind]} = {my_dict['Sim'][cols[p + 1]][ind]} </s> "
                             for p in range(0, k * 2, 2) if p not in selected_prompt_set]
        print(10*"--")
        for p in range(0, k * 2, 2):
            if p not in selected_prompt_set:
                print(p)
        print(10*"--")
        
        
        
        print(available_prompts)

        print("length of available_prompts ", len(available_prompts))
        
        for new_prompt in available_prompts:
            print("new_prompt: ", new_prompt)
            print("complete_prompt: ", prompt + new_prompt + src[ind] + " = ")
            ttr, mtld, yules_k, length, similarity, unigram, bigram = analyze_lines(prompt + new_prompt, ind, src)
            results_list.append({'ttr': ttr, 'mtld': mtld, 'yules_k': yules_k, 'length': length, 'Similarity': similarity, 'unigram': unigram, 'bigram': bigram})
            print(30 * "---")

        # Convert results_list to a DataFrame
        df = pd.DataFrame(results_list)
        # Normalize the DataFrame
        normalized_df = (df - df.min()) / (df.max() - df.min())
        normalized_df.fillna(df.mean(), inplace=True)
        # Define weights
        weights = {
            'TTR': 0,
            'MTLD': 0,
            "Yule's K": 0,
            'Length': 0,
            'Similarity': 1, 
            'unigram': 0,
            'bigram': 0,
        }
        # Calculate the combined score for each prompt
        df['Combined_Score'] = (
            weights['TTR'] * normalized_df['ttr'] +
            weights['MTLD'] * normalized_df['mtld'] +
            weights["Yule's K"] * normalized_df['yules_k'] +
            weights['Length'] * normalized_df['length'] + 
            weights['Similarity'] * normalized_df['Similarity'] +
            weights['unigram'] * normalized_df['unigram'] +
            weights['bigram'] * normalized_df['bigram']
        )

        print(df)
        # Find the prompt with the highest combined score
        best_prompt_index_df = df['Combined_Score'].idxmax()

        search_string = available_prompts[best_prompt_index_df]

        my_list = [f"{my_dict['Sim'][cols[p]][ind]} = {my_dict['Sim'][cols[p + 1]][ind]} </s> "
                             for p in range(0, k * 2, 2)]
        
        best_prompt_index = next((i for i, item in enumerate(my_list) if search_string in item), None)
        if best_prompt_index is not None:
            print(f"Found at index {best_prompt_index}: {my_list[best_prompt_index]}")
        else:
            print("Not found")

        # Add the selected prompt index to the set
        selected_prompt_set.add(best_prompt_index*2)


        selected_prompt = prompt + available_prompts[best_prompt_index_df]
        print("The best index:", best_prompt_index_df)
        print("Selected prompt:", selected_prompt)
        prompt = selected_prompt

        # Provide the source separately to the model
        input_ids = tokenizer.encode(prompt + src[ind] + " = ", return_tensors='pt').to(device)
        output = model.generate(input_ids, max_new_tokens=200, do_sample=False)
        final_output = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
        print("input: ", prompt + src[ind] + " = ")
        print("final_output: ", final_output)
        print("actual label: ", target[ind])

        # target should come from QE
        # QE needs source and translation
        # Source is = > src[ind] and Translation is = > final_output
        # We need to load the QE model and QE tokenizer

        QE_input = tokenize_QE_function(src[ind], final_output, tokenizer_QE, device)
        # print(QE_input)

        # Perform regression inference
        with torch.no_grad():
            outputs = model_QE(**QE_input)

        # Get the regression output
        current_bleu_score = outputs.logits.item()
        # Your regression output is now in 'regression_output'
        print("QE BLEU:", current_bleu_score)

        qes.append(current_bleu_score)


        # Calculate BLEU score for the current output and target sentence
        real_bleu = sentence_sacrebleu(final_output, [target[ind]])
        print("real_bleu: ", real_bleu)

        reals.append(real_bleu)
        
        beam.append((prompt, current_bleu_score, final_output))

        # Stop exploration if BLEU score is 100
        if current_bleu_score >= 100.0:
            return beam
        
        # Check if BLEU score has improved
        best_bleu_score = max(best_bleu_score, beam[0][1])

        # If BLEU score hasn't improved, increment patience counter
        if best_bleu_score <= beam[0][1]:
            patience_counter += 1
        else:
            patience_counter = 0

        print(30 * "+")

        itr += 1
    
    return beam


# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-7.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-7.5B", torch_dtype=torch.float16)
model.half()
model.to(device)
model.eval()

# QE Model

tokenizer_QE = AutoTokenizer.from_pretrained("QE/IT_BLEU_3/checkpoint-30000", 
                                             do_lower_case=False, normalization=True)
model_QE = XLMRobertaForSequenceClassification.from_pretrained("QE/IT_BLEU_3/checkpoint-30000",
                                                                num_labels=1, hidden_dropout_prob=0.1).to(device)

model_QE.eval()

# Create Similar and Dissimilar index
my_dict = {"Sim": read_csv_to_dict(f"{data_dir}/{domain}_ext/Similar{k}_1.csv"),
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
    beam = beam_search(device, model, tokenizer, tokenizer_QE, model_QE, src, tgt, cols, ind, beam_size=beam_size, max_prompts=max_prompts, \
                      early_stop_patience=early_stop_patience)
    
    # Record the end time for each iteration
    iteration_end_time = time.time()

    # Calculate the elapsed time for each iteration
    iteration_elapsed_time = iteration_end_time - iteration_start_time

    # Select the top beam_size candidates
    beam = sorted(beam, key=lambda x: x[1], reverse=True)[:beam_size]

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
    output_file_path = f"decoding/{domain}{'/Optimized_Beam_'}_{early_stop_patience}_{iteration}{'-XGLM-7.5b-'}{'0+'}{iteration}.bp"
    with open(output_file_path, "a") as output_file:
        output_file.write(f"{best_prompt}\n")
    
    output_file_path = f"decoding/{domain}{'/Optimized_Beam_'}_{early_stop_patience}_{iteration}{'-XGLM-7.5b-'}{'0+'}{iteration}.out"
    with open(output_file_path, "a") as output_file:
        output_file.write(f"{best_output}\n")
    
    output_file_path = f"decoding/{domain}{'/Optimized_Beam_'}_{early_stop_patience}_{iteration}{'-XGLM-7.5b-'}{'0+'}{iteration}.time"
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
output_file_path = f"decoding/{domain}{'/Optimized_Beam_'}_{early_stop_patience}_{iteration}{'-XGLM-7.5b-'}{'0+'}{iteration}.TT"
with open(output_file_path, "a") as output_file:
    output_file.write(f"{total_elapsed_time}\n")

# tracker stops
tracker.stop()
