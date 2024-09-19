from tqdm import tqdm
import random
from transformers import XGLMTokenizer, XGLMForCausalLM, T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
from utils import *
import time
from carbontracker.tracker import CarbonTracker

domain = 'it'
model_name = 'XGLM'


eval_src, eval_tgt = get_data(domain, 'de', 'en', "dev", '../datasets')
train_src, train_tgt = get_data(domain, 'de', 'en', "train", '../datasets')


num_itrs = 3
k = 16
prompts = {}
lengths = [len(x)*2 for x in eval_src]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_name == 'flan-t5':
     tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
     model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
elif model_name == 'XGLM':
     tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-7.5B")
     model = XGLMForCausalLM.from_pretrained("facebook/xglm-7.5B", torch_dtype=torch.float16)
else:
     assert "No model was selected!"

model.half()
model.to(device)
model.eval()

predictions = []
scores = {}
scores.setdefault("itr", [])
scores.setdefault("id", [])
scores.setdefault("score", [])

# Carbon tracker
tracker = CarbonTracker(epochs=len(train_src), monitor_epochs=-1, epochs_before_pred = -1, verbose=2, components="gpu", log_dir=f'decoding/{domain},')

# Record the start time
start_time = time.time()

for it in range(num_itrs):

    #tracker starts
    tracker.epoch_start()

    randomSeed = it
    random.seed(randomSeed)
    RandomNumber = random.sample(range(len(train_src)), k)
    print("Random number(s) for this iteration: ", RandomNumber)
    for i, sample in tqdm(enumerate(eval_src)):
        if len(RandomNumber)==1:
             if model_name=='XGLM':
                prompts[it, i] = train_src[RandomNumber[0]] + " = " + train_tgt[RandomNumber[0]] + " </s> " + sample + " = "
             if model_name=="flan-t5":
                prompts[it, i] = "German: " + train_src[RandomNumber[0]] + "\nEnglish: " + train_tgt[RandomNumber[0]] + "\nTranslate German to English: " + sample
        if len(RandomNumber)>1:
             if model_name=='XGLM':
                prompts[it, i] = (" </s> ").join([f"{train_src[item]} = {train_tgt[item]}" for item in RandomNumber])
                prompts[it, i] = prompts[it, i] + " </s> " + sample + " = "
             if model_name=='flan-t5':
                prompts[it, i] = (" </s> ").join([f"{train_src[item]}: {train_tgt[item]}" for item in RandomNumber])
                prompts[it, i] = prompts[it, i] + " </s> Translate German to English: " + sample
        input_ids = tokenizer.encode(prompts[it, i], return_tensors='pt').to(device)
        output = model.generate(input_ids, max_new_tokens=200, do_sample=False)
        if model_name=='XGLM':
            final_output = tokenizer.decode(output[0, input_ids.shape[1]: ], skip_special_tokens=True)
        if model_name=='flan-t5':
            final_output = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(final_output)
        with open(f"{domain}{'/random-'}{model_name}{'-7.5b-'}{k}{'+0'}.{it}", "a") as f:
            f.write(final_output + "\n")
        print("-------")
        print("trial: ", it, ", iterator in loop: ", i)
        print("prompt: ", prompts[it, i])
        print("model ouput: ", final_output)
        print("actual label: ", eval_tgt[i])
        outputs = get_outputs(predictions, truncate=True, max_length=lengths[:i+1])
        print("bleu score so far: ", score(outputs, eval_tgt[:i+1]))
        # print("sacrebleu score so far: ", score_sacrebleu(outputs, eval_tgt[:j+1]))
    with open(f"{domain}{'/random-'}{model_name}{'-7.5b-'}{k}{'+0'}.{it}.{'bleu'}", "w") as f:
            f.write(str(score(outputs, eval_tgt)))
    scores["itr"].append(it) 
    scores["score"].append(score(outputs, eval_tgt)['score'])
    print(score(outputs, eval_tgt))
    predictions = []

    # tracker ends
    tracker.epoch_end()

# Record the end time
end_time = time.time()

# Calculate the total elapsed time
total_elapsed_time = end_time - start_time

# Save the total elapsed time
print(f"Total Elapsed Time: {total_elapsed_time} seconds")
output_file_path = f"{domain}{'/random-'}{model_name}{'-7.5b-'}{k}{'+0'}.{it}.TT"
with open(output_file_path, "a") as output_file:
    output_file.write(f"{total_elapsed_time}\n")

# tracker stops
tracker.stop()


