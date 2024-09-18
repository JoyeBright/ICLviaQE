from tqdm import tqdm
import random
from transformers import XGLMTokenizer, XGLMForCausalLM
import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import evaluate
import json
from utils import *


# Set the domain
domain = 'it'


train_src, train_tgt = get_data(domain, 'de', 'en', "train", '../datasets')
eval_src, eval_tgt = get_data(domain, 'de', 'en', "test", '../datasets')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = XGLMTokenizer.from_pretrained("facebook/xglm-7.5B")
model = XGLMForCausalLM.from_pretrained("facebook/xglm-7.5B", torch_dtype=torch.float16)
model.half()
model.to(device)
model.eval()

# Set the best number obtained (from either random or task-level)
# Don't forget to change the output name accordingly
BestNumber = 369713
k = 1
print("best number: ", BestNumber)

predictions = []
prompts = {}

lengths = [len(x)*2 for x in eval_src]

for j, sample in tqdm(enumerate(eval_src)):
    if k==1:
        prompts[j, sample] = train_src[BestNumber] + " = " + train_tgt[BestNumber] + " </s> " + sample + " = "
    if k>1:
        randomSeed = BestNumber
        random.seed(randomSeed)
        RandomNumber = random.sample(range(len(train_src)), k)  
        prompts[j, sample] = (" </s> ").join([f"{train_src[item]} = {train_tgt[item]}" for item in RandomNumber])
        prompts[j, sample] = prompts[j, sample] + " </s> " + sample + " = "
    input_ids = tokenizer.encode(prompts[j, sample], return_tensors='pt').to(device)
    output = model.generate(input_ids, max_new_tokens=200, do_sample=False)
    final_output = tokenizer.decode(output[0, input_ids.shape[1]: ], skip_special_tokens=True)
    predictions.append(final_output)
    print("-------")
    print("iterator: ", j)
    print("prompt: ", prompts[j, sample])
    print("model ouput: ", final_output)
    print("actual label: ", eval_tgt[j])
    outputs = get_outputs(predictions, truncate=True, max_length=lengths[:j+1])
    print("bleu score so far: ", score(outputs, eval_tgt[:j+1]))
    with open(f"{domain}{'.task-level-XGLM-7.5-'}{k}{'+0'}", "a") as f:
            f.write(final_output + "\n")

print(score(outputs, eval_tgt))
