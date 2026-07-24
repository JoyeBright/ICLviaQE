import evaluate
import sacrebleu
from sacrebleu import sentence_bleu
from tqdm import tqdm
import csv

def score(predictions, references):
    assert len(predictions) == len(references)
    metric = evaluate.load("sacrebleu")
    scores = metric.compute(predictions=predictions, references=references)
    return scores


def get_data(domain, src_lang, tgt_lang, split, DOMAIN_DATA_DIR):
    src = read_file(f"{DOMAIN_DATA_DIR}/{domain}/{split}.{src_lang}")
    tgt = read_file(f"{DOMAIN_DATA_DIR}/{domain}/{split}.{tgt_lang}")
    return src, tgt

def sentence_sacrebleu(prediction, reference):
    current_bleu_score = sentence_bleu(prediction, reference).score
    return current_bleu_score

def score_sacrebleu(predictions, references):

    assert len(predictions) == len(references)

    return sacrebleu.corpus_bleu(predictions, [references]), [sacrebleu.sentence_bleu(x,[y]).score for x, y in zip(predictions, references)]

def read_file(fname, transform=lambda x: x):
    data = []
    num_lines = sum(1 for line in open(fname,'r'))
    with open(fname) as f:
        for line in tqdm(f, total=num_lines):
            data.append(transform(line.strip()))
    return data

def read_csv_to_dict(fname):
    data_dict = {}
    with open(fname, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            for key, value in row.items():
                if key not in data_dict:
                    data_dict[key] = []
                data_dict[key].append(value)
    # Access the data by column name e.g., data_dict['top1']
    return data_dict

def get_outputs(predictions, lower=False, truncate=False, max_length=None):
    outputs = []
    for i, cand in enumerate(predictions):
        if lower:
            cand = cand.lower()
        if truncate:
            if isinstance(max_length, list):
                cand = cand[:max_length[i]]
            else:
                cand = cand[:max_length]
        outputs.append(cand)
    return outputs
