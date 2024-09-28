# ICLviaQE

This repository contains the data, code, and models required to replicate the experiments from our paper: **"Guiding In-Context Learning of LLMs through Quality Estimation for Machine Translation."** The paper has been accepted to AMTA 2024. https://aclanthology.org/2024.amta-research.9/

## Overview

Below is a visual summary of our methodology.

<img src="/Overview.png" width="700px" alt="Methodology Overview" title="Our Methodology">

📽️ [Watch the YouTube presentation of our paper](https://www.youtube.com/watch?v=CkVs-XV0LW0&ab_channel=JavadPourmostafa)

## Methodology Breakdown

1. **Relevant Translation Pair Selection**: For each source text, relevant translation pairs are selected from the training set using BM25.  
   Code: [`BM25.py`](BM25.py)
   
2. **Selection, Translation, and Estimation**:  
   This step involves performing the selection, translation, and quality estimation of the target outputs.  
   Code: [`ICLviaQE.py`](ICLviaQE.py)

To run Stage 2, you'll need a QE model, which we've shared on the Hugging Face Hub. Access it here:  
👉 [ICLviaQE Model on Hugging Face](https://huggingface.co/joyebright/ICLviaQE/tree/main)

To prioritize translation pairs with unigram overlaps with the source, set the unigram weight to 1 (default is 0) in `ICLviaQE.py`. Note that other weights from our analysis not discussed in the paper should be ignored.

## Baselines

- **Random**: Generates translations by randomly selecting examples.  
  Code: [`random_file.py`](random_file.py) -> [`run_generation.py`](run_generation.py)
  
- **Task-level**: Translations are generated based on task-level context.  
  Code: [`create_task_file.py`](create_task_file.py) -> [`run_generation.py`](run_generation.py)

- **BM25**: Uses BM25 to find similar source-target pairs.  
  Code: [`create_BM25_file.py`](create_BM25_file.py)

- **R-BM25**: An enhanced version of BM25. Implementation is available here:  
  [R-BM25 Repository](https://github.com/sweta20/inContextMT)

- **mBART-50**: 
  [mBART-50 Repository](https://github.com/JoyeBright/MT-HF)

## Running the Code

To run any of the stages or baselines, follow the instructions provided in the relevant code files. 

## Summary
The quality of output from large language models (LLMs), particularly in machine translation (MT), is closely tied to the quality of in-context examples (ICEs) provided along with the query, i.e., the text to translate. The effectiveness of these ICEs is influenced by various factors, such as the domain of the source text, the order in which the ICEs are presented, the number of these examples, and the prompt templates used. Naturally, selecting the most impactful ICEs depends on understanding how these affect the resulting translation quality, which ultimately relies on translation references or human judgment. Our work presents a novel methodology for in-context learning (ICL) that relies on a search algorithm guided by domain-specific quality estimation (QE). Leveraging the XGLM model, our methodology estimates the resulting translation quality without the need for translation references, selecting effective ICEs for MT to maximize translation quality. Our results demonstrate significant improvements over existing ICL methods and higher translation performance compared to fine-tuning a pre-trained language model (PLM), specifically mBART-50.

## Authors

- **Javad Pourmostafa** - [Email](mailto:j.pourmostafa@tilburguniversity.edu), [Website](https://javad.pourmostafa.me)
- **Dimitar Shterionov** - [Email](mailto:d.shterionov@tilburguniversity.edu), [Website](https://ilk.uvt.nl/~shterion/)
- **Pieter Spronck** - [Email](mailto:p.spronck@tilburguniversity.edu), [Website](https://www.spronck.net/)

## Citation

```bibtex
@inproceedings{pourmostafa-roshan-sharami-etal-2024-guiding,
    title = "Guiding In-Context Learning of {LLM}s through Quality Estimation for Machine Translation",
    author = "Pourmostafa Roshan Sharami, Javad  and
      Shterionov, Dimitar  and
      Spronck, Pieter",
    editor = "Knowles, Rebecca  and
      Eriguchi, Akiko  and
      Goel, Shivali",
    booktitle = "Proceedings of the 16th Conference of the Association for Machine Translation in the Americas (Volume 1: Research Track)",
    month = sep,
    year = "2024",
    address = "Chicago, USA",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2024.amta-research.9",
    pages = "88--101",
    abstract = "The quality of output from large language models (LLMs), particularly in machine translation (MT), is closely tied to the quality of in-context examples (ICEs) provided along with the query, i.e., the text to translate. The effectiveness of these ICEs is influenced by various factors, such as the domain of the source text, the order in which the ICEs are presented, the number of these examples, and the prompt templates used. Naturally, selecting the most impactful ICEs depends on understanding how these affect the resulting translation quality, which ultimately relies on translation references or human judgment. This paper presents a novel methodology for in-context learning (ICL) that relies on a search algorithm guided by domain-specific quality estimation (QE). Leveraging the XGLM model, our methodology estimates the resulting translation quality without the need for translation references, selecting effective ICEs for MT to maximize translation quality. Our results demonstrate significant improvements over existing ICL methods and higher translation performance compared to fine-tuning a pre-trained language model (PLM), specifically mBART-50.",
}
