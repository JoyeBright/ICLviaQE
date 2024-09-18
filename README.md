# ICLviaQE
This repository contains the data, code, and models required to replicate the experiments from our paper: **"Guiding In-Context Learning of LLMs through Quality Estimation for Machine Translation."**

## Overview

Our method guides large language models (LLMs) in In-Context Learning (ICL) through Quality Estimation (QE), improving machine translation accuracy by selecting higher-quality translations from context. Below is a visual summary of our methodology.

<img src="/Overview.png" width="700px" alt="Methodology Overview" title="Our Methodology">

## Methodology Breakdown

1. **Relevant Translation Pair Selection**: For each source text, relevant translation pairs are selected from the training set using BM25.  
   Code: [`BM25.py`](BM25.py)
   
2. **Selection, Translation, and Quality Estimation**:  
   This step involves performing the selection, translation, and quality estimation of the target outputs.  
   Code: [`ICLviaQE.py`](ICLviaQE.py)

To run Stage 2, you'll need a QE model, which we've shared on the Hugging Face Hub. Access it here:  
👉 [ICLviaQE Model on Hugging Face](https://huggingface.co/joyebright/ICLviaQE/tree/main)

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

## Authors

- **Javad Pourmostafa**  - [Email](mailto:j.pourmostafa@tilburguniversity.edu), [Website](https://javad.pourmostafa.me)
- **Dimitar Shterionov** - [Email](mailto:d.shterionov@tilburguniversity.edu), [Website](https://ilk.uvt.nl/~shterion/)
- **Pieter Spronck**     - [Email](mailto:p.spronck@tilburguniversity.edu), [Website](https://www.spronck.net/)

## Citation

Please cite the preprint version of our paper until it is officially published:

```bibtex
@misc{sharami2024guidingincontextlearningllms,
      title={Guiding In-Context Learning of LLMs through Quality Estimation for Machine Translation}, 
      author={Javad Pourmostafa Roshan Sharami and Dimitar Shterionov and Pieter Spronck},
      year={2024},
      eprint={2406.07970},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.07970}, 
}
