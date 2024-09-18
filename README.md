# ICLviaQE
This repository contains the data, code, and model required to replicate the results of the experiments in our paper, "Guiding In-Context Learning of LLMs through Quality Estimation for Machine Translation."

# Our methodology
<img src="/Overview.png" width="700px" title="hover text">

1. Finding relevant translation pairs from the training set per source text: use BM25.py

# Baselines
**Random**: random_file.py -> run_generation.py </br>
**Task-level**: create_task_file.py -> run_generation.py </br>
**BM25**: create_BM25_file.py </br>
**R-BM25**: implementation is available at https://github.com/sweta20/inContextMT </br>
**mBART-50**: https://github.com/JoyeBright/MT-HF


## Authors

- **Javad Pourmostafa**  - [Email](mailto:j.pourmostafa@tilburguniversity.edu), [Website](https://javad.pourmostafa.me)
- **Dimitar Shterionov** - [Email](mailto:d.shterionov@tilburguniversity.edu), [Website](https://ilk.uvt.nl/~shterion/)
- **Pieter Spronck**     - [Email](mailto:p.spronck@tilburguniversity.edu), [Website](https://www.spronck.net/)

## Cite the paper
You may cite the preprint version of the paper until it is published in the ACL proceedings.
```
@misc{sharami2024guidingincontextlearningllms,
      title={Guiding In-Context Learning of LLMs through Quality Estimation for Machine Translation}, 
      author={Javad Pourmostafa Roshan Sharami and Dimitar Shterionov and Pieter Spronck},
      year={2024},
      eprint={2406.07970},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.07970}, 
}
```
