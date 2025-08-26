# LLM-Enhanced Linear Autoencoders for Recommendation (CIKM 2025)

This is the official code for our accepted CIKM 2025 paper: <br>[`LLM-Enhanced Linear Autoencoders for Recommendation`](https://arxiv.org/abs/2508.13500).</br>

The slides can be found [here](https://drive.google.com/file/d/1gW-E8iFiUScBBs_N7QEIjEsYlyRr2jrG/view?usp=sharing), and a summary is on [our lab blog](https://dial.skku.edu/blog/2025_l3ae) (in Korean).

## Citation

Please cite our paper if using this code.

```
@inproceedings{MoonPL25L3AE,
  author    = {Jaewan Moon and
               Seongmin Park and
               Jongwuk Lee},
  title     = {LLM-Enhanced Linear Autoencoders for Recommendation},
  booktitle = {CIKM},
  pages     = {to be updated},
  year      = {2025},
}
```

---

## Setup Python environment

### Install python environment

```bash
conda env create -f environment.yml   
```

### Activate environment
```bash
conda activate L3AE
```

---

## Datasets
### Usage
1. Create a data directory at the project root:
   ```bash
   mkdir -p data
   ```
2. Download the datasets and unzip them into the `data` directory.
     - [Games](https://drive.google.com/file/d/1do9gCPqvxNXkf2J8nPfZkJrDkdPE3htz/view?usp=sharing), [Toys](https://drive.google.com/file/d/1nQiYUmIcJO5s1ZmGS-RH6Y524Y9LdHxC/view?usp=sharing), and [Books](https://drive.google.com/file/d/1juy8y3R_VIKqMwZ8mzgMtwNoinezq60x/view?usp=sharing)

  After extraction, your directory structure look like:
  ```bash
  <project root>/
    data/
      games/
      toys/
      books/
    encoder/
    script/
    README.md
    requirements.txt
    run_script.sh
  ```
  
---

## Reproducibility
### Usage
- To reproduce the results of paper, run the `run_script.sh` shell script. This script contains the following lines:
```bash
sh script/Table2_best_param_nv-embed-v2_games.sh
sh script/Table2_best_param_nv-embed-v2_toys.sh
sh script/Table2_best_param_nv-embed-v2_books.sh

sh script/Table3_performance_over_fusion_methods.sh

sh script/Table4_best_param_llama-3.2-3b_games.sh
sh script/Table4_best_param_llama-3.2-3b_toys.sh
sh script/Table4_best_param_llama-3.2-3b_books.sh

sh script/Table4_best_param_qwen3-embedding-8b_games.sh
sh script/Table4_best_param_qwen3-embedding-8b_toys.sh
sh script/Table4_best_param_qwen3-embedding-8b_books.sh
```

#### Arguments
- See more arguments in `encoder/config/modelconf` for each model. You can also find arguments in `encoder/config/configurator.py` and `encoder/config/linear_configurator.py`.
- dataset: Games, Toys, and Books (amazon 2023 datasets)
- linear models
    - EASE, GF-CF, BSPM, SGFCF, Collective EASE, Addictive EASE, and **L^3AE**
    - `L3AE_Collective` and `L3AE_Addictive` models are fusion variants of **L^3AE**.
- non-linear models
    - LightGCN, SimGCL, RLMRec-Con-LightGCN, RLMRec-Gen-LightGCN, RLMRec-Con-SimGCL, RLMRec-Con-SimGCL, AlphaRec
