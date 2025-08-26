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
- https://drive.google.com/file/d/1qRDWRMp5U86jwInnWT6OirsjT4UKhNE2/view?usp=sharing

---

## Reproducibility
### Usage
- To reproduce the results of paper, run `run_script.sh` shell script file. This script file contains script lines as belows:
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

#### Arguments (See more arguments in `encoder/config/modelconf` for each model. You can also find arguments in `encoder/config/configurator.py` or `encoder/config/linear_configurator.py`)
- dataset: games, toys, and books (amazon 2023 datasets)
- linear models
    - EASE, GF-CF, BSPM, SGFCF, Collective EASE, Addictive EASE, and **L^3AE**
- non-linear models
    - LightGCN, SimGCL, RLMRec-Con-LightGCN, RLMRec-Gen-LightGCN, RLMRec-Con-SimGCL, RLMRec-Con-SimGCL
