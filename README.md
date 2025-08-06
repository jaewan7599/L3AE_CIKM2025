# LLM-Enhanced Linear Autoencoders for Recommendation (CIKM 2025)

This is the official code for our accepted SIGIR 2025 paper: <br>[`LLM-Enhanced Linear Autoencoders for Recommendation`](https://arxiv.org/abs/2305.12922).</br>

The slides can be found [here](https://drive.google.com/file/d/1gW-E8iFiUScBBs_N7QEIjEsYlyRr2jrG/view?usp=sharing).

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
- To reproduce the results of Table 1, go to the '' directory.

#### In terminal
- Run the shell file for one of the datasets.

#### Arguments (see more arguments in `parse.py`)
- dataset: games, toys and books
- linear models
    - EASE, GF-CF, BSPM, SGFCF, CEASE, ADDEASE **L^3AE**
- hyperparams1 (for specific models)
    - 
- hyperparams2
    - [0.1, 0.2, ..., 0.9]
