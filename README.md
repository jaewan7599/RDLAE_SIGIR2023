# RLAE & RDLAE - It's Enough: Relaxing Diagonal Constraints in Linear Autoencoders for Recommendation (SIGIR 2023)

This is the official code for our accepted SIGIR 2023 paper: <br>[`It's Enough: Relaxing Diagonal Constraints in Linear Autoencoders for Recommendation`](https://arxiv.org/abs/2305.12922).</br>


We implemented our framework based on the following github repositories; [**MultVAE**](https://github.com/dawenl/vae_cf) and [**LT-OCF**](https://github.com/jeongwhanchoi/LT-OCF).</br> 

The slides can be found [here](https://drive.google.com/file/d/1gW-E8iFiUScBBs_N7QEIjEsYlyRr2jrG/view?usp=sharing).

## Citation

Please cite our paper if using this code.

```
@inproceedings{MoonKL23RDLAE,
  author    = {Jaewan Moon and
               Hye-young Kim and
               Jongwuk Lee},
  title     = {It's Enough: Relaxing Diagonal Constraints in Linear Autoencoders for Recommendation},
  booktitle = {SIGIR},
  pages     = {--(will be updated)},
  year      = {2023},
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
conda activate RDLAE
```

---

## Datasets
- Strong generalization: https://drive.google.com/file/d/1qRDWRMp5U86jwInnWT6OirsjT4UKhNE2/view?usp=sharing
- Weak generalization: https://drive.google.com/file/d/1Yo5roKrJ3mkKTOSHxFNz9RoOjEueQnpS/view?usp=sharing

---

## Reproducibility
### Usage
- To reproduce the results of Table 2 (strong generalization), go to the 'strong' directory.
- To reproduce the results of Table 3 (weak generalization), go to the 'weak' directory.

#### In terminal
- Run the shell file for one of the datasets at the specific directory, i.e., 'strong' and 'weak', of the project.

#### Arguments (see more arguments in `parse.py`)
- dataset
    - strong: ml-20m, netflix, msd, gowalla, yelp2018, amazon-book
    - weak: gowalla, yelp2018, amazon-book
- model
    - EASE, EDLAE, **RLAE**, **RDLAE**
- diag_const (for EASE and EDLAE)
    - True, False
- drop_p, xi
    - [0.1, 0.2, ..., 0.9]
