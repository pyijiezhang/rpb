# Recursive PAC-Bayes: A Frequentist Approach to Sequential Prior Updates with No Information Loss (NeurIPS 2024, Spotlight)

[![](https://img.shields.io/badge/arXiv-2203.16481-red)](https://arxiv.org/abs/2405.14681)

This is the official PyTorch implementation of `Recursive PAC-Bayes`, proposed in the paper _Recursive PAC-Bayes: A Frequentist Approach to Sequential Prior Updates with No Information Loss_ (published at NeurIPS 2024, spotlight). 

## Setup
All requirements are listed in [environment.yml](./environment.yml). Create a `conda` environment using:
```
conda env create -n <env_name>
```

## Usage
For each of the four methods (Uninformed, Informed, InformedExcess, and RPB), run the corresponding script to train and save the models:
```shell
python rpb_train.py --name_data="mnist" \
                    --model="fcn" \
                    --objective="fclassic" \
                    --T=6 \
                    --split="geometric" \
                    --gamma_t=0.5 \
                    --recursive_step_1=False \
                    --sigma_prior=0.03 \
                    --pmin=1e-5 \
                    --delta=0.025 \
                    --delta_test=0.01 \
                    --kl_penalty=1 \
                    --initial_lamb=1.0 \
                    --train_epochs=200 \
                    --learning_rate=0.001 \
                    --momentum=0.95 \
                    --batch_size=250 \
                    --verbose=True \
                    --seed=0 
```
Then, to evaluate the risk, run the corresponding script to evaluate the saved models:
```shell
python rpb_eval.py --name_data="mnist" \
                   --model="fcn" \
                   --objective="fclassic" \
                   --T=6 \
                   --split="geometric" \
                   --gamma_t=0.5 \
                   --recursive_step_1=False \
                   --seed=0 \
                   --batch_size=250 \
                   --risk_laststep=False
```
### Bibtex
If you find our paper and/or code useful for your research, please consider citing our paper:
```
@misc{wu2024recursivepacbayesfrequentistapproach,
      title={Recursive PAC-Bayes: A Frequentist Approach to Sequential Prior Updates with No Information Loss}, 
      author={Yi-Shan Wu and Yijie Zhang and Badr-Eddine Chérief-Abdellatif and Yevgeny Seldin},
      year={2024},
      eprint={2405.14681},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.14681}, 
}
```