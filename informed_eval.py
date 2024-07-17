#
# Evaluate the posterior by PAC-Bayes with informed prior.
#
# Usage: python rpb_train.py  --<argument1>=[option1] --<argument2>=[option2]
#        name_data   : 'mnist', 'fmnist'
#        model       : 'fcn', 'cnn'
#                      'fcn'        = fully connected network
#                                     used for "name_data" = 'mnist'
#                      'cnn'        = convolution neural network
#                                     used for "name_data" = 'fmnist
#        objective   : 'fclassic' (default), 'fquad', 'flamb', 'bbb'
#                      'fclassic'   = McAllester's bound
#                      'fquad'      = PAC-Bayes-quadratic bound by Rivasplata et al., 2019
#                      'flamb'      = PAC-Bayes-lambda by Thiemann et al., 2017
#                      'bbb'        = a PAC-Bayes inspired optimization objective (see Rivasplata et al., 2019)
#
# Return: posteriors saved under results/informed
#

"""
This is initiated by Ambroladze et al. (2007) and later improved by Pérez-Ortiz et al. (2021).

# key idea - We demonstrate with the classic McAllester's bound:
E_rho[L(h)] \le E_rho[\hat L(h,_)] + \sqrt{ KL(rho || pi_S1)/2|_| }

-   Both take a part of the dataset S1 to learn the informed prior \pi_S1.
-   Both take the rest of the dataset S2=S\S1 to evaluate the posterior rho, i.e., place S2 in _.
-   The only difference comes when learning the posterior rho:
        During training, as rho can depend on entire S, it is valid to put S into _ to learn rho.
        Pérez-Ortiz et al. (2021) did that, while Ambroladze et al. (2007) only put S2 into _ to learn rho.

See the related paper:
1.  Tighter PAC-Bayes bounds by Amiran Ambroladze, Emilio Parrado-Hernández, and John Shawe-Taylor (NeurIPS 2007)
2.  Tighter risk certificates for neural networks by María Pérez-Ortiz, Omar Rivasplata, John Shawe-Taylor, and Csaba Szepesvári (JMLR 2021)
3.  Recursive PAC-Bayes: A Frequentist Approach to Sequential Prior Updates with No Information Loss by Yi-Shan Wu, Yijie Zhang, Badr-Eddine Chérief-Abdellatif, and Yevgeny Seldin (2024)
"""

import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

from rpb import data
from rpb.eval import mcsampling_01, solve_kl_sup


def main(
    name_data="mnist",
    model="fcn",
    objective="fclassic",
    seed=0,
    delta_test=0.01,
    delta=0.025,
    batch_size=250,
):

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader_kargs = (
        {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    )

    # load data
    train, test = data.loaddataset(name_data)
    n_train = len(train)
    n_test = len(test)
    n_prior = int(n_train / 2)
    n_posterior = n_train - n_prior

    eval_loader = data.loadbatches_eval(
        train, loader_kargs, batch_size, [n_prior, n_posterior], seed
    )[-1]
    train_loader = data.loadbatches_eval(
        train, loader_kargs, batch_size, [n_train], seed
    )[0]
    test_loader = data.loadbatches_eval(test, loader_kargs, batch_size, [n_test], seed)[
        0
    ]

    # load model
    exp_settings = f"{name_data}_{model}_{objective}_{seed}.pt"
    dir_posterior = f"./saved_models/informed/posterior_2_" + exp_settings
    posterior = torch.load(dir_posterior, map_location=torch.device(device))

    # eval loss
    eval_loss = 0
    n_bound = n_posterior
    for _, (input, target) in enumerate(tqdm(eval_loader)):
        input, target = input.to(device), target.to(device)
        eval_loss += mcsampling_01(posterior, input, target) * input.shape[0]
    eval_loss /= n_bound

    # risk
    mc_samples = n_posterior
    inv_1 = solve_kl_sup(eval_loss, np.log(1 / delta_test) / mc_samples)
    kl = posterior.compute_kl().detach().cpu().numpy()
    risk = solve_kl_sup(
        inv_1,
        (kl + np.log((2 * np.sqrt(n_bound)) / delta)) / n_bound,
    )

    # train loss
    train_loss = 0
    for _, (input, target) in enumerate(tqdm(train_loader)):
        input, target = input.to(device), target.to(device)
        train_loss += mcsampling_01(posterior, input, target) * input.shape[0]
    train_loss /= n_train

    # test loss
    test_loss = 0
    for _, (input, target) in enumerate(tqdm(test_loader)):
        input, target = input.to(device), target.to(device)
        test_loss += mcsampling_01(posterior, input, target) * input.shape[0]
    test_loss /= n_test

    results = {
        "kl": kl,
        "risk": risk,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "test_loss": test_loss,
    }

    if not os.path.exists("./results/informed"):
        os.makedirs("./results/informed")
    results_dir = f"./results/informed/results_" + exp_settings

    with open(results_dir, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
