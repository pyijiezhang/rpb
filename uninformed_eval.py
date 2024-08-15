#
# Evaluate the posterior by PAC-Bayes with uninformed prior.
#
# Usage: python uninformed_eval.py  --<argument1>=[option1] --<argument2>=[option2]
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
# Return: results saved under results/uninformed
#

""" Using PAC-Bayes method with uninformed prior.

This is the classic method for PAC-Bayes analysis, where the prior is uninformed.

General Info:
# Prior:
The prior is uninformed (data-independent). We choose a Gaussian with mean (rand_init) and a diagonal covariance (\sqrt{0.03}I)

See the related paper:
1. Some PAC-Bayesian theorems by David McAllester (COLT 1998)
2. A note on the PAC-Bayesian theorem by Andreas Maurer (2004)
3. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data
by Gintare Karolina Dziugaite and Daniel M. Roy (UAI 2017)
"""

import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import time
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
    train_loader = data.loadbatches_eval(
        train, loader_kargs, batch_size, [n_train], seed
    )[0]
    test_loader = data.loadbatches_eval(
        test, loader_kargs, batch_size, [n_test], seed
    )[0]

    # load model
    exp_settings = f"{name_data}_{model}_{objective}_{seed}.pt"
    dir_posterior = f"./saved_models/uninformed/posterior_" + exp_settings
    posterior = torch.load(dir_posterior, map_location=torch.device(device))

    start = time.time()
    # train loss
    train_loss = 0
    for _, (input, target) in enumerate(tqdm(train_loader)):
        input, target = input.to(device), target.to(device)
        train_loss += mcsampling_01(posterior, input, target) * input.shape[0]
    train_loss /= n_train

    # risk
    eval_loss = train_loss
    mc_samples = n_train
    n_bound = n_train
    inv_1 = solve_kl_sup(eval_loss, np.log(1 / delta_test) / mc_samples)
    kl = posterior.compute_kl().detach().cpu().numpy()
    risk = solve_kl_sup(
        inv_1,
        (kl + np.log((2 * np.sqrt(n_bound)) / delta)) / n_bound,
    )
    end = time.time()
    eval_time = end - start

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
        "eval_time": eval_time,
    }

    if not os.path.exists("./results/uninformed"):
        os.makedirs("./results/uninformed")
    results_dir = f"./results/uninformed/results_" + exp_settings

    with open(results_dir, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Evaluation time:", eval_time)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
