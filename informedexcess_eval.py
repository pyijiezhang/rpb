#
# Evaluate the posterior by PAC-Bayes with informed prior + excess loss.
#
# Usage: python informedexcess_eval.py  --<argument1>=[option1] --<argument2>=[option2]
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
# Return: results saved under results/informedexcess
#

""" Using PAC-Bayes method with informed prior and excess loss

This is a method used by e.g. Mhammedi et al. (2019) and Wu and Seldin (2022).

General Info:
# key idea
The underlying idea can be summarized by the decomposition:
E_rho[L(h)] = E_rho[L(h) - L(h^*)] + L(h^*)

1.  (Informed prior: \pi_S1)
    The informed prior \pi_S1 is learned with PAC-Bayes method from S1.
2.  (Excess loss: E_rho[L(h) - L(h^*)])
    -   The reference prediction rule h^* is an ERU from S1.
    -   Bound the excess loss by PAC-Bayes-split-kl inequality (Wu and Seldin (2022)).
3.  (Reference loss: L(h^*))
    Bound the loss by test set bound by Langford (2005).

See the related paper:
1.  PAC-Bayes un-expected Bernstein inequality by Zakaria Mhammedi, Peter Grünwald, and Benjamin Guedj (NeurIPS 2019)
2.  Split-kl and PAC-Bayes-split-kl inequalities for ternary random variables by Yi-Shan Wu and Yevgeny Seldin (NeurIPS 2022)
3.  Tutorial on practical prediction theory for classification by John Langford (JMLR 2005)
"""

import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import time
from rpb import data
from rpb.eval import compute_risk_informedexcess, mcsampling_01


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
    test_loader = data.loadbatches_eval(test, loader_kargs, batch_size, [n_test], seed
    )[0]

    # load model
    exp_settings = f"{name_data}_{model}_{objective}_{seed}.pt"

    dir_posterior = f"./saved_models/informedexcess/posterior_2_" + exp_settings
    posterior = torch.load(dir_posterior, map_location=torch.device(device))

    dir_h = f"./saved_models/informedexcess/posterior_h_" + exp_settings
    h = torch.load(dir_h, map_location=torch.device(device))

    # eval loss
    start = time.time()
    eval_loss = 0
    n_eval = n_posterior
    for _, (input, target) in enumerate(tqdm(eval_loader)):
        input, target = input.to(device), target.to(device)
        eval_loss += mcsampling_01(posterior, input, target) * input.shape[0]
    eval_loss /= n_eval

    # risk
    risk = compute_risk_informedexcess(posterior, h, eval_loader, delta_test, delta)

    end = time.time()
    eval_time = end - start

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

    kl = posterior.compute_kl().detach().cpu().numpy()

    results = {
        "kl": kl,
        "risk": risk, # risk, risk_h, loss_excess, loss_01
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "test_loss": test_loss,
        "eval_time": eval_time,
    }

    if not os.path.exists("./results/informedexcess"):
        os.makedirs("./results/informedexcess")
    results_dir = f"./results/informedexcess/results_" + exp_settings

    with open(results_dir, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
