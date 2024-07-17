#
# Runs optimization procedure for PAC-Bayes with uninformed prior.
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
# Return: posteriors saved under saved_models/uninformed
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
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
from rpb.models import Lambda_var, trainPNNet
from rpb.bounds import PBBobj
from rpb import data
from rpb.utils import init_posterior


def main(
    name_data="mnist",
    objective="fclassic",
    model="fcn",
    sigma_prior=0.03,
    pmin=1e-5,
    learning_rate=0.005,
    momentum=0.95,
    delta=0.025,
    delta_test=0.01,
    kl_penalty=1,
    initial_lamb=6.0,
    train_epochs=50,
    verbose=True,
    batch_size=250,
    seed=0,
):

    exp_settings = f"{name_data}_{model}_{objective}_{seed}.pt"
    if not os.path.exists("./saved_models/uninformed"):
        os.makedirs("./saved_models/uninformed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_kargs = (
        {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    )

    # load data
    train, _ = data.loaddataset(name_data)
    classes = len(train.classes)
    n_train = len(train.data)
    train_loader = data.loadbatches_train(
        train, loader_kargs, batch_size, [n_train], seed
    )[0]
    n_posterior = n_train

    # train the posterior
    posterior = init_posterior(model, sigma_prior, prior=None, device=device)

    pbobj = PBBobj(
        objective,
        pmin,
        classes,
        delta,
        delta_test,
        kl_penalty,
        device,
        n_posterior,
        use_excess_loss=False,
    )

    if objective == "flamb":
        lambda_var = Lambda_var(initial_lamb, n_posterior).to(device)
        optimizer_lambda = optim.SGD(
            lambda_var.parameters(), lr=learning_rate, momentum=momentum
        )
    else:
        optimizer_lambda = None
        lambda_var = None

    optimizer = optim.SGD(posterior.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in trange(train_epochs):
        trainPNNet(
            posterior,
            optimizer,
            pbobj,
            epoch,
            train_loader,
            lambda_var,
            optimizer_lambda,
            verbose,
        )

    dir_posterior = f"./saved_models/uninformed/posterior_" + exp_settings
    torch.save(posterior, dir_posterior)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
