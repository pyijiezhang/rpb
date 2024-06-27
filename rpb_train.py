#
# Runs optimization procedure for recursive PAC-Bayes.
#
# Usage: python rpb_train.py  --<argument1>=[option1] --<argument2>=[option2]
#        name_data   : 'mnist', 'fmnist'
#        model       : 'fcn', 'cnn'
#                      'fcn'        = fully connected network
#                                     used for "name_data" = 'mnist'
#                      'cnn'        = convolution neural network
#                                     used for "name_data" = 'fmnist
#        layer       : 4 (default), add more later
#        objective   : 'fclassic' (default), 'fquad', 'flamb', 'bbb'
#                      'fclassic'   = McAllester's bound
#                      'fquad'      = PAC-Bayes-quadratic bound by Rivasplata et al., 2019
#                      'flamb'      = PAC-Bayes-lambda by Thiemann et al., 2017
#                      'bbb'        = a PAC-Bayes inspired optimization objective (see Rivasplata et al., 2019)
#        T           : integer              : in general
#                    : 2, 4, 6 (default), 8 : when "split"='geometric'
#        split       : 'uniform', 'geometric' (default)
#        gamma_t     : real value in (0, 1)
#        recursive_step_1     : bool
#                               true              = recursion from t=1
#                               false (default)   = recursion from t=2
#
# Return: posteriors saved under saved_models/rpb
#

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
    model="fcn",
    layers=4,
    objective="fclassic",
    T=6,
    split="geometric",
    gamma_t=0.5,
    recursive_step_1=False,
    sigma_prior=0.03,
    pmin=1e-5, # lower-bounding the probability assigned to Y
               # to give a bounded cross-entropy loss
    delta=0.025,
    delta_test=0.01, # MC evaluation of a fixed posterior
                     # Used only in evaluation
    kl_penalty=1, # for objective='bbb', not used in this work
    initial_lamb=1.0,
    train_epochs=50,
    learning_rate=0.005,
    momentum=0.95,
    batch_size=250,
    verbose=True,
    seed=0,
):

    exp_settings = f"{name_data}_{model}_{layers}_{objective}_{split}_{T}_{recursive_step_1}_{gamma_t}_{seed}.pt"
    if not os.path.exists(f"./saved_models/rpb"):
        os.makedirs(f"./saved_models/rpb", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    loader_kargs = (
        {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    )

    train, _ = data.loaddataset(name_data)
    classes = len(train.classes)
    n_train = len(train.data)

    if split == "uniform":
        T_splits = [int(n_train / T)] * (T-1)
        T_splits.append(n_train - int(n_train / T)*(T-1))
    elif split == "geometric":
        if T == 2:
            T_splits = [30000, 30000]
        elif T == 4:
            T_splits = [7500, 7500, 15000, 30000]
        elif T == 6:
            T_splits = [1875, 1875, 3750, 7500, 15000, 30000]
        elif T == 8:
            T_splits = [468, 469, 938, 1875, 3750, 7500, 15000, 30000]

    train_loaders = data.loadbatches_train(
        train, loader_kargs, batch_size, T_splits, seed
    )

    n_train_t_cumsum = np.cumsum(
        [len(train_loader.sampler.indices) for train_loader in train_loaders]
    )

    for t in range(1, T + 1): # following the index in the paper

        train_loader = train_loaders[t - 1]

        if t == 1:
            # initialize an uninformed prior pi_0
            prior = init_posterior(model, sigma_prior, prior=None, device=device)
            n_posterior = n_train # n^val_t in the paper

            if recursive_step_1:
                use_excess_loss = True
            else:
                use_excess_loss = False

            dir_prior = f"./saved_models/rpb/posterior_0_" + exp_settings
            torch.save(prior, dir_prior)
        else:
            # initialize the prior by the previous posterior
            prior = torch.load(dir_posterior, map_location=torch.device(device))
            n_posterior = n_train - n_train_t_cumsum[t - 2] # n^val_t in the paper
            use_excess_loss = True

        print("Current step:", t)
        print("n_posterior:", n_posterior)

        # initialize posterior
        posterior = init_posterior(model, sigma_prior, prior, device)

        # define the PAC-Bayes inspired bound used to learn the posterior
        pbobj = PBBobj(
            objective,
            pmin,
            classes,
            delta,
            delta_test, # not used in optimization
            kl_penalty, # not used so =1 in the work
            device,
            n_posterior, # n^val_t in the paper
            use_excess_loss=use_excess_loss,
        )

        if objective == "flamb":
            lambda_var = Lambda_var(initial_lamb, n_posterior).to(device)
            optimizer_lambda = optim.SGD(
                lambda_var.parameters(), lr=learning_rate, momentum=momentum
            )
        else:
            optimizer_lambda = None
            lambda_var = None

        # define the optimizer
        optimizer = optim.SGD(
            posterior.parameters(), lr=learning_rate, momentum=momentum
        )

        # train the posterior for - train_epochs - epochs
        for epoch in trange(train_epochs):
            trainPNNet(
                posterior,
                optimizer, # using the defined optimizer
                pbobj, # using the defined bound
                epoch, # the current training epoch
                train_loader,
                lambda_var,
                optimizer_lambda,
                verbose,
                prior, # using the defined prior
                gamma_t, # using the offset gamma_t when training with the excess loss
            )

        dir_posterior = f"./saved_models/rpb/posterior_{t}_" + exp_settings
        torch.save(posterior, dir_posterior)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
