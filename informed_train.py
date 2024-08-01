#
# Runs optimization procedure for PAC-Bayes with informed prior.
#
# Usage: python informed_train.py  --<argument1>=[option1] --<argument2>=[option2]
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
# Return: posteriors saved under saved_models/informed
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
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import time
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
    if not os.path.exists("./saved_models/informed"):
        os.makedirs("./saved_models/informed")

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
    n_prior = int(n_train / 2)
    n_posterior = n_train - n_prior
    train_loader = data.loadbatches_train(
        train,
        loader_kargs,
        batch_size,
        [n_prior, n_posterior],
        seed,
    )

    start = time.time()

    train_loader_prior, train_loader_posterior = train_loader[0], train_loader[1]

    # train the informed prior
    prior = init_posterior(model, sigma_prior, prior=None, device=device)

    pbobj = PBBobj(
        objective,
        pmin,
        classes,
        delta,
        delta_test,
        kl_penalty,
        device,
        n_prior,
        use_excess_loss=False,
    )

    if objective == "flamb":
        lambda_var = Lambda_var(initial_lamb, n_prior).to(device)
        optimizer_lambda = optim.SGD(
            lambda_var.parameters(), lr=learning_rate, momentum=momentum
        )
    else:
        optimizer_lambda = None
        lambda_var = None

    optimizer = optim.SGD(prior.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in trange(train_epochs):
        trainPNNet(
            prior,
            optimizer,
            pbobj,
            epoch,
            train_loader_prior,
            lambda_var,
            optimizer_lambda,
            verbose,
        )

    dir_prior = f"./saved_models/informed/posterior_1_" + exp_settings
    torch.save(prior, dir_prior)

    # train the posterior
    posterior = init_posterior(model, sigma_prior, prior=prior, device=device)

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
            train_loader_posterior,
            lambda_var,
            optimizer_lambda,
            verbose,
        )

    dir_posterior = f"./saved_models/informed/posterior_2_" + exp_settings
    torch.save(posterior, dir_posterior)

    end = time.time()
    print("Train time: ", end - start)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
