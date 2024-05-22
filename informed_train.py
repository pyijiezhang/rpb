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

    train, _ = data.loaddataset(name_data)
    classes = len(train.classes)
    n_train = len(train.data)
    n_prior = int(n_train / 2)
    n_posterior = n_train - n_prior
    train_loader_prior, train_loader_posterior = data.loadbatches_train(
        train,
        loader_kargs,
        batch_size,
        [n_prior, n_posterior],
        seed,
    )

    prior = init_posterior(model, sigma_prior, prior=None, device=device)

    bound = PBBobj(
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
            bound,
            epoch,
            train_loader_prior,
            lambda_var,
            optimizer_lambda,
            verbose,
        )

    dir_prior = f"./saved_models/informed/posterior_1_" + exp_settings
    torch.save(prior, dir_prior)

    posterior = init_posterior(model, sigma_prior, prior=prior, device=device)

    bound = PBBobj(
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
            bound,
            epoch,
            train_loader_posterior,
            lambda_var,
            optimizer_lambda,
            verbose,
        )

    dir_posterior = f"./saved_models/informed/posterior_2_" + exp_settings
    torch.save(posterior, dir_posterior)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
