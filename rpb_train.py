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
    objective="fclassic",
    T=6,
    split="geometric",
    gamma_t=0.5,
    sigma_prior=0.03,
    pmin=1e-5,
    delta=0.025,
    delta_test=0.01,
    kl_penalty=1,
    initial_lamb=1.0,
    train_epochs=50,
    learning_rate=0.005,
    momentum=0.95,
    batch_size=250,
    verbose=True,
    seed=0,
):

    exp_settings = f"{name_data}_{model}_{objective}_{split}_{T}_{gamma_t}_{seed}.pt"
    if not os.path.exists("./saved_models/rpb"):
        os.makedirs("./saved_models/rpb")

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
        T_splits = [int(n_train / T)] * T
    elif split == "geometric":
        if T == 2:
            T_splits = [20000, 40000]
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

    for t in range(T):

        train_loader = train_loaders[t]

        if t == 0:
            prior = init_posterior(model, sigma_prior, prior=None, device=device)
            n_posterior = n_train
            use_excess_loss = False

            dir_prior = f"./saved_models/rpb/posterior_0_" + exp_settings
            torch.save(prior, dir_prior)
        else:
            prior = torch.load(dir_posterior, map_location=torch.device(device))
            n_posterior = n_train - n_train_t_cumsum[t - 1]
            use_excess_loss = True

        posterior = init_posterior(
            model,
            sigma_prior,
            prior,
            device,
        )

        bound = PBBobj(
            objective,
            pmin,
            classes,
            delta,
            delta_test,
            kl_penalty,
            device,
            n_posterior,
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

        optimizer = optim.SGD(
            posterior.parameters(), lr=learning_rate, momentum=momentum
        )

        for epoch in trange(train_epochs):
            trainPNNet(
                posterior,
                optimizer,
                bound,
                epoch,
                train_loader,
                lambda_var,
                optimizer_lambda,
                verbose,
                prior,
                gamma_t,
            )

        dir_posterior = f"./saved_models/rpb/posterior_{t+1}_" + exp_settings
        torch.save(posterior, dir_posterior)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
