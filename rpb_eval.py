import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

from rpb import data
from rpb.eval import (
    compute_risk_rpb,
    compute_risk_rpb_recursive_step_1,
    mcsampling_01,
    compute_risk_rpb_laststep,
)


def main(
    name_data="mnist",
    model="fcn",
    layers=4,
    objective="fclassic",
    T=6,
    split="geometric",
    gamma_t=0.5,
    recursive_step_1=False,
    seed=0,
    batch_size=250,
    risk_laststep=False,
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

    if split == "uniform":
        T_splits = [int(n_train / T)] * (T - 1)
        T_splits.append(n_train - int(n_train / T) * (T - 1))
    elif split == "geometric":
        if T == 2:
            T_splits = [20000, 40000]
        elif T == 4:
            T_splits = [7500, 7500, 15000, 30000]
        elif T == 6:
            T_splits = [1875, 1875, 3750, 7500, 15000, 30000]
        elif T == 8:
            T_splits = [468, 469, 938, 1875, 3750, 7500, 15000, 30000]

    eval_loaders = data.loadbatches_eval(
        train, loader_kargs, batch_size, T_splits, seed
    )

    # load posteriors
    posteriors = []
    exp_settings = f"{name_data}_{model}_{layers}_{objective}_{split}_{T}_{recursive_step_1}_{gamma_t}_{seed}.pt"

    if recursive_step_1:
        start_step = 0
    else:
        start_step = 1

    for t in range(start_step, T + 1):
        dir_posterior = f"./saved_models/rpb/posterior_{t}_" + exp_settings
        posterior = torch.load(dir_posterior, map_location=torch.device(device))
        posteriors.append(posterior)

    # compute risk
    if risk_laststep:
        E_ts = [0.0]
        loss_ts, kl_ts, B_ts = compute_risk_rpb_laststep(
            posteriors[-1], eval_loaders[-1]
        )
    elif recursive_step_1:
        loss_ts, kl_ts, E_ts, B_ts = compute_risk_rpb_recursive_step_1(
            posteriors, eval_loaders
        )
    else:
        loss_ts, kl_ts, E_ts, B_ts = compute_risk_rpb(posteriors, eval_loaders)

    # compute train and test loss
    train_loader = data.loadbatches_eval(
        train, loader_kargs, batch_size, [n_train], seed
    )[0]
    test_loader = data.loadbatches_eval(test, loader_kargs, batch_size, [n_test], seed)[
        0
    ]

    test_loss_ts = []
    for t in range(1, T + 1):
        posterior = posteriors[t - start_step]

        # compute the test loss of all the posteriors
        test_loss = 0
        for _, (input, target) in enumerate(tqdm(test_loader)):
            input, target = input.to(device), target.to(device)
            test_loss += mcsampling_01(posterior, input, target) * input.shape[0]
        test_loss /= n_test
        test_loss_ts.append(test_loss)

        if t == T:
            # compute the train loss of the last posterior
            train_loss_T = 0
            for _, (input, target) in enumerate(tqdm(train_loader)):
                input, target = input.to(device), target.to(device)
                train_loss_T += mcsampling_01(posterior, input, target) * input.shape[0]
            train_loss_T /= n_train

    results = {
        "loss": loss_ts,
        "kl": kl_ts,
        "excess_risk": E_ts,
        "risk": B_ts,
        "train_loss": train_loss_T,
        "test_loss": test_loss_ts,
    }

    if not os.path.exists("./results/rpb"):
        os.makedirs("./results/rpb", exist_ok=True)

    exp_settings = f"{name_data}_{model}_{layers}_{objective}_{split}_{T}_{recursive_step_1}_{risk_laststep}_{gamma_t}_{seed}.pt"
    results_dir = f"./results/rpb/results_" + exp_settings

    with open(results_dir, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
