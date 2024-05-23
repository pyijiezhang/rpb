import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

from rpb import data
from rpb.eval import compute_risk_rpb, get_loss_01


def main(
    name_data="mnist",
    model="fcn",
    objective="fclassic",
    T=6,
    split="geometric",
    gamma_t=0.5,
    seed=0,
    batch_size=128,
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

    eval_loaders = data.loadbatches_eval(
        train, loader_kargs, batch_size, T_splits, seed
    )

    posteriors = []
    T = len(T_splits)
    exp_settings = f"{name_data}_{model}_{objective}_{split}_{T}_{gamma_t}_{seed}.pt"
    for t in range(T):
        dir_posterior = f"./saved_models/rpb/posterior_{t+1}_" + exp_settings
        posterior = torch.load(dir_posterior, map_location=torch.device(device))
        posteriors.append(posterior)

    # compute risk
    loss_ts, kl_ts, E_ts, B_ts = compute_risk_rpb(posteriors, eval_loaders)

    # compute train and test loss
    test_loader = data.loadbatches_eval(test, loader_kargs, n_test, [n_test], seed)
    test_loss_ts = []
    for t in range(T):
        posterior = posteriors[t]
        for _, (input_batch, target_batch) in enumerate(tqdm(test_loader[0])):
            test_loss = get_loss_01(posterior, input_batch, target_batch, sample=True)
            test_loss_ts.append(test_loss.sum().numpy() / n_test)

    results = {
        "loss": loss_ts,
        "kl": kl_ts,
        "excess_risk": E_ts,
        "risk": B_ts,
        "test_loss": test_loss_ts,
    }

    if not os.path.exists("./results/rpb"):
        os.makedirs("./results/rpb")
    results_dir = f"./results/rpb/results_" + exp_settings

    with open(results_dir, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
