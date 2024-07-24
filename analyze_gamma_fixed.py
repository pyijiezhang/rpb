import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
import pickle
from rpb.models import Lambda_var, trainPNNet
from rpb.bounds import PBBobj
from rpb import data
from rpb.utils import init_posterior
from rpb.eval import compute_risk_rpb_onestep


def main(
    name_data="mnist",
    model="fcn",
    objective="fclassic",
    T=6,
    split="geometric",
    gamma_t_model=0.5,
    recursive_step_1=False,
    gamma_ts=[0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    delta=0.025,
    delta_test=0.01,
    batch_size=250,
    seed=0,
):

    exp_settings = f"{name_data}_{model}_{objective}_{split}_{T}_{recursive_step_1}_{gamma_t_model}_{seed}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    loader_kargs = (
        {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    )

    train, _ = data.loaddataset(name_data)
    n_train = len(train.data)

    if split == "uniform":
        T_splits = [int(n_train / T)] * T
    elif split == "geometric":
        if T == 2:
            T_splits = [30000, 30000]
        elif T == 4:
            T_splits = [7500, 7500, 15000, 30000]
        elif T == 6:
            T_splits = [1875, 1875, 3750, 7500, 15000, 30000]
        elif T == 8:
            T_splits = [468, 469, 938, 1875, 3750, 7500, 15000, 30000]

    eval_loaders = data.loadbatches_eval(
        train, loader_kargs, batch_size, T_splits, seed
    )

    if recursive_step_1:
        start_step = 1
    else:
        start_step = 2

    results_gamma = {}
    for t in range(start_step, T + 1):

        print("Current step:", t)

        results_gamma[t] = {}

        eval_loader = eval_loaders[t - 1]

        dir_prior = f"./saved_models/rpb/posterior_{t-1}_" + exp_settings
        prior = torch.load(dir_prior, map_location=torch.device(device))

        dir_posterior = f"./saved_models/rpb/posterior_{t}_" + exp_settings
        posterior = torch.load(dir_posterior, map_location=torch.device(device))

        for gamma_t in gamma_ts:

            print("Current gamma_t:", gamma_t)

            loss_excess, loss_excess_sum, E_t, kl, loss_prior, loss_posterior = compute_risk_rpb_onestep(
                posterior, prior, eval_loader, gamma_t, T, delta_test, delta
            )
            results = {
                "loss_excess": loss_excess,
                "loss_excess_sum": loss_excess_sum,
                "posterior-gam_prior": loss_posterior - gamma_t * loss_prior,
                "E_t": E_t,
                "kl": kl,
                "loss_prior": loss_prior,
                "loss_posterior": loss_posterior,
            }
            print("Current results: ", results)
            results_gamma[t][gamma_t] = results

        if not os.path.exists("./results/rpb"):
            os.makedirs("./results/rpb", exist_ok=True)
        results_dir = f"./results/rpb/results_gamma_fixed_" + exp_settings

        with open(results_dir, "wb") as handle:
            pickle.dump(results_gamma, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import fire

    fire.Fire(main)