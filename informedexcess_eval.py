import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

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
    test_loader = data.loadbatches_eval(test, loader_kargs, batch_size, [n_test], seed)[
        0
    ]

    # load model
    exp_settings = f"{name_data}_{model}_{objective}_{seed}.pt"

    dir_posterior = f"./saved_models/informedexcess/posterior_2_" + exp_settings
    posterior = torch.load(dir_posterior, map_location=torch.device(device))

    dir_h = f"./saved_models/informedexcess/posterior_h_" + exp_settings
    h = torch.load(dir_h, map_location=torch.device(device))

    # eval loss
    eval_loss = 0
    n_eval = 30000
    for _, (input, target) in enumerate(tqdm(eval_loader)):
        input, target = input.to(device), target.to(device)
        eval_loss += mcsampling_01(posterior, input, target) * input.shape[0]
    eval_loss /= n_eval

    # risk
    risk = compute_risk_informedexcess(posterior, h, eval_loader, delta_test, delta)

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

    kl = posterior.compute_kl().detach().numpy()

    results = {
        "kl": kl,
        "risk": risk,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "test_loss": test_loss,
    }

    if not os.path.exists("./results/informedexcess"):
        os.makedirs("./results/informedexcess")
    results_dir = f"./results/informedexcess/results_" + exp_settings

    with open(results_dir, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
