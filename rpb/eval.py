import torch
import numpy as np
from scipy.stats import binom
from scipy import optimize
from math import log
from tqdm import tqdm, trange
import torch.nn.functional as F


def get_loss_01(pi, input, target, sample=True):
    """Compute 0-1 loss of h~pi(h) for a data.
    Inputs:
        pi      - can be prior or posterior
        input   - data x
        target  - data x
    Outputs: a vector with size = len(X)
    """
    outputs = pi(input, sample=sample)
    outputs = F.log_softmax(outputs, dim=1)
    pred = outputs.max(1)[1]
    loss_01 = (pred != target).long()
    return loss_01


def get_excess_j(loss_01_posterior, loss_01_prior, js, gamma_t):
    """Compute excess loss delta_j^{\hat}(h_2, h_1, x, y) for a data and each j.
    Inputs:
        loss_01_posterior    - the 0-1 loss of posteriors on (x,y)
        loss_01_prior        - the 0-1 loss of priors on (x,y)
        js                   - the thresholds for excess loss
    Output: The excess loss of (x,y) for each j in js
            with shape = len(js)
    """
    delta_js = []
    for j in js:
        # compute the indicator function
        delta_j = ((loss_01_posterior - gamma_t * loss_01_prior) >= j - 1e-7).float()
        delta_js.append(delta_j)
    return torch.tensor(delta_js)


def mcsampling_01(pi, input, target, sample=True):
    """Compute the empirical mean of 0-1 loss of h~pi(h) for data (X, Y).
    Inputs:
        pi      - can be prior or posterior
        input   - data X
        target  - data Y
    Outputs: the empirical 0-1 loss (\in[0,1])
    """
    loss_01 = 0
    for i in trange(input.shape[0]):
        loss_01_i = get_loss_01(pi, input[i : i + 1], target[i : i + 1], sample=sample)
        loss_01 += loss_01_i
    return loss_01.cpu().numpy()[0] / input.shape[0]


def mcsampling_excess(posterior, prior, input, target, sample_prior=True, gamma_t=0.5):
    """Compute the mean of excess loss delta_j^{\hat}(h_2, h_1, X, Y), where
       h_2 is sampled from posterior and h_1 is obtained from prior.
    Inputs:
        posterior    - posterior
        prior        - prior
        input        - data X
        target       - data Y
        sample_prior - (bool) : True  = sample an h_1 from prior for each data
                                False = use a fixed h_1 from prior for all data
        gamma_t      - the parameter for excess loss
    Output: Empirical excess loss over samples for each j
            with shape = len(js) and each value \in[0,1]
    """
    if gamma_t == 1:
        rv = np.array([-1, 0, 1])
    else:
        rv = np.array([-gamma_t, 0, 1 - gamma_t, 1])
    js = rv[1:]
    delta_js = torch.zeros(len(js))

    loss_prior = 0
    loss_posterior = 0

    for i in trange(input.shape[0]):
        loss_01_prior = get_loss_01(
            prior,
            input[i : i + 1],
            target[i : i + 1],
            sample=sample_prior,
        )
        loss_01_posterior = get_loss_01(
            posterior, input[i : i + 1], target[i : i + 1], sample=True
        )
        delta_js_mc = get_excess_j(loss_01_posterior, loss_01_prior, js, gamma_t)
        loss_prior += loss_01_prior
        loss_posterior += loss_01_posterior
        delta_js += delta_js_mc

    return delta_js.cpu().numpy() / input.shape[0], loss_prior.cpu().numpy() / input.shape[0], loss_posterior.cpu().numpy() / input.shape[0]


def compute_risk_rpb(
    posteriors, eval_loaders, gamma_t=0.5, delta_test=0.01, delta=0.025
):
    """Compute risk of T-step posteriors.
        Use B_1 and (E_t)_{t >= 2}.
    Inputs:
        posteriors      - the recursive posteriors
        eval_loaders    - evaluation datasets for the recursive posteriors
        gamma_t         - offset parameter of the excess loss
        delta_test      - for the test set bound of the posteriors
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = len(posteriors)
    loss_ts = []
    kl_ts = []
    E_ts = []
    B_ts = []
    for t in range(1, T + 1):
        # load posterior
        posterior = posteriors[t - 1]
        posterior.eval()
        # load eval data
        eval_loader = eval_loaders[t - 1]
        n_bound = len(eval_loader.sampler.indices)

        print("Current step:", t)
        print("n_bound:", n_bound)

        # compute kl between the posterior and its prior
        kl = posterior.compute_kl().detach().cpu().numpy()
        kl_ts.append(kl)

        if t == 1:
            # compute B_1 using 0-1 losses
            loss_01 = 0
            for _, (input, target) in enumerate(tqdm(eval_loader)):
                input, target = input.to(device), target.to(device)
                loss_01 += mcsampling_01(posterior, input, target) * input.shape[0]
            loss_01 /= n_bound

            B_1 = compute_B_1(loss_01, kl, T, n_bound, delta_test, delta)
            loss_ts.append(loss_01)
        else:
            # compute E_t using excess losses
            prior = posteriors[t - 2]
            prior.eval()

            loss_excess = 0
            for _, (input, target) in enumerate(tqdm(eval_loader)):
                input, target = input.to(device), target.to(device)
                emp_losses = mcsampling_excess(
                    posterior, prior, input, target, gamma_t=gamma_t
                )
                loss_excess += emp_losses[0] * input.shape[0]
            loss_excess /= n_bound

            E_t = compute_E_t(loss_excess, kl, T, gamma_t, n_bound, delta_test, delta)
            E_ts.append(E_t)
            loss_ts.append(loss_excess)

    # compute B_t recursively using B_1, (E_t)_t and gamma_t
    B_ts = compute_B_t(B_1, E_ts, gamma_t)
    return loss_ts, kl_ts, E_ts, B_ts


def compute_risk_rpb_recursive_step_1(
    posteriors, eval_loaders, gamma_t=0.5, delta_test=0.01, delta=0.025
):
    """Compute risk of T-step posteriors.
        Use B_0 and (E_t)_{t >= 1}.
    Inputs:
        posteriors      - the recursive posteriors
        eval_loaders    - evaluation datasets for the recursive posteriors
        gamma_t         - offset parameter of the excess loss
        delta_test      - for the test set bound of the posteriors
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = len(posteriors) - 1
    loss_ts = []
    kl_ts = []
    E_ts = []
    B_ts = []
    for t in range(1, T + 1):
        # load prior
        prior = posteriors[t - 1]
        prior.eval()
        # load posterior
        posterior = posteriors[t]
        posterior.eval()
        # load eval data
        eval_loader = eval_loaders[t - 1]
        n_bound = len(eval_loader.sampler.indices)

        print("Current step:", t)
        print("n_bound:", n_bound)

        # compute kl between the posterior and its prior
        kl = posterior.compute_kl().detach().cpu().numpy()
        kl_ts.append(kl)

        if t == 1:
            # compute B_0 using loss_01
            loss_01 = 0
            for _, (input, target) in enumerate(tqdm(eval_loader)):
                input, target = input.to(device), target.to(device)
                loss_01 += mcsampling_01(prior, input, target) * input.shape[0]
            loss_01 /= n_bound
            loss_ts.append(loss_01)
            B_0 = compute_B_1(loss_01, 0, T + 1, n_bound, delta_test, delta)

        # compute (E_t)_{t >= 1}
        loss_excess = 0

        for _, (input, target) in enumerate(tqdm(eval_loader)):
            input, target = input.to(device), target.to(device)
            emp_losses = mcsampling_excess(
                posterior, prior, input, target, gamma_t=gamma_t
            )
            loss_excess += emp_losses[0] * input.shape[0]
        loss_excess /= n_bound

        loss_ts.append(loss_excess)
        E_t = compute_E_t(loss_excess, kl, T + 1, gamma_t, n_bound, delta_test, delta)
        E_ts.append(E_t)

        """
        if t == 1:
            loss_01 = 0
            for _, (input, target) in enumerate(tqdm(eval_loader)):
                input, target = input.to(device), target.to(device)
                loss_01 += mcsampling_01(prior, input, target) * input.shape[0]
            loss_01 /= n_bound

            loss_excess = 0
            for _, (input, target) in enumerate(tqdm(eval_loader)):
                input, target = input.to(device), target.to(device)
                loss_excess += (
                    mcsampling_excess(posterior, prior, input, target, gamma_t=gamma_t)
                    * input.shape[0]
                )
            loss_excess /= n_bound

            B_0, E_1 = _compute_B_1_recursive_step_1(
                loss_01, loss_excess, kl, T, n_bound, gamma_t, delta_test, delta
            )
            E_ts.append(E_1)
            loss_ts.append([loss_01, loss_excess])
        else:
            loss_excess = 0
            for _, (input, target) in enumerate(tqdm(eval_loader)):
                input, target = input.to(device), target.to(device)
                loss_excess += (
                    mcsampling_excess(posterior, prior, input, target, gamma_t=gamma_t)
                    * input.shape[0]
                )
            loss_excess /= n_bound
            E_t = compute_E_t(loss_excess, kl, T, gamma_t, n_bound, delta_test, delta)
            E_ts.append(E_t)
            loss_ts.append(loss_excess)
        """

    B_ts = compute_B_t(B_0, E_ts, gamma_t)
    return loss_ts, kl_ts, E_ts, B_ts


def compute_risk_rpb_laststep(posterior, eval_loader, evaloss_post=None, delta_test=0.01, delta=0.025):
    """Bound \pi_t by \pi_{t-1} without further recursion
        \pi_{t_1} is only needed for computing kl, which is already recorded by posterior
    Inputs:
        posterior       - \pi_t
        delta_test      - for the test set bound of the posteriors
        eval_loader     - evaluation dataset
        evaloss_post    - evaluation loss of the posterior on eval_loader already computed somewhere else
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    posterior.eval()
    n_bound = len(eval_loader.sampler.indices)
    print("n_bound:", n_bound)

    kl = posterior.compute_kl().detach().cpu().numpy()

    if evaloss_post is not None:
        loss_01 = evaloss_post
    else:
        loss_01 = 0
        for _, (input, target) in enumerate(tqdm(eval_loader)):
            input, target = input.to(device), target.to(device)
            loss_01 += mcsampling_01(posterior, input, target) * input.shape[0]
        loss_01 /= n_bound

    inv_1 = solve_kl_sup(loss_01, np.log(1 / delta_test) / n_bound)
    B_T = solve_kl_sup(
        inv_1,
        (kl + np.log((2 * np.sqrt(n_bound)) / delta)) / n_bound,
    )
    return [loss_01], [kl], [B_T]


def compute_risk_rpb_onestep(
    posterior, prior, eval_loader, gamma_t, T, delta_test=0.01, delta=0.025
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior.eval()
    posterior.eval()

    kl = posterior.compute_kl().detach().cpu().numpy()
    n_bound = len(eval_loader.sampler.indices)

    print("n_bound:", n_bound)

    rv = np.array([-gamma_t, 0, 1 - gamma_t, 1])
    js_minus = rv[1:] - rv[0:-1]

    loss_excess = 0
    loss_prior = 0
    loss_posterior = 0
    for _, (input, target) in enumerate(tqdm(eval_loader)):
        input, target = input.to(device), target.to(device)
        emp_losses = mcsampling_excess(
            posterior, prior, input, target, gamma_t=gamma_t
        )
        loss_excess += emp_losses[0] * input.shape[0]
        loss_prior += emp_losses[1] * input.shape[0]
        loss_posterior += emp_losses[2] * input.shape[0]

    loss_excess /= n_bound
    loss_prior /= n_bound
    loss_posterior /= n_bound

    loss_excess_sum = (loss_excess * js_minus).sum(0) + rv[0]

    E_t = compute_E_t(loss_excess, kl, T, gamma_t, n_bound, delta_test, delta)

    # print some stats
    #print("n_bound: ", n_bound, "gamma_t: ", gamma_t)
    #print("posterior - gam * prior: ", loss_posterior - gamma_t * loss_prior, "; loss_excess_sum: ", loss_excess_sum)
    #print("E_t", E_t)
    return loss_excess, loss_excess_sum, E_t, kl, loss_prior, loss_posterior


def compute_B_t(B_1, E_ts, gamma_t):
    """Compute risk of T-step posteriors using the recursive formula:
    B_t = E_t + gam * B_{t-1}
    Inputs:
        B_1     - The first term B_1
        E_ts    - The excess losses until t-step (E_1,\cdots,E_t)
    Outputs:
        B_ts    - The bounds of (pi_1,\cdots,pi_t)
    """
    B_ts = [B_1]
    for i in range(len(E_ts)):
        B_t = B_ts[i] * gamma_t + E_ts[i]
        B_ts.append(B_t)
    return B_ts


def compute_B_1(loss_01, kl, T, n_bound, delta_test=0.01, delta=0.025):
    """Compute B_1."""
    inv_1 = solve_kl_sup(loss_01, np.log(T / delta_test) / n_bound)
    B_1 = solve_kl_sup(
        inv_1,
        (kl + np.log((2 * T * np.sqrt(n_bound)) / delta)) / n_bound,
    )
    return B_1


"""
def compute_B_1_recursive_step_1(
    loss_01, loss_excess, kl, T, n_bound, gamma_t, delta_test=0.01, delta=0.025
):

    inv_1 = solve_kl_sup(loss_01, np.log(T / delta_test) / n_bound)
    B_0 = solve_kl_sup(inv_1, np.log(T / delta) / n_bound)

    if gamma_t == 1:
        rv = np.array([-1, 0, 1])
    else:
        rv = np.array([-gamma_t, 0, 1 - gamma_t, 1])
    js_minus = rv[1:] - rv[0:-1]

    E_1 = rv[0]
    for i in range(len(loss_excess)):
        inv_1 = solve_kl_sup(
            loss_excess[i],
            np.log(len(js_minus) * T / delta_test) / n_bound,
        )
        inv_2 = solve_kl_sup(
            inv_1,
            (kl + np.log((len(js_minus) * T * 2 * np.sqrt(n_bound)) / delta)) / n_bound,
        )
        E_1 += inv_2 * js_minus[i]

    return B_0, E_1
"""


def compute_E_t(loss_excess, kl, T, gamma_t, n_bound, delta_test=0.01, delta=0.025):
    """Compute E_t."""

    if gamma_t == 1:
        rv = np.array([-1, 0, 1])
    else:
        rv = np.array([-gamma_t, 0, 1 - gamma_t, 1])
    js_minus = rv[1:] - rv[0:-1]

    E_t = rv[0]
    for i in range(len(loss_excess)):
        inv_1 = solve_kl_sup(
            loss_excess[i],
            np.log(len(js_minus) * T / delta_test) / n_bound,
        )
        inv_2 = solve_kl_sup(
            inv_1,
            (kl + np.log((len(js_minus) * T * 2 * np.sqrt(n_bound)) / delta)) / n_bound,
        )
        E_t += inv_2 * js_minus[i]
    return E_t


def compute_risk_informedexcess(
    posterior, h_pnet, eval_loader, delta_test=0.01, delta=0.025
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_bound = len(eval_loader.sampler.indices)

    h_pnet.eval()
    posterior.eval()
    kl = posterior.compute_kl().detach().cpu().numpy()

    loss_excess = 0
    loss_01 = 0
    for _, (input, target) in enumerate(tqdm(eval_loader)):
        input, target = input.to(device), target.to(device)
        emp_losses = mcsampling_excess(
            posterior, h_pnet, input, target, gamma_t=1.0
        )
        loss_excess += emp_losses[0] * input.shape[0]
        loss_01 += mcsampling_01(h_pnet, input, target, sample=False) * input.shape[0]
    loss_excess /= n_bound
    loss_01 /= n_bound

    rv = np.array([-1, 0, 1])
    js = rv[1:]
    js_minus = rv[1:] - rv[0:-1]

    risk = rv[0]
    for i in range(len(loss_excess)):
        inv_1 = solve_kl_sup(
            loss_excess[i],
            np.log(len(js) / delta_test) / n_bound,
        )
        inv_2 = solve_kl_sup(
            inv_1,
            (kl + np.log((len(js_minus) * 2 * np.sqrt(n_bound)) / delta)) / n_bound,
        )
        risk += inv_2 * js_minus[i]

    risk_h = compute_risk_h(loss_01, n_bound, delta)
    risk += risk_h

    return risk, risk_h, loss_excess, loss_01


def compute_risk_h(loss_01, n_bound, delta=0.025):
    risk_h = solve_kl_sup(loss_01, np.log(2 / delta) / n_bound)
    return risk_h


def KL(Q, P):
    """
    Compute Kullback-Leibler (KL) divergence between distributions Q and P.
    """
    return sum([q * log(q / p) if q > 0.0 else 0.0 for q, p in zip(Q, P)])


def KL_binomial(q, p):
    """
    Compute the KL-divergence between two Bernoulli distributions of probability
    of success q and p. That is, Q=(q,1-q), P=(p,1-p).
    """
    return KL([q, 1.0 - q], [p, 1.0 - p])


def get_binominal_inv(n, k, delta):
    for p in np.linspace(1, 0, 100001):
        if binom.pmf(k, n, p) >= delta:
            return p


def solve_kl_sup(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x > q
    """
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1.0 - 1e-9) <= 0.0:
        return 1.0 - 1e-9
    else:
        return optimize.brentq(f, q, 1.0 - 1e-11)
