import math
import torch
from rpb.models import (
    NNet4l,
    CNNet4l,
    ProbNNet4l,
    ProbCNNet4l,
)


def init_posterior(model, sigma_prior, prior, device):
    """Given a prior, initialize a posterior."""

    rho_prior = math.log(math.exp(sigma_prior) - 1.0)

    if model == "cnn":
        if not prior:
            posterior_mean = CNNet4l(dropout_prob=0.0).to(device)
        else:
            posterior_mean = None
        posterior = ProbCNNet4l(
            rho_prior,
            prior_dist="gaussian",
            device=device,
            init_net=posterior_mean,
            init_pnet=prior,
        ).to(device)
    elif model == "fcn":
        if not prior:
            posterior_mean = NNet4l(dropout_prob=0.0).to(device)
        else:
            posterior_mean = None
        posterior = ProbNNet4l(
            rho_prior,
            prior_dist="gaussian",
            device=device,
            init_net=posterior_mean,
            init_pnet=prior,
        ).to(device)
    else:
        raise RuntimeError(f"Architecture {model} not supported")

    return posterior


def init_posterior_mean(model, sigma_prior, posterior_mean, device):

    rho_prior = math.log(math.exp(sigma_prior) - 1.0)

    if model == "cnn":
        posterior = ProbCNNet4l(
            rho_prior,
            prior_dist="gaussian",
            device=device,
            init_net=posterior_mean,
        ).to(device)
    elif model == "fcn":
        posterior = ProbNNet4l(
            rho_prior,
            prior_dist="gaussian",
            device=device,
            init_net=posterior_mean,
        ).to(device)
    else:
        raise RuntimeError(f"Architecture {model} not supported")

    return posterior


def get_mu(posterior):
    return torch.cat(
        [param.view(-1) for param in posterior.parameters() if param.requires_grad]
    )


def get_mu_pnet(posterior):
    return torch.cat(
        [
            param.detach().view(-1)
            for name, param in posterior.named_parameters()
            if param.requires_grad and ("weight.mu" in name or "bias.mu" in name)
        ]
    )


def get_sigma_pnet(posterior):
    rho = torch.cat(
        [
            param.detach().view(-1)
            for name, param in posterior.named_parameters()
            if param.requires_grad and ("weight.rho" in name or "bias.rho" in name)
        ]
    )
    return torch.log(torch.exp(rho) + 1)


def get_kl_q_p(mu_q, sigma_q, mu_p, sigma_p):
    q = torch.distributions.normal.Normal(mu_q, sigma_q)
    p = torch.distributions.normal.Normal(mu_p, sigma_p)
    return torch.distributions.kl.kl_divergence(q, p).sum().item()
