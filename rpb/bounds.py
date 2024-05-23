import math
import numpy as np
import torch
import torch.distributions as td
from tqdm import tqdm, trange
import torch.nn.functional as F
from rpb.eval import get_loss_01


class PBBobj:
    """Class including all functionalities needed to train a NN with a PAC-Bayes inspired
    training objective and evaluate the risk certificate at the end of training.

    Parameters
    ----------
    objective : string
        training objective to be optimised (choices are fquad, flamb, fclassic or fbbb)

    pmin : float
        minimum probability to clamp to have a loss in [0,1]

    classes : int
        number of classes in the learning problem

    train_size : int
        n (number of training examples)

    delta : float
        confidence value for the training objective

    delta_test : float
        confidence value for the chernoff bound (used when computing the risk)

    kl_penalty : float
        penalty for the kl coefficient in the training objective

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(
        self,
        objective="fclassic",
        pmin=1e-5,
        classes=10,
        delta=0.025,
        delta_test=0.01,
        kl_penalty=1,
        device="cuda",
        n_posterior=30000,
        use_excess_loss=False,
        sample_prior=True,
    ):
        super().__init__()
        self.objective = objective
        self.pmin = pmin
        self.classes = classes
        self.device = device
        self.delta = delta
        self.delta_test = delta_test
        self.kl_penalty = kl_penalty
        self.n_posterior = n_posterior
        self.use_excess_loss = use_excess_loss
        self.sample_prior = sample_prior

    def compute_empirical_risk(self, outputs, targets, bounded=True):
        # compute negative log likelihood loss and bound it with pmin (if applicable)
        # empirical_risk = F.nll_loss(outputs, targets)
        c1 = 3
        empirical_risk = F.cross_entropy(outputs * c1, targets)

        if bounded == True:
            empirical_risk = (1.0 / (np.log(1.0 / self.pmin))) * empirical_risk
        return empirical_risk

    def compute_losses(
        self,
        net,
        data,
        target,
        clamping=True,
        prior=None,
        gamma_t=0.5,
    ):
        # compute both cross entropy and 01 loss
        # returns outputs of the network as well

        outputs = net(data, sample=True, clamping=clamping, pmin=self.pmin)
        loss_ce = self.compute_empirical_risk(outputs, target, clamping)
        pred = outputs.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = target.size(0)
        loss_01 = 1 - (correct / total)

        if self.use_excess_loss:

            prior.eval()

            c1 = 3
            c2 = 3

            outputs_prior = prior(
                data, sample=self.sample_prior, clamping=clamping, pmin=self.pmin
            )
            loss_ce_excess_prior = F.cross_entropy(
                outputs_prior * c1, target, reduce=False
            )
            loss_ce_excess_posterior = F.cross_entropy(
                outputs * c2, target, reduce=False
            )
            loss_ce_excess = loss_ce_excess_posterior - loss_ce_excess_prior * gamma_t

            loss_excess = []
            if gamma_t == 1:
                rv = np.array([-1, 0, 1])
            else:
                rv = np.array([-gamma_t, 0, 1 - gamma_t, 1])
            js = rv[1:]
            for j in js:
                loss_excess.append(F.sigmoid(c1 * (loss_ce_excess - j)).mean())
        else:
            loss_excess = None
        return loss_ce, loss_01, outputs, loss_excess

    def bound(self, empirical_risk, kl, train_size, lambda_var=None, gamma_t=0.5):
        # compute training objectives
        if not self.use_excess_loss:
            if self.objective == "fquad":
                kl = kl * self.kl_penalty
                repeated_kl_ratio = torch.div(
                    kl + np.log((2 * np.sqrt(train_size)) / self.delta), 2 * train_size
                )
                first_term = torch.sqrt(empirical_risk + repeated_kl_ratio)
                second_term = torch.sqrt(repeated_kl_ratio)
                train_obj = torch.pow(first_term + second_term, 2)
            elif self.objective == "flamb":
                kl = kl * self.kl_penalty
                lamb = lambda_var.lamb_scaled
                kl_term = torch.div(
                    kl + np.log((2 * np.sqrt(train_size)) / self.delta),
                    train_size * lamb * (1 - lamb / 2),
                )
                first_term = torch.div(empirical_risk, 1 - lamb / 2)
                train_obj = first_term + kl_term
            elif self.objective == "fclassic":
                kl = kl * self.kl_penalty
                kl_ratio = torch.div(
                    kl + np.log((2 * np.sqrt(train_size)) / self.delta), 2 * train_size
                )
                train_obj = empirical_risk + torch.sqrt(kl_ratio)
            elif self.objective == "bbb":
                # ipdb.set_trace()
                train_obj = empirical_risk + self.kl_penalty * (kl / train_size)
            else:
                raise RuntimeError(f"Wrong objective {self.objective}")
        else:
            train_obj_total = 0

            if gamma_t == 1:
                rv = np.array([-1, 0, 1])
            else:
                rv = np.array([-gamma_t, 0, 1 - gamma_t, 1])
            js_minus = rv[1:] - rv[0:-1]

            for j, risk_term in zip(js_minus, empirical_risk):
                if self.objective == "fquad":
                    kl = kl * self.kl_penalty
                    repeated_kl_ratio = torch.div(
                        kl + np.log((2 * np.sqrt(train_size)) / self.delta),
                        2 * train_size,
                    )
                    first_term = torch.sqrt(risk_term + repeated_kl_ratio)
                    second_term = torch.sqrt(repeated_kl_ratio)
                    train_obj = torch.pow(first_term + second_term, 2)
                elif self.objective == "flamb":
                    kl = kl * self.kl_penalty
                    lamb = lambda_var.lamb_scaled
                    kl_term = torch.div(
                        kl + np.log((2 * np.sqrt(train_size)) / self.delta),
                        train_size * lamb * (1 - lamb / 2),
                    )
                    first_term = torch.div(risk_term, 1 - lamb / 2)
                    train_obj = first_term + kl_term
                elif self.objective == "fclassic":
                    kl = kl * self.kl_penalty
                    kl_ratio = torch.div(
                        kl + np.log((2 * np.sqrt(train_size)) / self.delta),
                        2 * train_size,
                    )
                    train_obj = risk_term + torch.sqrt(kl_ratio)
                elif self.objective == "bbb":
                    train_obj = risk_term + self.kl_penalty * (kl / train_size)
                else:
                    raise RuntimeError(f"Wrong objective {self.objective}")
                train_obj_total += train_obj * j
            train_obj = rv[0] + train_obj_total
        return train_obj

    def train_obj(
        self,
        net,
        input,
        target,
        clamping=True,
        lambda_var=None,
        prior=None,
        gamma_t=0.5,
    ):
        # compute train objective and return all metrics
        outputs = torch.zeros(target.size(0), self.classes).to(self.device)
        kl = net.compute_kl()
        loss_ce, loss_01, outputs, loss_excess = self.compute_losses(
            net, input, target, clamping, prior, gamma_t
        )
        if self.use_excess_loss:
            train_obj = self.bound(
                loss_excess, kl, self.n_posterior, lambda_var, gamma_t
            )
        else:
            train_obj = self.bound(loss_ce, kl, self.n_posterior, lambda_var, gamma_t)
        return train_obj, kl / self.n_posterior, outputs, loss_ce, loss_01
