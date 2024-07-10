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
        confidence value for the single hypothesis bound (used when computing the risk)

    kl_penalty : float (NOT USED, =1 in the work)
        penalty for the kl coefficient in the training objective

    device : string
        Device the code will run in (e.g. 'cuda')

    n_posterior : integer
        Number of future data for evaluation (n^val_t) in the paper

    use_excess_loss : bool
        Whether we use excess loss to learn the posterior at the current step

    sample_prior : bool
        Whether we sample a hypothesis from the prior for each sample

    c1 : float
        Parameter of sigmoid function to convexify the 0-1 loss

    c2 : float
        Parameter of softmax function to turn the output of the network to prediction
        
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
        c1=3,
        c2=3,
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
        self.c1 = c1
        self.c2 = c2

    def compute_negative_log_losses(self, outputs, targets, bounded=True, reduce=True):
        """ compute negative log likelihood loss and bound it with pmin (if applicable)
            outputs : real-valued vector
                output of the network (R^K)
            bounded : bool
                return a bounded loss (by clamping) or not
            reduce : bool
                True = return the average loss over samples
                False = return losses for all samples
        """
        logprobs = F.log_softmax(self.c2 * outputs, dim=1)  # log probability with softmax parameter c2
        if bounded == True:
            logprobs = torch.clamp(logprobs, np.log(self.pmin)) # lower-bounding the probability
            empirical_risk = F.nll_loss(logprobs, targets, reduce=reduce) # compute the negative log loss
            #empirical_risk = (1.0 / (np.log(1.0 / self.pmin))) * empirical_risk # DO NOT RESCALE
        else:
            empirical_risk = F.cross_entropy(logprobs, targets, reduce=reduce) # compute the cross-entropy loss
        return empirical_risk

    def compute_losses(self, net, input, target, bounded=True, prior=None, gamma_t=0.5):
        # compute both cross-entropy loss, 01 loss, and excess loss
        # returns outputs of the network as well

        outputs = net(input, sample=True) # output of the network (R^K)

        # compute the cross-entropy loss
        loss_ce = self.compute_negative_log_losses(outputs, target, bounded, reduce=True)
        #loss_ce = (1.0 / (np.log(1.0 / self.pmin))) * loss_ce

        # compute the 01 loss
        pred = outputs.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = target.size(0)
        loss_01 = 1 - (correct / total)

        if self.use_excess_loss:
            # compute the excess loss

            prior.eval()

            outputs_prior = prior(input, sample=self.sample_prior)
            #logprobs_prior = F.log_softmax(self.c2 * outputs_prior, dim=1)
            #logprobs_prior = torch.clamp(logprobs_prior, np.log(self.pmin))
            #loss_ce_excess_prior = F.cross_entropy(
            #    logprobs_prior, target, reduce=False
            #)
            loss_ce_excess_prior = self.compute_negative_log_losses(outputs_prior, target, bounded, reduce=False)
            #print("ce_prior : ", loss_ce_excess_prior[:10])
            #pred_prior = outputs_prior.max(1, keepdim=True)[1]
            #correct_prior = pred_prior.eq(target.view_as(pred_prior)).long()
            #correct_prior = torch.squeeze(correct_prior, 1)
            #loss_01_prior = 1 - correct_prior
            #print("loss_01_prior : ", loss_01_prior[:10])
            ##outputs = F.log_softmax(c2 * outputs, dim=1)
            ##outputs = torch.clamp(outputs, np.log(pmin))
            #loss_ce_excess_posterior = F.cross_entropy(
            #    outputs, target, reduce=False
            #)
            loss_ce_excess_posterior = self.compute_negative_log_losses(outputs, target, bounded, reduce=False)
            #print("ce_posterior : ", loss_ce_excess_posterior[:10])
            #pred_posterior = outputs.max(1, keepdim=True)[1]
            #correct_posterior = pred_posterior.eq(target.view_as(pred_posterior)).long()
            #correct_posterior = torch.squeeze(correct_posterior, 1)
            #loss_01_posterior = 1 - correct_posterior
            #print("loss_01_posterior : ", loss_01_posterior[:10])
            loss_ce_excess = loss_ce_excess_posterior - loss_ce_excess_prior * gamma_t
            #print("ce_excess_loss : ", loss_ce_excess[:10])
            #loss_01_excess = loss_01_posterior - loss_01_prior * gamma_t
            #print("01_excess_loss : ", loss_01_excess[:10])
            #print("01_excess_diff : ", loss_ce_excess[:10] - loss_01_excess[:10])
            #print("-------------------")

            loss_excess = []
            if gamma_t == 1:
                rv = np.array([-1, 0, 1])
            else:
                rv = np.array([-gamma_t, 0, 1 - gamma_t, 1])
            js = rv[1:]
            for j in js:
                # convexification of the excess loss by sigmoid
                loss_excess.append(F.sigmoid(self.c1 * (loss_ce_excess - j)).mean())
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

    def train_obj(self, net, input, target, clamping=True, lambda_var=None, prior=None, gamma_t=0.5):
        """ Compute train objective and return all metrics

            Parameters
            ----------
            net : NNet/CNNet object
                Network object to train

            input : tensor
                input feature

            target : tensor
                the label of the input

            clamping : bool
                whether to clamp the output probabilities

            lambda_var : Lambda_var object
                Lambda variable for training objective flamb

            prior : NNet/CNNet object
                The prior

            gamma_t : in [0,1]
                The offset parameter for recursive PB        
        """
        outputs = torch.zeros(target.size(0), self.classes).to(self.device)
        kl = net.compute_kl()
        loss_ce, loss_01, outputs, loss_excess = self.compute_losses(
            net, input, target, clamping, prior, gamma_t
        )
        if self.use_excess_loss:
            train_obj = self.bound(loss_excess, kl, self.n_posterior, lambda_var, gamma_t)
        else:
            train_obj = self.bound(loss_ce, kl, self.n_posterior, lambda_var, gamma_t)
        return train_obj, kl / self.n_posterior, outputs, loss_ce, loss_01
