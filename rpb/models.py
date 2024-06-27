import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used works best if :math:`\text{mean}` is
    near the center of the interval.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1.0 - eps), max=(1.0 - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class Gaussian(nn.Module):
    """Implementation of a Gaussian random variable, using softplus for
    the standard deviation and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Gaussian.

    rho : Tensor of floats
        Scale parameter of the Gaussian (to be transformed to std
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the Gaussian is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device="cuda", fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div


class Linear(nn.Module):
    """Implementation of a Linear layer (reimplemented to use
    truncated normal as initialisation for fair comparison purposes)

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, in_features, out_features, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1 / np.sqrt(in_features)

        # same initialisation as before for the prob layer
        self.weight = nn.Parameter(
            trunc_normal_(
                torch.Tensor(out_features, in_features),
                0,
                sigma_weights,
                -2 * sigma_weights,
                2 * sigma_weights,
            ),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        return F.linear(input, weight, bias)


class Lambda_var(nn.Module):
    """Class for the lambda variable included in the objective
    flambda

    Parameters
    ----------
    lamb : float
        initial value

    n : int
        Scaling parameter (lamb_scaled is between 1/sqrt(n) and 1)

    """

    def __init__(self, lamb, n):
        super().__init__()
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=True)
        self.min = 1 / np.sqrt(n)

    @property
    def lamb_scaled(self):
        # We restrict lamb_scaled to be between 1/sqrt(n) and 1.
        m = nn.Sigmoid()
        return m(self.lamb) * (1 - self.min) + self.min


class ProbLinear(nn.Module):
    """Implementation of a Probabilistic Linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_layer : Linear object
        Linear layer object used to initialise the prior

    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """

    def __init__(
        self,
        in_features,
        out_features,
        rho_prior,
        prior_dist="gaussian",
        device="cuda",
        init_prior="weights",
        init_layer=None,
        init_player=None,
        init_layer_prior=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1 / np.sqrt(in_features)

        if init_player:
            weights_mu_init = init_player.weight.mu
            bias_mu_init = init_player.bias.mu
            weights_rho_init = init_player.weight.rho
            bias_rho_init = init_player.bias.rho
        elif init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
            weights_rho_init = torch.ones(out_features, in_features) * rho_prior
            bias_rho_init = torch.ones(out_features) * rho_prior
        else:
            # Initialise distribution means using truncated normal
            weights_mu_init = trunc_normal_(
                torch.Tensor(out_features, in_features),
                0,
                sigma_weights,
                -2 * sigma_weights,
                2 * sigma_weights,
            )
            bias_mu_init = torch.zeros(out_features)
            weights_rho_init = torch.ones(out_features, in_features) * rho_prior
            bias_rho_init = torch.ones(out_features) * rho_prior

        if init_prior == "zeros":
            bias_mu_prior = torch.zeros(out_features)
            weights_mu_prior = torch.zeros(out_features, in_features)
        elif init_prior == "random":
            weights_mu_prior = trunc_normal_(
                torch.Tensor(out_features, in_features),
                0,
                sigma_weights,
                -2 * sigma_weights,
                2 * sigma_weights,
            )
            bias_mu_prior = torch.zeros(out_features)
        elif init_prior == "weights":
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                # otherwise initialise to posterior weights
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
        else:
            raise RuntimeError(f"Wrong type of prior initialisation!")

        if prior_dist == "gaussian":
            dist = Gaussian
        else:
            raise RuntimeError(f"Wrong prior_dist {prior_dist}")

        self.weight = dist(
            weights_mu_init.clone(),
            weights_rho_init.clone(),
            device=device,
            fixed=False,
        )

        self.bias = dist(
            bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=False
        )

        self.weight_prior = dist(
            weights_mu_prior.clone(),
            weights_rho_init.clone(),
            device=device,
            fixed=True,
        )
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True
        )

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior
            ) + self.bias.compute_kl(self.bias_prior)

        return F.linear(input, weight, bias)


class ProbConv2d(nn.Module):
    """Implementation of a Probabilistic Convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the layer

    out_channels : int
        Number of output channels for the layer

    kernel_size : int
        size of the convolutional kernel

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    stride : int
        Stride of the convolution

    padding: int
        Zero-padding added to both sides of the input

    dilation: int
        Spacing between kernel elements

    init_layer : Linear object
        Linear layer object used to initialise the prior

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        rho_prior,
        prior_dist="gaussian",
        device="cuda",
        stride=1,
        padding=0,
        dilation=1,
        init_prior="weights",
        init_layer=None,
        init_player=None,
        init_layer_prior=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1

        # Compute and set sigma for the truncated gaussian of weights
        in_features = self.in_channels
        for k in self.kernel_size:
            in_features *= k
        sigma_weights = 1 / np.sqrt(in_features)

        if init_player:
            weights_mu_init = init_player.weight.mu
            bias_mu_init = init_player.bias.mu
            weights_rho_init = init_player.weight.rho
            bias_rho_init = init_player.bias.rho
        elif init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
            # set scale parameters
            weights_rho_init = (
                torch.ones(out_channels, in_channels, *self.kernel_size) * rho_prior
            )
            bias_rho_init = torch.ones(out_channels) * rho_prior
        else:
            weights_mu_init = trunc_normal_(
                torch.Tensor(out_channels, in_channels, *self.kernel_size),
                0,
                sigma_weights,
                -2 * sigma_weights,
                2 * sigma_weights,
            )
            bias_mu_init = torch.zeros(out_channels)
            # set scale parameters
            weights_rho_init = (
                torch.ones(out_channels, in_channels, *self.kernel_size) * rho_prior
            )
            bias_rho_init = torch.ones(out_channels) * rho_prior

        if init_prior == "zeros":
            bias_mu_prior = torch.zeros(out_features)
            weights_mu_prior = torch.zeros(out_features, in_features)
        elif init_prior == "random":
            weights_mu_prior = trunc_normal_(
                torch.Tensor(out_channels, in_channels, *self.kernel_size),
                0,
                sigma_weights,
                -2 * sigma_weights,
                2 * sigma_weights,
            )
            bias_mu_prior = torch.zeros(out_features)
        elif init_prior == "weights":
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                # otherwise initialise to posterior weights
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
        else:
            raise RuntimeError(f"Wrong type of prior initialisation!")

        if prior_dist == "gaussian":
            dist = Gaussian
        else:
            raise RuntimeError(f"Wrong prior_dist {prior_dist}")

        self.weight = dist(
            weights_mu_init.clone(),
            weights_rho_init.clone(),
            device=device,
            fixed=False,
        )
        self.bias = dist(
            bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=False
        )
        self.weight_prior = dist(
            weights_mu_prior.clone(),
            weights_rho_init.clone(),
            device=device,
            fixed=True,
        )
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True
        )

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior
            ) + self.bias.compute_kl(self.bias_prior)

        return F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )


class NNet4l(nn.Module):
    """Implementation of a standard Neural Network with 4 layers and dropout
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, dropout_prob=0.0, device="cuda"):
        super().__init__()
        self.l1 = Linear(28 * 28, 600, device)
        self.l2 = Linear(600, 600, device)
        self.l3 = Linear(600, 600, device)
        self.l4 = Linear(600, 10, device)
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x):
        # forward pass for the network
        x = x.view(-1, 28 * 28)
        x = self.d(self.l1(x))
        x = F.relu(x)
        x = self.d(self.l2(x))
        x = F.relu(x)
        x = self.d(self.l3(x))
        x = F.relu(x)
        x = self.l4(x)
        return x


class CNNet4l(nn.Module):
    """Implementation of a standard Convolutional Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.d = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        x = self.d(self.conv1(x))
        x = F.relu(x)
        x = self.d(self.conv2(x))
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.d(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        #return output
        return x


class ProbNNet4l(nn.Module):
    """Implementation of a Probabilistic Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : NNet object
        Network object used to initialise the prior

    init_pnet : PNNet object
        Probabilistic Network object used to initialise the prior

    """

    def __init__(
        self,
        rho_prior,
        prior_dist="gaussian",
        device="cuda",
        init_net=None,
        init_pnet=None,
    ):
        super().__init__()
        self.l1 = ProbLinear(
            28 * 28,
            600,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.l1 if init_net else None,
            init_player=init_pnet.l1 if init_pnet else None,
        )
        self.l2 = ProbLinear(
            600,
            600,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.l2 if init_net else None,
            init_player=init_pnet.l2 if init_pnet else None,
        )
        self.l3 = ProbLinear(
            600,
            600,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.l3 if init_net else None,
            init_player=init_pnet.l3 if init_pnet else None,
        )
        self.l4 = ProbLinear(
            600,
            10,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.l4 if init_net else None,
            init_player=init_pnet.l4 if init_pnet else None,
        )

    def forward(self, x, sample=False):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        x = self.l4(x)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div + self.l4.kl_div


class ProbCNNet4l(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size,
    number of classes and kernel size).

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : CNNet object
        Network object used to initialise the prior

    init_pnet : ProbCNNet object
        Probabilistic Network object used to initialise the prior

    """

    def __init__(
        self,
        rho_prior,
        prior_dist="gaussian",
        device="cuda",
        init_net=None,
        init_pnet=None,
    ):
        super().__init__()

        self.conv1 = ProbConv2d(
            1,
            32,
            3,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.conv1 if init_net else None,
            init_player=init_pnet.conv1 if init_pnet else None,
        )
        self.conv2 = ProbConv2d(
            32,
            64,
            3,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.conv2 if init_net else None,
            init_player=init_pnet.conv2 if init_pnet else None,
        )
        self.fc1 = ProbLinear(
            9216,
            128,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.fc1 if init_net else None,
            init_player=init_pnet.fc1 if init_pnet else None,
        )
        self.fc2 = ProbLinear(
            128,
            10,
            rho_prior,
            prior_dist=prior_dist,
            device=device,
            init_layer=init_net.fc2 if init_net else None,
            init_player=init_pnet.fc2 if init_pnet else None,
        )

    def forward(self, x, sample=False):
        # forward pass for the network
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x, sample))
        x = self.fc2(x, sample)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.fc1.kl_div + self.fc2.kl_div

# NOT USED
def output_logprobability(x, c2=3):
    output = F.log_softmax(c2 * x, dim=1)
    return output

# NOT USED
def output_transform(x, clamping=False, pmin=1e-4):
    """Computes the log softmax and clamps the values using the
    min probability given by pmin.

    Parameters
    ----------
    x : tensor
        output of the network

    clamping : bool
        whether to clamp the output probabilities

    pmin : float
        threshold of probabilities to clamp.
    """
    # lower bound output prob
    output = F.log_softmax(x, dim=1)
    if clamping:
        output = torch.clamp(output, np.log(pmin))
    return output


def trainNNet(net, optimizer, epoch, train_loader, device="cuda", verbose=False):
    """Train function for a standard NN (including CNN)

    Parameters
    ----------
    net : NNet/CNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    epoch : int
        Current training epoch

    train_loader: DataLoader object
        Train loader to use for training

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print training metrics

    """
    # train and report training metrics
    net.train()
    total, correct, avgloss = 0.0, 0.0, 0.0
    for batch_id, (input, target) in enumerate(tqdm(train_loader)):
        input, target = input.to(device), target.to(device)
        net.zero_grad()
        output = net(input)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        avgloss = avgloss + loss.detach()
    # show the average loss and KL during the epoch
    if verbose:
        print(
            f"-Epoch {epoch :.5f}, Train loss: {avgloss/batch_id :.5f}, Train err:  {1-(correct/total):.5f}"
        )


def trainPNNet(
    net,
    optimizer,
    pbobj,
    epoch,
    train_loader,
    lambda_var=None,
    optimizer_lambda=None,
    verbose=False,
    prior=None,
    gamma_t=0.5,
):
    """Train function for a probabilistic NN (including CNN)

    Parameters
    ----------
    net : ProbNNet/ProbCNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    pbobj : pbobj object
        PAC-Bayes inspired training objective to use for training

    epoch : int
        Current training epoch

    train_loader: DataLoader object
        Train loader to use for training

    lambda_var : Lambda_var object
        Lambda variable for training objective flamb

    optimizer_lambda : optim object
        Optimizer to use for the learning the lambda_variable

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print test metrics

    prior : ProbNNet/ProbCNNet object
        Network object that serves as prior

    gamma_t : in [0,1]
        The offset parameter for recursive PB

    """
    net.train()
    # variables that keep information about the results of optimising the bound
    avgerr, avgbound, avgkl, avgloss = 0.0, 0.0, 0.0, 0.0

    if pbobj.objective == "flamb":
        lambda_var.train()
        # variables that keep information about the results of optimising lambda (only for flamb)
        avgerr_l, avgbound_l, avgkl_l, avgloss_l = 0.0, 0.0, 0.0, 0.0

    if pbobj.objective == "bbb":
        clamping = False
    else:
        # lower-bounding the probability assigned to Y
        # to give a bounded cross-entropy loss for the training objective
        clamping = True

    for batch_id, (input, target) in enumerate(tqdm(train_loader)):
        input, target = input.to(pbobj.device), target.to(pbobj.device)
        net.zero_grad()
        bound, kl, _, loss, err = pbobj.train_obj(
            net,
            input,
            target,
            clamping=clamping,
            lambda_var=lambda_var,
            prior=prior,
            gamma_t=gamma_t,
        )

        bound.backward()
        optimizer.step()
        avgbound += bound.item()
        avgkl += kl
        avgloss += loss.item()
        avgerr += err

        if pbobj.objective == "flamb":
            # for flamb we also need to optimise the lambda variable
            lambda_var.zero_grad()
            bound_l, kl_l, _, loss_l, err_l = pbobj.train_obj(
                net, input, target, clamping=clamping, lambda_var=lambda_var, prior=prior, gamma_t = gamma_t
            )
            bound_l.backward()
            optimizer_lambda.step()
            avgbound_l += bound_l.item()
            avgkl_l += kl_l
            avgloss_l += loss_l.item()
            avgerr_l += err_l

    if verbose:
        # show the average of the metrics during the epoch
        print(
            f"-Batch average epoch {epoch :.0f} results, Train obj: {avgbound/batch_id :.5f}, KL/n: {avgkl/batch_id :.5f}, NLL loss: {avgloss/batch_id :.5f}, Train 0-1 Error:  {avgerr/batch_id :.5f}"
        )
        if pbobj.objective == "flamb":
            print(
                f"-After optimising lambda: Train obj: {avgbound_l/batch_id :.5f}, KL/n: {avgkl_l/batch_id :.5f}, NLL loss: {avgloss_l/batch_id :.5f}, Train 0-1 Error:  {avgerr_l/batch_id :.5f}, last lambda value: {lambda_var.lamb_scaled.item() :.5f}"
            )
