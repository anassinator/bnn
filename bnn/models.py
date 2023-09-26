"""Bayesian neural network models."""

import torch
import inspect

from functools import partial
from torch.nn import Parameter
from collections import OrderedDict
from collections.abc import Iterable


class BDropout(torch.nn.Dropout):

    """Binary dropout with regularization and resampling.

    See: Gal Y., Ghahramani Z., "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning", 2016.
    """

    def __init__(self, rate=0.1, reg=1.0, **kwargs):
        """Constructs a BDropout.

        Args:
            rate (float): Dropout probability.
            reg (float): Regularization scale.
        """
        super(BDropout, self).__init__(**kwargs)
        self.register_buffer("rate", torch.tensor(rate))
        self.p = 1 - self.rate
        self.register_buffer("reg", torch.tensor(reg))
        self.register_buffer("noise", torch.bernoulli(self.p))

    def regularization(self, weight, bias):
        """Computes the regularization cost.

        Args:
            weight (Tensor): Weight tensor.
            bias (Tensor): Bias tensor.

        Returns:
            Regularization cost (Tensor).
        """
        self.p = 1 - self.rate
        weight_reg = self.p * (weight**2).sum()
        bias_reg = (bias**2).sum() if bias is not None else 0
        return self.reg * (weight_reg + bias_reg)

    def resample(self):
        """Resamples the dropout noise."""
        self._update_noise(self.noise)

    def _update_noise(self, x):
        """Updates the dropout noise.

        Args:
            x (Tensor): Input.
        """
        self.p = 1 - self.rate
        self.noise.data = torch.bernoulli(self.p.expand(x.shape))

    def forward(self, x, resample=False, mask_dims=0, **kwargs):
        """Computes the binary dropout.

        Args:
            x (Tensor): Input.
            resample (bool): Whether to force resample.
            mask_dims (int): Number of dimensions to sample noise for
                (0 for all).

        Returns:
            Output (Tensor).
        """
        sample_shape = x.shape[-mask_dims:]
        if sample_shape != self.noise.shape:
            sample = x.view(-1, *sample_shape)[0]
            self._update_noise(sample)
        elif resample:
            return x * torch.bernoulli(self.p.expand(x.shape))

        return x * self.noise

    def extra_repr(self):
        """Formats module representation.

        Returns:
            Module representation (str).
        """
        return "rate={}".format(self.rate)


class CDropout(BDropout):

    """Concrete dropout with regularization and resampling.

    See: Gal Y., Hron, J., Kendall, A. "Concrete Dropout", 2017.
    """

    def __init__(self, temperature=0.1, rate=0.5, reg=1.0, **kwargs):
        """Constructs a CDropout.

        Args:
            temperature (float): Temperature.
            rate (float): Initial dropout rate.
            reg (float): Regularization scale.
        """
        super(CDropout, self).__init__(rate, reg, **kwargs)
        self.temperature = Parameter(
            torch.tensor(temperature), requires_grad=False)

        # We need to constrain p to [0, 1], so we train logit(p).
        self.logit_p = Parameter(-torch.log(self.p.reciprocal() - 1.0))

    def regularization(self, weight, bias):
        """Computes the regularization cost.

        Args:
            weight (Tensor): Weight tensor.
            bias (Tensor): Bias tensor.

        Returns:
            Regularization cost (Tensor).
        """
        self.p.data = self.logit_p.sigmoid()
        reg = super(CDropout, self).regularization(weight, bias)
        reg -= -(1 - self.p) * (1 - self.p).log() - self.p * self.p.log()
        return reg

    def _update_noise(self, x):
        """Updates the dropout noise.

        Args:
            x (Tensor): Input.
        """
        self.noise.data = torch.rand_like(x)

    def forward(self, x, resample=False, mask_dims=0, **kwargs):
        """Computes the concrete dropout.

        Args:
            x (Tensor): Input.
            resample (bool): Whether to force resample.
            mask_dims (int): Number of dimensions to sample noise for
                (0 for all).

        Returns:
            Output (Tensor).
        """
        sample_shape = x.shape[-mask_dims:]
        noise = self.noise
        if sample_shape != noise.shape:
            sample = x.view(-1, *sample_shape)[0]
            self._update_noise(sample)
            noise = self.noise
        elif resample:
            noise = torch.rand_like(x)

        self.p.data = self.logit_p.sigmoid()
        concrete_p = self.logit_p + noise.log() - (1 - noise).log()
        concrete_noise = (concrete_p / self.temperature).sigmoid()

        return x * concrete_noise

    def extra_repr(self):
        """Formats module representation.

        Returns:
            Module representation (str).
        """
        return "temperature={}".format(self.temperature)


class BSequential(torch.nn.Sequential):

    """Extension of Sequential module with regularization and resampling."""

    def resample(self):
        """Resample all child modules."""
        for child in self.children():
            if isinstance(child, BDropout):
                child.resample()

    def regularization(self):
        """Computes the total regularization cost of all child modules.

        Returns:
            Total regularization cost (Tensor).
        """
        reg = torch.tensor(0.0)
        children = list(self._modules.values())
        for i, child in enumerate(children):
            if isinstance(child, BSequential):
                reg += child.regularization()
            elif isinstance(child, BDropout):
                for next_child in children[i:]:
                    if hasattr(next_child, "weight") and hasattr(
                            next_child, "bias"):
                        reg += child.regularization(next_child.weight,
                                                    next_child.bias)
                        break
        return reg

    def forward(self, x, resample=False, **kwargs):
        """Computes the model.

        Args:
            x (Tensor): Input.
            resample (bool): Whether to force resample.

        Returns:
            Output (Tensor).
        """
        for module in self._modules.values():
            if isinstance(module, (BDropout, BSequential)):
                x = module(x, resample=resample, **kwargs)
            else:
                x = module(x)
        return x


def bayesian_model(in_features,
                   out_features,
                   hidden_features,
                   nonlin=torch.nn.ReLU,
                   output_nonlin=None,
                   weight_initializer=partial(
                       torch.nn.init.xavier_normal_,
                       gain=torch.nn.init.calculate_gain("relu")),
                   bias_initializer=partial(
                       torch.nn.init.uniform_, a=-1.0, b=1.0),
                   dropout_layers=CDropout,
                   input_dropout=None):
    """Constructs and initializes a Bayesian neural network with dropout.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        hidden_features (list<int>): Ordered list of hidden dimensions.
        nonlin (Module): Activation function for all hidden layers.
        output_nonlin (Module): Activation function for output layer.
        weight_initializer (callable): Function to initialize all module
            weights to pass to module.apply().
        bias_initializer (callable): Function to initialize all module
            biases to pass to module.apply().
        dropout_layers (Dropout or list<Dropout>): Dropout type to apply to
            hidden layers.
        input_dropout (Dropout): Dropout to apply to input layer.

    Returns:
        Bayesian neural network (BSequential).
    """
    dims = [in_features] + hidden_features
    if not isinstance(dropout_layers, Iterable):
        dropout_layers = [dropout_layers] * len(hidden_features)

    modules = OrderedDict()

    # Input layer.
    if inspect.isclass(input_dropout):
        input_dropout = input_dropout()
    if input_dropout is not None:
        modules["drop_in"] = input_dropout

    # Hidden layers.
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        drop_i = dropout_layers[i]
        if inspect.isclass(drop_i):
            drop_i = drop_i()

        modules["fc_{}".format(i)] = torch.nn.Linear(din, dout)
        if drop_i is not None:
            modules["drop_{}".format(i)] = drop_i
        modules["nonlin_{}".format(i)] = nonlin()

    # Output layer.
    modules["fc_out"] = torch.nn.Linear(dims[-1], out_features)
    if output_nonlin is not None:
        modules["nonlin_out"] = output_nonlin()

    def init(module):
        if callable(weight_initializer) and hasattr(module, "weight"):
            weight_initializer(module.weight)
        if callable(bias_initializer) and hasattr(module, "bias"):
            if module.bias is not None:
                bias_initializer(module.bias)

    # Initialize weights and biases.
    net = BSequential(modules)
    net.apply(init)

    return net
