import logging
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._torch_docs import reproducibility_notes

log = logging.getLogger(__name__)


class QRFixingModel(nn.Module):
    def __init__(self, existing_model: nn.Module, stability_frequency: int = 10):
        super().__init__()
        self.track_stab_lst = {}
        self.target_model = self._replace_layers(existing_model)
        # print(self.target_model)
        # raise ValueError("")
        self.stability_frequency = stability_frequency
        self.call_count = 0
        self.skip_stability = False

    def _replace_layers(self, module, name=None, process_group=None):
        module_output = module
        # this will remove all the BatchNorm layers from the network

        for name, mod in module_output.named_modules():
            # print(name, mod)
            if len(mod._modules) == 0:
                if isinstance(mod, nn.Linear):
                    lpmod = QRLinear(
                        in_features=mod.in_features,
                        out_features=mod.out_features,
                        bias=mod.bias is not None,
                    ).to(device=mod.weight.device, dtype=mod.weight.dtype)
                    # del mod
                    # module_output.add_module(name, lpmod)
                    # setattr(module_output._modules, name, lpmod)
                    module_output._modules[name] = lpmod
                    self.track_stab_lst[name] = 0
                    # self.reset_layers = [module, name]
                elif isinstance(mod, nn.Conv2d):
                    lpmod = QRConv2d(
                        in_channels=mod.in_channels,
                        out_channels=mod.out_channels,
                        kernel_size=mod.kernel_size,
                        stride=mod.stride,
                        padding=mod.padding,
                        dilation=mod.dilation,
                        groups=mod.groups,
                        bias=mod.bias is not None,
                        padding_mode=mod.padding_mode,
                    ).to(device=mod.weight.device, dtype=mod.weight.dtype)
                    # del mod
                    # setattr(module_output._modules, name, lpmod)
                    module_output._modules[name] = lpmod
                    self.track_stab_lst[name] = 0
            # if len(mod._modules) == 0 and hasattr(mod, "test_q_stability"):
            #     self.track_stab_lst.append([name, 0])
        del module
        return module_output

    def test_basis_stability_all_layers(self, module):
        if self.skip_stability:
            return
        log.info("Testing stability")
        for name, mod in module.named_modules():
            # print(name, mod)
            if hasattr(mod, "test_q_stability"):
                changing = mod.test_q_stability()
                self.track_stab_lst[name] = changing
                if changing == 1:
                    log.info(f"Fixing Q for layer: {name}")
                elif changing == 0:
                    log.debug(f"Training normally for layer: {name}")
                else:
                    log.debug(f"Q was fixed previously for layer: {name}")  # TODO: remove!
        all_stable = True
        for layer in self.track_stab_lst:
            if self.track_stab_lst[layer] == 0:
                all_stable = False
                break
        if all_stable:
            self.skip_stability = True

    def forward(self, inputs):
        self.call_count += 1
        # print(self.call_count % self.stability_frequency, self.call_count, self.stability_frequency)
        if (
            self.target_model.training
            and self.call_count % self.stability_frequency == self.stability_frequency - 1
        ):
            self.test_basis_stability_all_layers(module=self.target_model)
        return self.target_model(inputs)


class QRLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(QRLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if out_features >= in_features:  # simplest case (no transpose)
            self.q = nn.Parameter(
                torch.empty((out_features, in_features), requires_grad=False, **factory_kwargs),
                requires_grad=False,
            )
            self.r = nn.Parameter(
                torch.empty((in_features, in_features), requires_grad=False, **factory_kwargs),
                requires_grad=True,
            )
            self.trans = False
        else:
            self.q = nn.Parameter(
                torch.empty((in_features, out_features), requires_grad=False, **factory_kwargs),
                requires_grad=False,
            )
            self.r = nn.Parameter(
                torch.empty((out_features, out_features), requires_grad=False, **factory_kwargs),
                requires_grad=True,
            )
            self.trans = True

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.cossim = nn.CosineSimilarity(dim=0)
        self.q_fixed = False

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.q_fixed:
            # with torch.no_grad():
            self.q.requires_grad = False
            w = (self.q @ self.r).T if self.trans else self.q @ self.r
        else:
            w = self.weight
        return F.linear(input, w, self.bias)

    @torch.no_grad()
    def test_q_stability(self):
        if self.q_fixed:
            # only do the switch once!
            return 2
        q, r = torch.linalg.qr(self.weight.T if self.trans else self.weight, mode="reduced")
        csim = self.cossim(q, self.q)
        csmean, _ = csim.mean(), csim.std()
        self.q.set_(q)
        if csmean > 0.99:  # todo: make this a class parameter!
            self.r.set_(r)
            self.q_fixed = True
            return 1
        return 0

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
        )


class QRConv2d(nn.modules.conv._ConvNd):
    __doc__ = (
        r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    """
        + r"""

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'valid', 'same'}} or a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        {depthwise_separable_note}

    Note:
        {cudnn_reproducibility_note}

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Note:
        This module supports complex data types i.e. ``complex32, complex64, complex128``.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """.format(
            **reproducibility_notes, **nn.modules.conv.convolution_notes
        )
        + r"""

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples:

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: nn.common_types._size_2_t,
        stride: nn.common_types._size_2_t = 1,
        padding: Union[str, nn.common_types._size_2_t] = 0,
        dilation: nn.common_types._size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = nn.modules.utils._pair(kernel_size)
        stride_ = nn.modules.utils._pair(stride)
        padding_ = padding if isinstance(padding, str) else nn.modules.utils._pair(padding)
        dilation_ = nn.modules.utils._pair(dilation)
        super(QRConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            nn.modules.utils._pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

        weight_shape = self.weight.shape
        wsh0 = weight_shape[0]
        wsh1 = weight_shape[1] * weight_shape[2] * weight_shape[3]  # todo: smarter? only Conv2D ...
        if wsh0 >= wsh1:  # simplest case (no transpose)
            self.q = nn.Parameter(
                torch.empty((wsh0, wsh1), requires_grad=False, **factory_kwargs),
                requires_grad=False,
            )
            self.r = nn.Parameter(
                torch.empty((wsh1, wsh1), requires_grad=False, **factory_kwargs),
                requires_grad=True,
            )
            self.trans = False
        else:
            self.q = nn.Parameter(
                torch.empty((wsh1, wsh0), requires_grad=False, **factory_kwargs),
                requires_grad=False,
            )
            self.r = nn.Parameter(
                torch.empty((wsh0, wsh0), requires_grad=False, **factory_kwargs),
                requires_grad=True,
            )
            self.trans = True
        self.q_fixed = False

    def _conv_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                nn.modules.utils._pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.q_fixed:
            self.q.requires_grad = False
            # with torch.no_grad():
            w = (self.q @ self.r).T if self.trans else self.q @ self.r
            if self.transposed:
                self.weight = w.reshape(
                    self.in_channels,
                    self.out_channels // self.groups,
                    *self.kernel_size,
                )
            else:
                self.weight = w.reshape(
                    self.out_channels,
                    self.in_channels // self.groups,
                    *self.kernel_size,
                )
        else:
            w = self.weight
        return self._conv_forward(input, w, self.bias)

    @torch.no_grad()
    def test_q_stability(self):
        if self.q_fixed:
            # only do the switch once!
            return 2

        w = self.weight.view(self.weight.shape[0], -1)
        q, r = torch.linalg.qr(w.T if self.trans else w, mode="reduced")
        csim = self.cossim(q, self.q)
        csmean, _ = csim.mean(), csim.std()
        self.q.set_(q)
        if csmean > 0.99:  # todo: make this a class parameter!
            self.r.set_(r)
            self.q_fixed = True
            return 1
        return 0
