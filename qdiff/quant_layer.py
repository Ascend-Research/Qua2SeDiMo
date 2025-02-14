import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np

logger = logging.getLogger(__name__)


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, always_zero: bool = False, fp: bool = False, mantissa_bits: int = None, 
                 weight_group_size: int = None, fp_biased_adaround: bool = True, 
                 online_act_quant: bool = False, 
                 group_quant: bool = False, **kwargs):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.running_stat = False
        self.always_zero = always_zero
        self.fp_biased_adaround = fp_biased_adaround

        if self.leaf_param:
            self.x_min, self.x_max = None, None
        if self.scale_method == "mse":
            self.alpha_dict = {}
        elif self.scale_method in ["kmeans", "coreset"]:
            self.dist_dev_list = []
            self.c_k = []
        elif self.scale_method == "kmeans_all":
            self.k_all = None

        self.online_act_quant = online_act_quant
        self.dynamic_idx = None
        self.dynamic_mantissa = None

        self.dim = 0

        self.group_quant = False
        self.group_size = weight_group_size

        if self.leaf_param == False:
            self.group_quant = group_quant

    def forward(self, x: torch.Tensor):
        if self.group_quant == True:
            x_old = x
            x = x.view(-1, self.group_size)
        
        if (self.online_act_quant) and (self.leaf_param):
            return self.online_per_tensor_quant(x)

        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise, print_stats=True)
                self.delta = torch.nn.Parameter(delta)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise, print_stats=True)
            self.inited = True

        if self.running_stat:
            self.act_momentum_update(x)

        if self.scale_method in ["kmeans", "coreset"]:
            return self.dequant_kmeans(x)
        elif self.scale_method == "kmeans_all":
            return self.k_all

        x_int = round_ste(x / self.delta) + self.zero_point
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant
    
    def act_momentum_update(self, x: torch.Tensor, act_range_momentum: float = 0.95):
        assert(self.inited)
        assert(self.leaf_param)

        x_min = x.data.min()
        x_max = x.data.max()
        self.x_min = self.x_min * act_range_momentum + x_min * (1 - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1 - act_range_momentum)

        if self.sym:
            delta = torch.max(self.x_min.abs(), self.x_max.abs()) / self.n_levels
        else:
            delta = (self.x_max - self.x_min) / (self.n_levels - 1) if not self.always_zero \
                else self.x_max / (self.n_levels - 1)
        
        delta = torch.clamp(delta, min=1e-8)
        if not self.sym:
            self.zero_point = (-self.x_min / delta).round() if not (self.sym or self.always_zero) else 0
        self.delta = torch.nn.Parameter(delta)
    
    def online_per_tensor_quant(self, x: torch.Tensor):
        '''
        [BS, n_token, C_in] Attn
        [batch_size, num_channels, height, width] 2D CNN
        [batch_size, num_features] MLP
        '''
        x_shape = x.shape
        try:
            assert len(x.shape) in [2, 3, 4], "Tensor shape must have length of 2, 3, or 4."
        except:
            import ipdb; ipdb.set_trace()
        if len(x.shape) == 3:
            Cin = x.shape[-1]
            x = x.reshape(-1, Cin)
        elif len(x.shape) == 4:
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
        elif  len(x.shape) == 2:
            pass
        x_min = x.min(dim=-1)[0]
        x_max = x.max(dim=-1)[0]

        x_dequant = self.quantize(x, x_max, x_min)
        x_dequant = x_dequant.reshape(x_shape)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, print_stats=False):
        if print_stats:
            x_mu, x_sig, x_min, x_max = x.mean().detach().item(), x.std().detach().item(), x.min().detach().item(), x.max().detach().item()
            print(f"{x_mu}, {x_sig}, {x_min}, {x_max}")
        delta, zero_point = None, None
        if self.scale_method == "kmeans_all":
            from sklearn.cluster import KMeans
            x_np = x.clone().detach().cpu().view(1, -1).numpy()
            mykm = KMeans(n_clusters=min(2 ** self.n_bits, x_np.shape[1]), max_iter=100).fit(x_np.T)
            for i in range(x_np.shape[1]):
                x_np[0, i] = mykm.cluster_centers_[mykm.labels_[i], :]
            x_b2t = torch.from_numpy(x_np).to(x.device, dtype=x.dtype).view(x.shape)
            self.k_all = x_b2t
            return 1, 0
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                x_min = x_clone.abs().min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
                x_min = x_clone.abs().min(dim=-1)[0].min(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
                x_min = x_clone.abs().min(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(-1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    delta = x_absmax / self.n_levels
                else:
                    delta = float(x.max().item() - x.min().item()) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score, best_i = 1e+10, -1
                for i in np.linspace(0, 90, 10): 
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_i = i
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1) \
                            if not self.always_zero else new_max / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round() if not self.always_zero else 0
                if best_i in self.alpha_dict.keys():
                    self.alpha_dict[best_i] += 1
                else:
                    self.alpha_dict[best_i] = 1
            elif self.scale_method == 'kmeans':
                from sklearn.cluster import KMeans
                x_np = x.clone().detach().cpu().view(1, -1).numpy()
                mykm = KMeans(n_clusters=min(2 ** self.n_bits, x_np.shape[1]), max_iter=100).fit(x_np.T)
                for i in range(x_np.shape[1]):
                    x_np[0, i] = mykm.cluster_centers_[mykm.labels_[i], :]
                x_b2t = torch.from_numpy(x_np).to(x.device, dtype=x.dtype).view(x.shape)
                self.c_k.append(x_b2t.unsqueeze(0))
                centers_dist = mykm.cluster_centers_[:, 0]
                centers_dist.sort()
                distances = []
                for i in range(len(centers_dist[1:])):
                    distances.append(abs(centers_dist[i] - centers_dist[i-1]))
                self.dist_dev_list.append(np.std(distances))
                delta, zero_point = 1, 0
            elif self.scale_method == 'coreset':
                from qdiff.utils import greedy_core_set_selection
                dist_func = lambda a, b, m: m[f"{a}-{b}"] if f"{a}-{b}" in m else torch.abs(a-b).item()
                x_np = x.clone().detach().view(-1)
                coreset, memo = greedy_core_set_selection(x_np , size=min(2**self.n_bits, x_np.shape[0]), dist_func=dist_func, verbose=False)
                for i in range(x_np.shape[0]):
                    distances = torch.Tensor([dist_func(x_np[i], point, memo) for point in coreset])
                    min_point_index = torch.argmin(distances)
                    x_np[i] = coreset[min_point_index]
                x_b2t = x_np.to(x.device, dtype=x.dtype).view(x.shape)
                self.c_k.append(x_b2t.unsqueeze(0))
                coreset.sort()
                distances = []
                for i in range(len(coreset[1:])):
                    distances.append(torch.abs(coreset[i] - coreset[i-1]).item())
                self.dist_dev_list.append(np.std(distances))
                delta, zero_point = 1, 0
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1) if not self.always_zero else max / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round() if not self.always_zero else 0
        delta = delta.unsqueeze(-1)
        x_int = torch.round(x / delta)
        if type(zero_point) is int:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_float_q = x_quant * delta
        else:
            zero_point = zero_point.unsqueeze(-1)
            x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)
    
    def dequant_kmeans(self, x):
        if type(self.c_k) is list:
            self.c_k = torch.cat(self.c_k, dim=0)
        return self.c_k

    def report_delta_shift(self):
        if self.max_delta is None:
            print("No delta reported")
            return
        percent_change = (self.delta - self.max_delta) / self.max_delta
        percent_change *= 100
        p_mu, p_sig, p_min, p_max = torch.mean(percent_change), torch.std(percent_change), percent_change.min().item(), percent_change.max().item()
        return f"{p_min}, {p_max}, {p_mu}, {p_sig}"


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear, nn.Conv1d], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, act_quant_mode: str = 'qdiff'):
        super(QuantModule, self).__init__()
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None

        self.use_weight_quant = False
        self.use_act_quant = False
        self.act_quant_mode = act_quant_mode
        self.disable_act_quant = disable_act_quant

        self.weight_quantizer = UniformAffineQuantizer(**self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer = UniformAffineQuantizer(**self.act_quant_params)
        self.split = 0

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.extra_repr = org_module.extra_repr

        if hasattr(org_module, "in_features"):
            self.in_features = org_module.in_features
        if hasattr(org_module, "nametag"):
            self.nametag = org_module.nametag
        else: self.nametag = None
        self.run_prints = True

    def report_delta_shift(self):
        if hasattr(self, "weight_quantizer"):
            print(f"WQ: {self.weight_quantizer.report_delta_shift()}")
        if hasattr(self, "weight_quantizer_0"):
            print(f"WQ0: {self.weight_quantizer_0.report_delta_shift()}")
        if hasattr(self, "act_quantizer"):
            print(f"AQ: {self.act_quantizer.report_delta_shift()}")
        if hasattr(self, "act_quantizer_0"):
            print(f"AQ: {self.act_quantizer_0.report_delta_shift()}")

    def forward(self, input: torch.Tensor, split: int = 0):
        og_dtype = input.dtype
        if split != 0 and self.split != 0:
            assert(split == self.split)
        elif split != 0:
            logger.info(f"split at {split}!")
            self.split = split
            self.set_split()

        if self.run_prints and self.nametag is not None:
            self.run_prints = False
        if not self.disable_act_quant and self.use_act_quant:
            if self.split != 0:
                if self.act_quant_mode == 'qdiff':
                    input_0 = self.act_quantizer(input[:, :self.split, :, :])
                    input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                if self.act_quant_mode == 'qdiff':
                    input = self.act_quantizer(input)
        if self.use_weight_quant:
            if self.split != 0:
                weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
                weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        if weight.dtype != input.dtype:
            weight = weight.to(input.dtype)
        if bias.dtype != input.dtype:
            bias = bias.to(input.dtype)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out.to(og_dtype)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_split(self):
        self.weight_quantizer_0 = UniformAffineQuantizer(**self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer_0 = UniformAffineQuantizer(**self.act_quant_params)

    def set_running_stat(self, running_stat: bool):
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer.running_stat = running_stat
            if self.split != 0:
                self.act_quantizer_0.running_stat = running_stat
