import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
from sklearn.cluster import KMeans
from qdiff.quant_layer import UniformAffineQuantizer as UAQuantizer


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


class MultiQuantizer(nn.Module):
    """
    Expansion of the class UniformAffineQuantizer found in qdiff/quant_layer.py. Does multiple forms of quantization, some of them not uniform scaled, e.g., K-Means, and MSE.

    n_bits are removed. This considers 4, 3 and 2 currently. 
    """
    def __init__(self, symmetric: bool = False, channel_wise: bool = False, 
                 leaf_param: bool = False, always_zero: bool = False, **kwargs):
        super(MultiQuantizer, self).__init__()
        self.sym = symmetric

        self.quant_range = ["mse", "kmeans", "kmeans_all"] 
        self.bit_range = [4, 3]
        
        self.inited, self.dopas = False, False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise

        self.running_stat = False
        self.always_zero = always_zero
        if self.leaf_param:
            self.x_min, self.x_max = None, None

        # These 2 dictionaries will have the same keys
        self.loss_dict = nn.ParameterDict()
        self.quant_params = nn.ParameterDict()
        self.quant_method = None
        self.original_precision = 32
        self.t_size = None
        self.num_channels = None

    def determine_original_precision_and_size(self, x: torch.Tensor):
        dtype = str(x.dtype)
        if dtype.endswith("16"):
            self.original_precision = 16
        elif dtype.endswith("64"):
            self.original_precision = 64
        else:
            print("Unable to determine original precision. Assuming FP32.")
        self.t_size = np.prod(x.shape)

    def quantify_sizes(self):
        if self.channel_wise is False:
            return 0, 0 # Activation quantization does not have model size savings, nor does it uses channel_wise
        
        quant_method, n_q = self.quant_method.split("-")
        n_q = int(n_q)
        original_size_bytes = int(self.t_size * (self.original_precision / 8))
        quant_size_bytes = int(self.t_size * (n_q / 8))
        if quant_method == "mse":
            quant_overhead = self.original_precision * self.num_channels
            if not self.always_zero:
                quant_overhead *= 2
        elif quant_method == "kmeans_all":
            quant_overhead = (2 ** n_q) * self.original_precision
        elif "kmeans" in quant_method:
            quant_overhead = (2 ** n_q) * self.original_precision * self.num_channels
        else:
            raise NotImplementedError
        quant_overhead = quant_overhead // 8
        return original_size_bytes, quant_size_bytes + quant_overhead
    
    def move_inactive_modules_to_cpu(self):
        for key in self.quant_params.keys():
            if key != self.quant_method and self.quant_params[key].device is not "cpu":
                self.quant_params[key] = self.quant_params[key].to("cpu")
    
    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.init_quantizations(x, self.channel_wise)
            self.inited = True

        if self.dopas is False:
            if self.channel_wise:
                self.num_channels = x.shape[0]
            self.determine_original_precision_and_size(x)    
            self.dopas = True
        
        if self.quant_method is None:
            self.set_min_quant_method()

        self.move_inactive_modules_to_cpu()

        if self.quant_method is None:
            raise Exception
        if self.quant_method.startswith("kmeans"):
            return self.quant_params[self.quant_method]
        elif self.quant_method.startswith("mse"):
            x_int = round_ste(x / self.quant_params[self.quant_method][0]) + self.quant_params[self.quant_method][1]
            if self.sym:
                x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
            else:
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - self.quant_params[self.quant_method][1]) * self.quant_params[self.quant_method][0]
            return x_dequant
        else:
            raise NotImplementedError
    
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

    def init_quantizations(self, x: torch.Tensor, channel_wise: bool = False):
        for q_method in self.quant_range:
            for b in self.bit_range:
                self.loss_dict[f"{q_method}-{b}"] = torch.Tensor([eval(f"self.init_{q_method}(x, b, channel_wise)")]).to(torch.float16)

    def set_min_quant_method(self):
        best_loss = float('inf')
        for k, v in self.loss_dict.items():
            v.requires_grad = False
            if v < best_loss:
                best_loss = v
                self.set_quant_method(k)
                
    def init_mse(self, x: torch.Tensor, bit: int, channel_wise: bool = False):
        delta, zero_point = None, None
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
            recon_losses = []
            for c in range(n_channels):
                recon_loss, d, zp = self.init_mse(x_clone[c], bit, channel_wise=False)
                recon_losses.append(recon_loss)
                delta[c] = d
                zero_point[c] = zp
            
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(-1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            param_dict_key = "-".join(["mse", str(bit)])
            delta_and_zp = torch.cat([delta, zero_point], dim=0).to("cpu")
            self.quant_params[param_dict_key] = nn.Parameter(delta_and_zp, requires_grad=False)
            return sum(recon_losses)
        else:
            self.x_min = x.data.min()
            self.x_max = x.data.max()

            x_max = x.max()
            x_min = x.min()
            best_score = 1e+10
            for i in np.linspace(0, 90, 10): 
                new_max = x_max * (1.0 - (i * 0.01))
                new_min = x_min * (1.0 - (i * 0.01))
                self.n_bits = bit
                self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 111
                x_q = self.quantize(x, bit, new_max, new_min)
                # L_p norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                score = lp_loss(x, x_q, p=2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** bit - 1) \
                        if not self.always_zero else new_max / (2 ** bit - 1)
                    zero_point = (- new_min / delta).round() if not self.always_zero else 0
            return best_score.item(), delta, zero_point

    def init_kmeans_all(self, x: torch.Tensor, bit: int, channel_wise: bool = False):
        x_np = x.clone().detach().cpu().view(1, -1).numpy()
        mykm = KMeans(n_clusters=min(2 ** bit, x_np.shape[1]), max_iter=100).fit(x_np.T)
        for i in range(x_np.shape[1]):
            x_np[0, i] = mykm.cluster_centers_[mykm.labels_[i], :]
        x_b2t = torch.from_numpy(x_np).to(x.device, dtype=x.dtype).view(x.shape)
        loss = lp_loss(x_b2t, x, p=2.4).item()
        self.quant_params["-".join(["kmeans_all", str(bit)])] = nn.Parameter(x_b2t.to("cpu"), requires_grad=False)
        return loss

    def init_kmeans(self, x: torch.Tensor, bit: int, channel_wise: bool = False):
        if channel_wise:
            n_channels = x.shape[0]
            
            x_kmeans, losses = [], []
            for c in range(n_channels):
                x_slice, loss = self.init_kmeans(x[c], bit, channel_wise=False)
                losses.append(loss)
                x_kmeans.append(x_slice)
            self.quant_params["-".join(["kmeans", str(bit)])] = nn.Parameter(torch.cat(x_kmeans, 0).to("cpu"), requires_grad=False)
            return sum(losses)
        else:
            x_np = x.clone().detach().cpu().view(1, -1).numpy()
            mykm = KMeans(n_clusters=min(2 ** bit, x_np.shape[1]), max_iter=100).fit(x_np.T)
            for i in range(x_np.shape[1]):
                x_np[0, i] = mykm.cluster_centers_[mykm.labels_[i], :]
            x_b2t = torch.from_numpy(x_np).to(x.device, dtype=x.dtype).view(x.shape)
            return x_b2t.unsqueeze(0), lp_loss(x_b2t, x, p=2.4, reduction="all").item()

    def set_quant_method(self, method_bit, method=None, bit=None):
        for qm in self.quant_params.keys():
            self.quant_params[qm] = self.quant_params[qm].to("cpu")
        if method is not None:
            assert bit is not None
            method_bit = "-".join([method, str(bit)])
        else:
            bit = method_bit.split("-")[-1]
        self.quant_method = method_bit
        self.quant_params[self.quant_method] = self.quant_params[self.quant_method].to("cuda")
        self.n_bits = int(bit)
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1

    def quantize(self, x, bit, max, min):
        delta = (max - min) / (2 ** bit - 1) if not self.always_zero else max / (2 ** bit - 1)
        zero_point = (- min / delta).round() if not self.always_zero else 0
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        return f"Multiquantizer. Inited = {self.inited}"


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
        self.weight_quantizer = MultiQuantizer(**self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer = UAQuantizer(**self.act_quant_params)
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
                    if len(input.shape) == 3:
                        input_0 = self.act_quantizer(input[:, :, :self.split])
                        input_1 = self.act_quantizer_0(input[:, :, self.split:])
                        input = torch.cat([input_0, input_1], dim=-1)
                    else:
                        input_0 = self.act_quantizer(input[:, :self.split, :, :])
                        input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
                        input = torch.cat([input_0, input_1], dim=1)
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
            bias = bias.to(input.dtype)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_split(self):
        self.weight_quantizer_0 = MultiQuantizer(**self.weight_quant_params)
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer_0 = UAQuantizer(**self.act_quant_params)

    def set_running_stat(self, running_stat: bool):
        if self.act_quant_mode == 'qdiff':
            self.act_quantizer.running_stat = running_stat
            if self.split != 0:
                self.act_quantizer_0.running_stat = running_stat

    # Special function for loading state dict.
    # Meant to be called by the special quant model
    # Assume custom_sd contains only key-values relevant.
    def custom_sd_load(self, custom_sd):
        if hasattr(self, "weight_quantizer_0"):
            self._find_rel_keys_and_replace(custom_sd, moduleName="weight_quantizer_0")
            self.weight_quantizer_0.inited = True
        self._find_rel_keys_and_replace(custom_sd, moduleName="weight_quantizer")
        self.weight_quantizer.inited = True


    def _find_rel_keys_and_replace(self, custom_sd, moduleName="weight_quantizer", sub_items=("loss_dict", "quant_params")):
        for si in sub_items:
            rel_keys = [k for k in custom_sd.keys() if (moduleName in k and si in k)]
            for rk in rel_keys:
                subkey = rk.split(".")[-1]
                module_to_edit = eval(f"self.{moduleName}.{si}")
                module_to_edit[subkey] = custom_sd[rk]
                module_to_edit[subkey].requires_grad = False
                del custom_sd[rk]

