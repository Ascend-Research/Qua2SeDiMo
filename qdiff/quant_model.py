import logging
import torch
import torch.nn as nn
from qdiff.quant_block import get_specials, BaseQuantBlock
from qdiff.quant_block import QuantBasicTransformerBlock
from qdiff.quant_block import QuantQKMatMul, QuantSMVMatMul, QuantBasicTransformerBlock, QuantAttnBlock
from qdiff.quant_block import QuantDiffBTB
from qdiff.quant_layer import QuantModule, StraightThrough
from ldm.modules.attention import BasicTransformerBlock
from qdiff.quant_layer_multi import QuantModule as QuantModuleMulti
from qdiff.quant_layer_multi import MultiQuantizer
import random
import numpy as np

logger = logging.getLogger(__name__)


class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials(act_quant_params['leaf_param'])

        if hasattr(model, "dtype"):
            self.dtype=model.dtype
        if hasattr(model, "config"):
            self.config = model.config
            if hasattr(model, "add_embedding"):
                self.add_embedding = model.add_embedding
                self.forward = self.forward_diffusers
            else:
                self.forward = model.forward

        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if not hasattr(module, "nametag"):
                module.nametag = "model"
            child_module.nametag = ".".join([module.nametag, name]) 
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock, QuantDiffBTB]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)
    
    def forward_diffusers(self, latent_model_input,
                    t,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                    added_cond_kwargs,
                    return_dict=False,
                ):
        return self.model(
            sample=latent_model_input, 
            timestep=t, 
            encoder_hidden_states=encoder_hidden_states, 
            cross_attention_kwargs=cross_attention_kwargs, 
            added_cond_kwargs=added_cond_kwargs, 
            return_dict=return_dict)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_q.running_stat = running_stat
                    m.attn2.act_quantizer_k.running_stat = running_stat
                    m.attn2.act_quantizer_v.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
            elif isinstance(m, QuantDiffBTB):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    if m.attn2 is not None:
                        m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    if m.attn2 is not None:
                        m.attn2.act_quantizer_q.running_stat = running_stat
                        m.attn2.act_quantizer_k.running_stat = running_stat
                        m.attn2.act_quantizer_v.running_stat = running_stat
                        m.attn2.act_quantizer_w.running_stat = running_stat
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)

    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantDiffBTB, QuantBasicTransformerBlock, BasicTransformerBlock)):
                m.checkpoint = grad_ckpt


class QuantModelMultiQ(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, prefix=None, sdv15=False, **kwargs):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials(act_quant_params['leaf_param'], sdv15=sdv15)

        if hasattr(model, "dtype"):
            self.dtype=model.dtype
        if hasattr(model, "config"):
            self.config = model.config
            if hasattr(model, "add_embedding"):
                self.add_embedding = model.add_embedding
                self.forward = self.forward_diffusers
            else:
                self.forward = model.forward

        # For the skip-connect splits in SDv1.5, which is defined in this repo.
        # For SDXL it is always enabled.
        # These skips also appear in HunYuan
        if hasattr(self.model, "split"):
            self.model.split = True
            
        # HunYuan-DiT
        if hasattr(self.model, "inner_dim"):
            self.inner_dim = self.model.inner_dim
        if hasattr(self.model, "num_heads"):
            self.num_heads = self.model.num_heads

        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params, prefix=prefix)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, prefix=None):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if not hasattr(module, "nametag"):
                module.nametag = "model"
            child_module.nametag = ".".join([module.nametag, name]) 
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if prefix is not None:
                    if prefix in child_module.nametag:
                        setattr(module, name, QuantModuleMulti(
                        child_module, weight_quant_params, act_quant_params))
                        prev_quantmodule = getattr(module, name)
                    else:
                        continue
                else:
                    setattr(module, name, QuantModuleMulti(
                        child_module, weight_quant_params, act_quant_params))
                    prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, prefix=prefix)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttnBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantSMVMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantQKMatMul:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params, )

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModuleMulti, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)
    
    def forward_diffusers(self, latent_model_input,
                    t,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                    added_cond_kwargs,
                    timestep_cond,
                    return_dict=False,
                ):
        return self.model(
            sample=latent_model_input, 
            timestep=t, 
            encoder_hidden_states=encoder_hidden_states, 
            cross_attention_kwargs=cross_attention_kwargs, 
            added_cond_kwargs=added_cond_kwargs, 
            timestep_cond=timestep_cond,
            return_dict=return_dict)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    m.attn2.act_quantizer_q.running_stat = running_stat
                    m.attn2.act_quantizer_k.running_stat = running_stat
                    m.attn2.act_quantizer_v.running_stat = running_stat
                    m.attn2.act_quantizer_w.running_stat = running_stat
            elif isinstance(m, QuantDiffBTB):
                if sm_only:
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    if m.attn2 is not None:
                        m.attn2.act_quantizer_w.running_stat = running_stat
                else:
                    m.attn1.act_quantizer_q.running_stat = running_stat
                    m.attn1.act_quantizer_k.running_stat = running_stat
                    m.attn1.act_quantizer_v.running_stat = running_stat
                    m.attn1.act_quantizer_w.running_stat = running_stat
                    if m.attn2 is not None:
                        m.attn2.act_quantizer_q.running_stat = running_stat
                        m.attn2.act_quantizer_k.running_stat = running_stat
                        m.attn2.act_quantizer_v.running_stat = running_stat
                        m.attn2.act_quantizer_w.running_stat = running_stat
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)

    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantDiffBTB, QuantBasicTransformerBlock, BasicTransformerBlock)):
                m.checkpoint = grad_ckpt

    def get_all_multiquantizers(self, module=None):
        mq_dict = {}
        if module is None:
            module = self.model
        for name, child_module in module.named_children():
            if not hasattr(module, "nametag"):
                module.nametag = "model"
            child_module.nametag = ".".join([module.nametag, name]) 
            if isinstance(child_module, MultiQuantizer):
                mq_dict[child_module.nametag] = child_module
            else:
                mq_dict = {**mq_dict, **self.get_all_multiquantizers(child_module)}
        return mq_dict

    def get_all_quantmodules(self, module=None):
        qm_dict = {}
        if module is None:
            module = self.model
        for name, child_module in module.named_children():
            if not hasattr(module, "nametag"):
                module.nametag = "model"
            child_module.nametag = ".".join([module.nametag, name]) 
            if isinstance(child_module, QuantModuleMulti):
                qm_dict[child_module.nametag] = child_module
            else:
                qm_dict = {**qm_dict, **self.get_all_quantmodules(child_module)}
        return qm_dict

    def quantify_model_savings(self):
        original_size, quant_size = 0, 0        
        for mq_name, mq_obj in self.get_all_multiquantizers().items():
            # NOTE Special case for HunYuan component as it causes issues bypassing multiquantizer.
            if "time_extra_emb.pooler" in mq_name and "_proj" in mq_name:
                continue
            module_og_size, module_q_size = mq_obj.quantify_sizes()
            original_size += module_og_size
            quant_size += module_q_size

        print("Original size in bytes: ", original_size)
        print("Quantized size in bytes: ", quant_size)
        print(f"Reduction to {(quant_size/original_size)*100}% of FP model size.")
        return quant_size

    def get_current_quant_config(self, skip_act=True, include_stats=False):
        mq_config = {}
        for mq_name, mq_obj in self.get_all_multiquantizers().items():
            if skip_act and "act_quantizer" in mq_name:
                continue
            if "time_extra_emb.pooler" in mq_name and "_proj" in mq_name:
                continue
            mq_config[mq_name] = [mq_obj.quant_method]
            if include_stats:
                module_og_size, module_q_size = mq_obj.quantify_sizes()
                module_error = mq_obj.loss_dict[mq_obj.quant_method].item()
                mq_config[mq_name] += [module_og_size, module_q_size, module_error]
        return mq_config

    def set_random_config(self, skip_act=True, dit=False, a_l=0.0, a_u=1.0): 
        mq_dict = self.get_all_multiquantizers()
        ave_bits_p = np.random.uniform(low=a_l, high=a_u)
        bits = np.random.choice([3, 4], size=len(mq_dict), p=[ave_bits_p, 1-ave_bits_p])
        i = 0
        for mq_name, mq_obj in mq_dict.items():
            if skip_act and "act_quantizer" in mq_name:
                continue
            if "time_extra_emb.pooler" in mq_name and "_proj" in mq_name:
                continue
            options = [o for o in mq_obj.loss_dict.keys()]
            config = random.choice(options)
            config = config[:-1]
            config += str(bits[i])
            mq_obj.set_quant_method(config)
            i += 1
        # This conditional is thrown in because the DiT design is weird in that all transformer blocks possess the same timestep embedding Linear-SiLU-Linear sequential and they all invoke it separately. However, it it one set of weights, and should possess one set of quantization settings.
        # So, in code, we will generate quantizers for all of them, even though its referring to the same weights actually. Makes the structure very weird.
        # Basically, when randomly sampling, since this is supposed to be 1 set of weights, we assign the same quant_method across the board.
        # Weird hack but it works.
        if dit:
            for i in range(1, 28):
                mq_dict[f'model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_1.weight_quantizer'].set_quant_method(mq_dict['model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_1.weight_quantizer'].quant_method)
                mq_dict[f'model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_2.weight_quantizer'].set_quant_method(mq_dict['model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_2.weight_quantizer'].quant_method)


    def load_mq_state_dicts(self, filepath=None, sd=None):
        if filepath is None:
            assert sd is not None
        elif sd is None:
            assert filepath is not None
            sd = torch.load(filepath)
        qm_dict = self.get_all_quantmodules()
        for qm_name in qm_dict.keys():
            sub_dict = {sd_key: sd[sd_key] for sd_key in sd.keys() if sd_key.startswith(qm_name)}
            qm_dict[qm_name].custom_sd_load(sub_dict)
            for sd_key in sub_dict.keys():
                del sd[sd_key]
        return sd
    
    def load_quant_config(self, filepath):
        import pickle
        with open(filepath, "rb") as f:
            quant_config_dict = pickle.load(f)

        all_wqs = self.get_all_multiquantizers()
        for k, v in quant_config_dict.items():
            all_wqs[k].set_quant_method(v)

        if "dit" in filepath:
            for i in range(1, 28):
                all_wqs[f'model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_1.weight_quantizer'].set_quant_method(all_wqs['model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_1.weight_quantizer'].quant_method)
                all_wqs[f'model.transformer_blocks.{i}.norm1.emb.timestep_embedder.linear_2.weight_quantizer'].set_quant_method(all_wqs['model.transformer_blocks.0.norm1.emb.timestep_embedder.linear_2.weight_quantizer'].quant_method)