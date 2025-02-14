import logging
from types import MethodType
import torch as th
from torch import einsum
import torch.nn as nn
from einops import rearrange, repeat

from qdiff.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough
from qdiff.quant_layer_multi import QuantModule as QuantModuleMultiQ
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepBlock, checkpoint
from ldm.modules.diffusionmodules.openaimodel import QKMatMul, SMVMatMul
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.attention import exists, default

import diffusers
from ddim.models.diffusion import ResnetBlock, AttnBlock, nonlinearity
import inspect
from typing import Optional, Any, Dict
from qdiff.quant_aware_attn_processors import QuantAttnProcessor, QuantHunyuanAttnProcessor2_0, QuantFusedHunyuanAttnProcessor2_0


logger = logging.getLogger(__name__)


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantResBlock(BaseQuantBlock, TimestepBlock):
    def __init__(
        self, res: ResBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm

        self.in_layers = res.in_layers

        self.updown = res.updown

        self.h_upd = res.h_upd
        self.x_upd = res.x_upd

        self.emb_layers = res.emb_layers
        self.out_layers = res.out_layers

        self.skip_connection = res.skip_connection

    def forward(self, x, emb=None, split=0):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if split != 0 and self.skip_connection.split == 0:
            return checkpoint(
                self._forward, (x, emb, split), self.parameters(), self.use_checkpoint
            )
        return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )  

    def _forward(self, x, emb, split=0):
        if emb is None:
            assert(len(x) == 2)
            x, emb = x
        assert x.shape[2] == x.shape[3]

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        if split != 0:
            return self.skip_connection(x, split=split) + h
        return self.skip_connection(x) + h


class QuantQKMatMul(BaseQuantBlock):
    def __init__(
        self, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.scale = None
        self.use_act_quant = False
        self.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        
    def forward(self, q, k):
        if self.use_act_quant:
            quant_q = self.act_quantizer_q(q * self.scale)
            quant_k = self.act_quantizer_k(k * self.scale)
            weight = th.einsum(
                "bct,bcs->bts", quant_q, quant_k
            ) 
        else:
            weight = th.einsum(
                "bct,bcs->bts", q * self.scale, k * self.scale
            )
        return weight

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant


class QuantSMVMatMul(BaseQuantBlock):
    def __init__(
        self, act_quant_params: dict = {}, sm_abit=8):
        super().__init__(act_quant_params)
        self.use_act_quant = False
        self.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['symmetric'] = False
        act_quant_params_w['always_zero'] = True
        self.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)
        
    def forward(self, weight, v):
        if self.use_act_quant:
            a = th.einsum("bts,bcs->bct", self.act_quantizer_w(weight), self.act_quantizer_v(v))
        else:
            a = th.einsum("bts,bcs->bct", weight, v)
        return a

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant


class QuantAttentionBlock(BaseQuantBlock):
    def __init__(
        self, attn: AttentionBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = attn.channels
        self.num_heads = attn.num_heads
        self.use_checkpoint = attn.use_checkpoint
        self.norm = attn.norm
        self.qkv = attn.qkv
        
        self.attention = attn.attention

        self.proj_out = attn.proj_out

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def cross_attn_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    if self.use_act_quant:
        quant_q = self.act_quantizer_q(q)
        quant_k = self.act_quantizer_k(k)
        sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -th.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    if self.use_act_quant:
        out = einsum('b i j, b j d -> b i d', self.act_quantizer_w(attn), self.act_quantizer_v(v))
    else:
        out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


class QuantBasicTransformerBlock(BaseQuantBlock):
    def __init__(
        self, tran: BasicTransformerBlock, act_quant_params: dict = {}, 
        sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.attn1 = tran.attn1
        self.ff = tran.ff
        self.attn2 = tran.attn2
        
        self.norm1 = tran.norm1
        self.norm2 = tran.norm2
        self.norm3 = tran.norm3
        self.checkpoint = tran.checkpoint

        self.attn1.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)

        self.attn2.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.attn2.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.attn2.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['always_zero'] = True
        self.attn1.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)
        self.attn2.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)

        self.attn1.forward = MethodType(cross_attn_forward, self.attn1)
        self.attn2.forward = MethodType(cross_attn_forward, self.attn2)
        self.attn1.use_act_quant = False
        self.attn2.use_act_quant = False

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        if context is None:
            assert(len(x) == 2)
            x, context = x

        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.attn1.use_act_quant = act_quant
        self.attn2.use_act_quant = act_quant

        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantResnetBlock(BaseQuantBlock):
    def __init__(
        self, res: ResnetBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.in_channels = res.in_channels
        self.out_channels = res.out_channels
        self.use_conv_shortcut = res.use_conv_shortcut

        self.norm1 = res.norm1
        self.conv1 = res.conv1
        self.temb_proj = res.temb_proj
        self.norm2 = res.norm2
        self.dropout = res.dropout
        self.conv2 = res.conv2
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = res.conv_shortcut
            else:
                self.nin_shortcut = res.nin_shortcut
        self.nin_check = None


    def forward(self, x, temb=None, split=0):
        if temb is None:
            assert(len(x) == 2)
            x, temb = x

        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                if self.nin_check is None:
                    self.nin_check = True if "split" in inspect.getfullargspec(self.nin_shortcut.forward).args else False
                if self.nin_check:
                    x = self.nin_shortcut(x, split=split)
                else:
                    x = self.nin_shortcut(x)
        out = x + h
        return out


class QuantAttnBlock(BaseQuantBlock):
    def __init__(
        self, attn: AttnBlock, act_quant_params: dict = {}, sm_abit=8):
        super().__init__(act_quant_params)
        self.in_channels = attn.in_channels

        self.norm = attn.norm
        self.q = attn.q
        self.k = attn.k
        self.v = attn.v
        self.proj_out = attn.proj_out

        self.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        self.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        if self.use_act_quant:
            q = self.act_quantizer_q(q)
            k = self.act_quantizer_k(k)
        w_ = th.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        if self.use_act_quant:
            v = self.act_quantizer_v(v)
            w_ = self.act_quantizer_w(w_)
        h_ = th.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        
        out = x + h_
        return out


def get_specials(quant_act=False, sdv15=False):

    specials = {
        ResBlock: QuantResBlock,
        BasicTransformerBlock: QuantBasicTransformerBlock,
        ResnetBlock: QuantResnetBlock,
        AttnBlock: QuantAttnBlock,
    }
    if not sdv15:
        specials[diffusers.models.attention.BasicTransformerBlock] = QuantDiffBTB
        specials[diffusers.models.resnet.ResnetBlock2D] = QuantDiffRB
        
    if quant_act:
        specials[QKMatMul] = QuantQKMatMul
        specials[SMVMatMul] = QuantSMVMatMul
    else:
        specials[AttentionBlock] = QuantAttentionBlock
    if int(diffusers.__version__.split('.')[1]) >= 28 and not sdv15:
        specials[diffusers.models.transformers.hunyuan_transformer_2d.HunyuanDiTBlock] = QuantHunyuanBlock
    return specials


class QuantHunyuanBlock(BaseQuantBlock):
    def __init__(self, tran,
                 act_quant_params: dict = {}, sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.norm1 = tran.norm1
        self.attn1 = tran.attn1
        self.norm2 = tran.norm2
        self.attn2 = tran.attn2
        self.norm3 = tran.norm3
        self.ff = tran.ff
        self._chunk_size = tran._chunk_size
        self._chunk_dim = tran._chunk_dim
        self.set_chunk_feed_forward = tran.set_chunk_feed_forward

        if tran.skip_linear is None:
            self.split = 0
            self.skip_linear = None
        else:
            self.skip_linear = tran.skip_linear
            self.skip_norm = tran.skip_norm
            self.split = tran.skip_linear.weight.shape[1] // 2
            if isinstance(self.skip_linear, QuantModule) or isinstance(self.skip_linear, QuantModuleMultiQ):
                self.skip_q = True
            else:
                self.skip_q = False

        assert isinstance(self.attn1.processor, (diffusers.models.attention_processor.HunyuanAttnProcessor2_0)), "Need to implement a different attention processor"
        self.attn1.set_processor(QuantHunyuanAttnProcessor2_0())

        if self.attn2 is not None:
            assert isinstance(self.attn2.processor, (diffusers.models.attention_processor.HunyuanAttnProcessor2_0)), "Need to implement a different attention processor"
            self.attn2.set_processor(QuantHunyuanAttnProcessor2_0())

        self.checkpoint = False
        self.attn1.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)

        if self.attn2 is not None:
            self.attn2.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
            self.attn2.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
            self.attn2.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)

        self.attn1.use_act_quant = False
        
        if self.attn2 is not None:
            self.attn2.use_act_quant = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.attn1.use_act_quant = act_quant
        if self.attn2 is not None:
            self.attn2.use_act_quant = act_quant

        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, 
        hidden_states: th.Tensor,
        encoder_hidden_states: Optional[th.Tensor] = None,
        temb: Optional[th.Tensor] = None,
        image_rotary_emb=None,
        skip=None,
                ):
    
        if len(self._forward_hooks) > 0:
            self.hs_cache = hidden_states
            self.ehs_cache = encoder_hidden_states
            self.t_cache = temb
            self.ire_cache = image_rotary_emb
            self.s_cache = skip
        else:
            self.hs_cache = None
            self.ehs_cache = None
            self.t_cache = None
            self.ire_cache = None
            self.s_cache = None
        return checkpoint(self._forward, (hidden_states, encoder_hidden_states, temb, image_rotary_emb, skip), self.parameters(), self.checkpoint)
    
    def _forward(
        self,
        hidden_states: th.Tensor,
        encoder_hidden_states: Optional[th.Tensor] = None,
        temb: Optional[th.Tensor] = None,
        image_rotary_emb=None,
        skip=None,
    ):
        if self.skip_linear is not None:
            cat = th.cat([hidden_states, skip], dim=-1)
            cat = self.skip_norm(cat)
            if self.skip_q:
                hidden_states = self.skip_linear(cat, self.split)
            else:
                hidden_states = self.skip_linear(cat)

        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states, temb)
        attn_output = self.attn1(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states


class QuantDiffRB(BaseQuantBlock):
    def __init__(self, tran,
                 act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.pre_norm = tran.pre_norm
        self.in_channels = tran.in_channels
        self.out_channels = tran.out_channels
        self.use_conv_shortcut = tran.use_conv_shortcut
        self.up = tran.up
        self.down = tran.down
        self.output_scale_factor = tran.output_scale_factor
        self.time_embedding_norm = tran.time_embedding_norm
        self.skip_time_act = tran.skip_time_act
        self.norm1 = tran.norm1
        self.conv1 = tran.conv1
        self.time_emb_proj = tran.time_emb_proj
        self.norm2 = tran.norm2
        self.dropout = tran.dropout
        self.conv2 = tran.conv2
        self.nonlinearity = tran.nonlinearity
        self.upsample = tran.upsample
        self.downsample = tran.downsample
        self.use_in_shortcut = tran.use_in_shortcut
        self.conv_shortcut = tran.conv_shortcut

        self.conv1_q, self.conv_shortcut_q = False, False
        if isinstance(self.conv1, QuantModule) or isinstance(self.conv1, QuantModuleMultiQ):
            self.conv1_q = True
            self.conv1_split = self.out_channels if self.in_channels - self.out_channels > 0 else 0
        if self.conv_shortcut is not None and isinstance(self.conv_shortcut, QuantModule):
            self.conv_shortcut_q = True
            out_c = self.conv_shortcut.org_weight.shape[0]
            self.conv_shortcut_split = out_c if self.in_channels - out_c > 0 else 0

    def forward(self, input_tensor, temb):
            hidden_states = input_tensor

            if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                hidden_states = self.norm1(hidden_states, temb)
            else:
                hidden_states = self.norm1(hidden_states)

            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            if self.conv1_q:
                hidden_states = self.conv1(hidden_states, split=self.conv1_split)
            else:
                hidden_states = self.conv1(hidden_states)

            if self.time_emb_proj is not None:
                if not self.skip_time_act:
                    temb = self.nonlinearity(temb)
                temb = self.time_emb_proj(temb)[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                hidden_states = self.norm2(hidden_states, temb)
            else:
                hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = th.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)

            if self.conv_shortcut is not None:
                if self.conv_shortcut_q:
                    input_tensor = self.conv_shortcut(input_tensor, self.conv_shortcut_split)
                else:
                    input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor


class QuantDiffBTB(BaseQuantBlock):
    def __init__(self, tran: diffusers.models.attention.BasicTransformerBlock,
                 act_quant_params: dict = {}, sm_abit: int = 8):
        super().__init__(act_quant_params)
        self.only_cross_attention = tran.only_cross_attention
        self.use_ada_layer_norm_zero = tran.use_ada_layer_norm_zero
        self.use_ada_layer_norm = tran.use_ada_layer_norm

        self.use_ada_layer_norm_single = tran.use_ada_layer_norm_single # new
        self.use_layer_norm = tran.use_layer_norm # new
        self.use_ada_layer_norm_continuous = tran.use_ada_layer_norm_continuous # new
        self.norm_type = tran.norm_type # new
        self.num_embeds_ada_norm = tran.num_embeds_ada_norm # new
        self.pos_embed = tran.pos_embed # new

        self.norm1 = tran.norm1
        self.attn1 = tran.attn1
        self.norm2 = tran.norm2
        self.attn2 = tran.attn2
        if hasattr(tran, "norm3"):
            self.norm3 = tran.norm3
        else:
            self.norm3 = self.norm2
        self.ff = tran.ff

        if hasattr(tran, "fuser"):
            self.fuser = tran.fuser
        if hasattr(tran, "scale_shift_table"):
            self.scale_shift_table = tran.scale_shift_table

        self._chunk_size = tran._chunk_size
        self._chunk_dim = tran._chunk_dim
        self.set_chunk_feed_forward = tran.set_chunk_feed_forward

        assert isinstance(self.attn1.processor, (diffusers.models.attention_processor.AttnProcessor, diffusers.models.attention_processor.AttnProcessor2_0)), "Need to implement a different attention processor"
        self.attn1.set_processor(QuantAttnProcessor())

        if self.attn2 is not None:
            assert isinstance(self.attn2.processor, (diffusers.models.attention_processor.AttnProcessor, diffusers.models.attention_processor.AttnProcessor2_0)), "Need to implement a different attention processor"
            self.attn2.set_processor(QuantAttnProcessor())

        self.checkpoint = False
        self.attn1.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        self.attn1.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)

        if self.attn2 is not None:
            self.attn2.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
            self.attn2.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
            self.attn2.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)

        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['always_zero'] = True
        self.attn1.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)

        self.attn1.use_act_quant = False
        
        if self.attn2 is not None:
            self.attn2.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)
            self.attn2.use_act_quant = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.attn1.use_act_quant = act_quant
        if self.attn2 is not None:
            self.attn2.use_act_quant = act_quant

        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, 
                hidden_states,
                attention_mask: Optional[th.Tensor] = None,
                encoder_hidden_states: Optional[th.Tensor] = None,
                encoder_attention_mask: Optional[th.Tensor] = None,
                timestep: Optional[th.LongTensor] = None,
                cross_attention_kwargs: Dict[str, Any] = None,
                class_labels: Optional[th.LongTensor] = None,
                added_cond_kwargs: Optional[Dict[str, th.Tensor]] = None):
    
        if len(self._forward_hooks) > 0:
            self.am_cache = attention_mask
            self.ehs_cache = encoder_hidden_states
            self.eam_cache = encoder_attention_mask
            self.ts_cache = timestep
            self.cak_cache = cross_attention_kwargs
            self.class_labels = class_labels
            self.added_cond_kwargs = added_cond_kwargs
        else:
            self.am_cache = None
            self.ehs_cache = None
            self.eam_cache = None
            self.ts_cache = None
            self.cak_cache = None
            self.class_labels = None
            self.added_cond_kwargs = added_cond_kwargs
        return checkpoint(self._forward, (hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, timestep, cross_attention_kwargs, class_labels, added_cond_kwargs), self.parameters(), self.checkpoint)
    
    def _forward(
        self,
        hidden_states,
        attention_mask: Optional[th.FloatTensor] = None,
        encoder_hidden_states: Optional[th.FloatTensor] = None,
        encoder_attention_mask: Optional[th.FloatTensor] = None,
        timestep: Optional[th.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[th.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, th.Tensor]] = None
    ):
        
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        
        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states
