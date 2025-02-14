import pickle
from bops.attn_bops import attn_bops_hunyuan, BOP_LUT, ACT_QUANT_BIT


with open("macs/hunyuan_1024.pkl", "rb") as f:
    macs_dict = pickle.load(f)


def annotate_bops(d):
    
    bops_quant = 0
    for k in macs_dict.keys():
        quant_k = k + ".weight_quantizer"
        if quant_k in d['config'].keys():
            # bops = MACs * b_w * b_a
            bops_quant += macs_dict[k] * BOP_LUT[d['config'][quant_k][0]] * ACT_QUANT_BIT
        elif macs_dict[k] > 0:
            # LayerNorm - MACs * 32 (they are always computed in 32-bit precision)
            if "norm" in k:
                bops_quant += macs_dict[k] * 32
            # GELU - MACs * 16
            else:
                bops_quant += macs_dict[k] * 16
    bops_quant += attn_bops_hunyuan(ACT_QUANT_BIT)
                
    return bops_quant // 1e12
