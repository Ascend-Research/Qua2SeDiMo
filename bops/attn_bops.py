BOP_LUT = {
    "kmeans-4": 16,
    "kmeans-3": 16,
    "kmeans_all-4": 16,
    "kmeans_all-3": 16,
    "mse-4": 4,
    "mse-3": 3
}

ACT_FP_BIT = 16
ACT_QUANT_BIT = 16

ALPHA_SA_MACS = 1.21 * 1e9
ALPHA_CA_MACS = 141.56 * 1e6

SIGMA_SA_MACS = 4.836 * 1e9
SIGMA_CA_MACS = 1.426 * 1e9

HUNYUAN_SA_MACS = 23.62 * 1e9
HUNYUAN_CA_MACS = 1.92 * 1e9


SDXL_SA_64_MACS = 671.09 * 1e6
SDXL_CA_64_MACS = 50.46 * 1e6
SDXL_SA_32_MACS = 83.89 * 1e6
SDXL_CA_32_MACS = 83.89 * 1e6


SDV15_SA_64_MACS = 335.56 * 1e6
SDV15_CA_64_MACS = 25.23 * 1e6
SDV15_SA_32_MACS = 41.94 * 1e6
SDV15_CA_32_MACS = 12.62 * 1e6
SDV15_SA_16_MACS = 5.24 * 1e6
SDV15_CA_16_MACS = 6.31 * 1e6


def attn_bops_dit(prec=16, n_layers=28):
    sa_qk_bops = ALPHA_SA_MACS * prec * prec
    sa_vout_bops = ALPHA_SA_MACS * prec * 16
    bops_per_layer = sa_qk_bops + sa_vout_bops
    return bops_per_layer * n_layers


def attn_bops_pixart_alpha(prec=16, n_layers=28):
    sa_qk_bops = ALPHA_SA_MACS * prec * prec
    sa_vout_bops = ALPHA_SA_MACS * prec * 16
    ca_qk_bops = ALPHA_CA_MACS * prec * prec
    ca_vout_bops = ALPHA_CA_MACS * prec * 16
    bops_per_layer = sa_qk_bops + sa_vout_bops + ca_qk_bops + ca_vout_bops
    return bops_per_layer * n_layers


def attn_bops_pixart_sigma(prec=16, n_layers=28):
    sa_qk_bops = SIGMA_SA_MACS * prec * prec
    sa_vout_bops = SIGMA_SA_MACS * prec * 16
    ca_qk_bops = SIGMA_CA_MACS * prec * prec
    ca_vout_bops = SIGMA_CA_MACS * prec * 16
    bops_per_layer = sa_qk_bops + sa_vout_bops + ca_qk_bops + ca_vout_bops
    return bops_per_layer * n_layers


def attn_bops_hunyuan(prec=16, n_layers=40):
    sa_qk_bops = HUNYUAN_SA_MACS * prec * prec
    sa_vout_bops = HUNYUAN_SA_MACS * prec * 16
    ca_qk_bops = HUNYUAN_CA_MACS * prec * prec
    ca_vout_bops = HUNYUAN_CA_MACS * prec * 16
    bops_per_layer = sa_qk_bops + sa_vout_bops + ca_qk_bops + ca_vout_bops
    return bops_per_layer * n_layers


def attn_bops_sdxl(prec=16):
    sa64_qk_bops = SDXL_SA_64_MACS * prec * prec
    sa64_vout_bops = SDXL_SA_64_MACS * prec * 16
    ca64_qk_bops = SDXL_CA_64_MACS * prec * prec
    ca64_vout_bops = SDXL_CA_64_MACS * prec * 16
    
    sa32_qk_bops = SDXL_SA_32_MACS * prec * prec
    sa32_vout_bops = SDXL_SA_32_MACS * prec * 16
    ca32_qk_bops = SDXL_CA_32_MACS * prec * prec
    ca32_vout_bops = SDXL_CA_32_MACS * prec * 16
    bops_per_layer = (sa64_qk_bops + sa64_vout_bops + ca64_qk_bops + ca64_vout_bops) * 10 + (sa32_qk_bops + sa32_vout_bops + ca32_qk_bops + ca32_vout_bops) * 60
    return bops_per_layer


def attn_bops_sdv15(prec=16):
    sa64_qk_bops = SDV15_SA_64_MACS * prec * prec
    sa64_vout_bops = SDV15_SA_64_MACS * prec * 16
    ca64_qk_bops = SDV15_CA_64_MACS * prec * prec
    ca64_vout_bops = SDV15_CA_64_MACS * prec * 16
    
    sa32_qk_bops = SDV15_SA_32_MACS * prec * prec
    sa32_vout_bops = SDV15_SA_32_MACS * prec * 16
    ca32_qk_bops = SDV15_CA_32_MACS * prec * prec
    ca32_vout_bops = SDV15_CA_32_MACS * prec * 16
    
    sa16_qk_bops = SDV15_SA_16_MACS * prec * prec
    sa16_vout_bops = SDV15_SA_16_MACS * prec * 16
    ca16_qk_bops = SDV15_CA_16_MACS * prec * prec
    ca16_vout_bops = SDV15_CA_16_MACS * prec * 16
    
    bops_per_layer = (sa64_qk_bops + sa64_vout_bops + ca64_qk_bops + ca64_vout_bops) * 4 + (sa32_qk_bops + sa32_vout_bops + ca32_qk_bops + ca32_vout_bops) * 4 + (sa16_qk_bops + sa16_vout_bops + ca16_qk_bops + ca16_vout_bops) * 8
    return bops_per_layer