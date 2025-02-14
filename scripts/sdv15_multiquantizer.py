import argparse, os, datetime, yaml
import logging
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
import time
from pytorch_lightning import seed_everything
import torch

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from qdiff import QuantModelMultiQ


def load_model_from_config(config, ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--res",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    # linear quantization configs
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=4,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt",
        help="path to checkpoint of model",
    )
    opt = parser.parse_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = PLMSSampler(model)
    
    setattr(sampler.model.model.diffusion_model, "split", True)

    wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param':  False}
    qnn = QuantModelMultiQ(
        model=sampler.model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
        act_quant_mode="qdiff", sdv15=True)
    qnn.to("cuda")
    qnn.eval()

    qnn.set_quant_state(weight_quant=True, act_quant=False)
    
    print("Generating all quantization configuration parameters. This will take a while and may cause CUDA OOM if there is insufficient VRAM.")
    input_x, input_t, input_c = torch.randn(1, 4, 64, 64).cuda(), torch.randint(0, 1000, (1,)).cuda(), torch.randn(1, 77, 768).cuda()
    sampler.model.model.diffusion_model = qnn
    with torch.no_grad():
        qnn(input_x, input_t, input_c)
            
    sd = qnn.state_dict()
    for k, v in sd.items():
        if "quant_params.mse" in k:
            old_shape = list(v.shape)
            new_shape = [2, v.shape[0] // 2] + old_shape[1:]
            v = v.reshape(new_shape)
            sd[k] = v
    torch.save(sd, "multiquantizers/sdv15_mq.pt")


if __name__ == "__main__":
    main()
