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


logger = logging.getLogger(__name__)


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
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="sdv15_sample"
    )
    parser.add_argument(
        "--res",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
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
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument("--multi_chkpt", type=str, default='multiquantizers/sdv15_mq.pt')
    parser.add_argument("--q_config", type=str, default='quant_configs/sdv15_40.pkl')
    parser.add_argument("--prompt", type=str, default="a lion riding a bike in Paris")
    parser.add_argument("--n_imgs", type=int, default=1)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = os.path.join(opt.outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(outpath)
    
    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

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
    qnn.cuda()
    qnn.eval()

    input_x, input_t, input_c = torch.randn(1, 4, 64, 64).cuda(), torch.randint(0, 1000, (1,)).cuda(), torch.randn(1, 77, 768).cuda()
    sampler.model.model.diffusion_model = qnn
    qnn.set_quant_state(weight_quant=False, act_quant=False)
    with torch.no_grad():
        qnn(input_x, input_t, input_c)
        
    if opt.multi_chkpt:
        qnn.load_mq_state_dicts(filepath=opt.multi_chkpt)
    
    qnn.set_quant_state(weight_quant=True, act_quant=False)
    with torch.no_grad():
        qnn(input_x, input_t, input_c)

    if opt.q_config is not None:
        assert opt.multi_chkpt is not None
        qnn.load_quant_config(opt.q_config)
    qnn.quantify_model_savings()

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # write config out
    sampling_file = os.path.join(outpath, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)

    torch.manual_seed(42) # Meaning of Life, the Universe and Everything
    with torch.no_grad():
        with model.ema_scope():
            tic = time.time()
            all_samples = list()
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_imgs * [""])
            prompts = [opt.prompt] * opt.n_imgs
            c = model.get_learned_conditioning(prompts)
            shape = [4, opt.res // opt.f, opt.res // opt.f]
            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                conditioning=c,
                                                batch_size=opt.n_imgs,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                x_T=None)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

            x_checked_image = x_samples_ddim

            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

            for x_sample in x_checked_image_torch:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                base_count += 1

            toc = time.time()

    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
