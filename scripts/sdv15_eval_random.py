import argparse, os, datetime, yaml
import logging
from pytorch_lightning import seed_everything
import torch
from qdiff import QuantModelMultiQ
import pickle
import glob
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from cleanfid import fid
from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from omegaconf import OmegaConf
from constants import COCO_2017_PROMPTS, COCO_2017_IMAGES
from bops.annotate_bops_sdv15 import annotate_bops


logger = logging.getLogger(__name__)
FP_SIZE = 3436308480
FP_PREC = 32


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
        default="sdv15_random_seed42"
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
        default=4,
        help="int bit for weight quantization",
    )

    # qdiff specific configs
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument("--multi_chkpt", type=str, default="multiquantizers/sdv15_mq.pt")
    parser.add_argument("--num_archs", type=int, default=500)
    parser.add_argument("--a_l", type=float, default=0.0)
    parser.add_argument("--a_u", type=float, default=1.0)
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
    qnn.to("cuda")
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
    
    import json
    with open(COCO_2017_PROMPTS) as f:
        captions = json.load(f)
        caption_list = captions['annotations'][:10]
        data = []
        for c in caption_list:
            data.append([c['caption']])

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    # write config out
    sampling_file = os.path.join(outpath, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)

    arch_list = []
    for n in tqdm(range(opt.num_archs), desc="configs"):
        qnn.set_random_config(a_l=opt.a_l, a_u=opt.a_u)
        arch_dict = {'config': qnn.get_current_quant_config(skip_act=True, include_stats=True)}
        arch_dict['quantized_size'] = qnn.quantify_model_savings()
        arch_dict['quantized_error'] = sum([m[-1] for m in arch_dict['config'].values()])
        batch_size = 1
        base_count = 0
        for i in tqdm(range(0, len(data), batch_size), desc="data"):
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            prompts = data[i:i + batch_size]
            prompts = [p[0] for p in prompts]
            c = model.get_learned_conditioning(prompts)
            shape = [4, 512 // 8, 512 // 8]
            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                            conditioning=c,
                                            batch_size=batch_size,
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

        arch_dict['FID-GT-1k'] = fid.compute_fid(f'{outpath}/samples/', COCO_2017_IMAGES)
        arch_dict['avg_bits'] = FP_PREC * arch_dict['quantized_size'] / FP_SIZE
        arch_dict['bops'] = annotate_bops(arch_dict)

        arch_list.append(arch_dict)

        with open(os.sep.join([outpath, "quant_cache.pkl"]), "wb") as f:
            pickle.dump(arch_list, f, protocol=4)

        for f in glob.glob(outpath + "/samples/*.png"):
            os.remove(f)

    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
                 f" \nEnjoy.")


if __name__ == "__main__":
    main()
