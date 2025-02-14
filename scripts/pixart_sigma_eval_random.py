import argparse, os, datetime, yaml
import logging
from pytorch_lightning import seed_everything
import torch
from qdiff import QuantModelMultiQ
import pickle
import glob
from tqdm import tqdm
from cleanfid import fid
from constants import COCO_2017_PROMPTS, COCO_2017_IMAGES
from bops.annotate_bops_sigma import annotate_bops


logger = logging.getLogger(__name__)
FP_SIZE = 1220456448
FP_PREC = 16


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="sigma_random_seed42"
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
    parser.add_argument("--multi_chkpt", type=str, default="multiquantizers/sigma_mq.pt")
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

    from diffusers import PixArtSigmaPipeline
    model = PixArtSigmaPipeline.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

    wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param':  False}
    qnn = QuantModelMultiQ(
        model=model.transformer, weight_quant_params=wq_params, act_quant_params=aq_params,
        act_quant_mode="qdiff")
    qnn.to("cuda")
    qnn.eval()

    model.transformer = qnn
    qnn.set_quant_state(weight_quant=False, act_quant=False)
    with torch.no_grad():
        model(prompt=["weight calibration prompt"], height=1024, width=1024)

    if opt.multi_chkpt:
        qnn.load_mq_state_dicts(filepath=opt.multi_chkpt)
    qnn.set_quant_state(weight_quant=True, act_quant=False)

    with torch.no_grad():
        model(prompt=["weight calibration prompt"], height=1024, width=1024)
    
    import json
    with open(COCO_2017_PROMPTS) as f:
        captions = json.load(f)
        caption_list = captions['annotations'][:1000]
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
        for i in tqdm(range(0, len(data), batch_size), desc="data"):
            generator = torch.manual_seed(42) # Meaning of Life, the Universe and Everything
            prompts = data[i:i + batch_size]
            prompts = [p[0] for p in prompts]

            image = model(prompt=prompts, height=1024, width=1024).images

            for j, img in enumerate(image):
                img.save(os.path.join(sample_path, f"{i+j}.png"))

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
