import argparse, os, datetime, yaml
import logging
from pytorch_lightning import seed_everything
import torch
from qdiff import QuantModelMultiQ


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="alpha_sample"
    )
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
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=4,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument("--multi_chkpt", type=str, default='multiquantizers/alpha_mq.pt')
    parser.add_argument("--q_config", type=str, default='quant_configs/alpha_40.pkl')
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

    from diffusers import DiffusionPipeline
    model = DiffusionPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", torch_dtype=torch.float16).to("cuda")

    wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param':  opt.quant_act, 'online_act_quant': True}
    qnn = QuantModelMultiQ(
        model=model.transformer, weight_quant_params=wq_params, act_quant_params=aq_params,
        act_quant_mode="qdiff")
    qnn.to("cuda")
    qnn.eval()

    model.transformer = qnn
    qnn.set_quant_state(weight_quant=False, act_quant=False)
    with torch.no_grad():
        model(prompt=["weight calibration prompt"], height=opt.res, width=opt.res)

    if opt.multi_chkpt:
        qnn.load_mq_state_dicts(filepath=opt.multi_chkpt)
    qnn.set_quant_state(weight_quant=True, act_quant=False)

    with torch.no_grad():
        model(prompt=["weight calibration prompt"], height=opt.res, width=opt.res)

    if opt.q_config is not None:
        assert opt.multi_chkpt is not None
        qnn.load_quant_config(opt.q_config)

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    # write config out
    sampling_file = os.path.join(outpath, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)

    batch_size = opt.n_imgs
    torch.manual_seed(42) # Meaning of Life, the Universe and Everything
    prompts = [opt.prompt] * opt.n_imgs
    image = model(prompt=prompts,
                    height=opt.res, width=opt.res).images

    for i, img in enumerate(image):
        img.save(os.path.join(sample_path, f"{i}.png"))

    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
