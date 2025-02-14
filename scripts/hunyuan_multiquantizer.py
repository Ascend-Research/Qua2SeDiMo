import argparse
from pytorch_lightning import seed_everything
import torch
from qdiff import QuantModelMultiQ
from diffusers import HunyuanDiTPipeline


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--res",
        type=int,
        default=1024,
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
    opt = parser.parse_args()

    seed_everything(opt.seed)

    model = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16).to("cuda")

    wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param':  False}
    qnn = QuantModelMultiQ(
        model=model.transformer, weight_quant_params=wq_params, act_quant_params=aq_params,
        act_quant_mode="qdiff")
    qnn.to("cuda")
    qnn.eval()

    qnn.set_quant_state(weight_quant=True, act_quant=False)
    model.transformer = qnn
    
    print("Generating all quantization configuration parameters. This will take a while and may cause CUDA OOM if there is insufficient VRAM.")
    with torch.no_grad():
        model(prompt=["weight calibration prompt"], height=opt.res, width=opt.res)
            
    sd = qnn.state_dict()
    for k, v in sd.items():
        if "quant_params.mse" in k:
            old_shape = list(v.shape)
            new_shape = [2, v.shape[0] // 2] + old_shape[1:]
            v = v.reshape(new_shape)
            sd[k] = v
    torch.save(sd, "multiquantizers/hunyuan_mq.pt")


if __name__ == "__main__":
    main()
