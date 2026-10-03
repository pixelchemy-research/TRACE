import os
import sys

sys.path.append("third_party/")
sys.path.append("third_party/StableCascade/")

import argparse
import copy

import traceback
import random
import numpy as np

import yaml
import torch
import torchvision.transforms as T
import PIL.Image
from tqdm import tqdm
from accelerate.utils import set_module_tensor_to_device

from core.utils import load_or_fail
from train import WurstCoreB
from trace import TRACE
from stage_c_trace import StageCTRACE
from utils.stage_c_utils import WurstCoreCTRACE
from utils.image_utils import save_images, calculate_latent_sizes, resize_image
from utils.offload_utils import (
    move_stage_c_all_to_gpu,
    move_stage_c_all_to_cpu,
    move_stage_c_condition_models_to_cpu,
    move_stage_b_all_to_gpu,
    move_stage_b_all_to_cpu,
    hard_clear_cuda,
    image_path_from_name,
    collect_names,
    make_pairs,
)

from gdf.schedulers import CosineSchedule
from gdf import VPScaler, CosineTNoiseCond, DDPMSampler, AdaptiveLossWeight
from gdf.targets import EpsilonTarget


def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--content_names", type=str, nargs="+", default=None)
    parser.add_argument("--style_names", type=str, nargs="+", default=None)

    parser.add_argument("--pair_mode", type=str, default="zip", choices=["all", "zip"])

    parser.add_argument("--content_dir", type=str, default="data/content")
    parser.add_argument("--style_dir", type=str, default="data/style")
    parser.add_argument("--out_dir", type=str, default="./samples")
    parser.add_argument("--output_prefix", type=str, default="TRACE")

    parser.add_argument("--content_ext", type=str, default="jpg")
    parser.add_argument("--style_ext", type=str, default="png")

    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--uncertainty_M", type=int, default=4)
    parser.add_argument("--noise_level", type=float, default=0.7)

    parser.add_argument("--stage_c_cfg", type=float, default=4.0)
    parser.add_argument("--stage_c_shift", type=float, default=2.0)
    parser.add_argument("--stage_c_total_steps", type=int, default=20)
    parser.add_argument("--stage_b_cfg", type=float, default=1.1)
    parser.add_argument("--stage_b_shift", type=float, default=1.0)
    parser.add_argument("--stage_b_timesteps", type=int, default=10)

    parser.add_argument("--low_vram", action="store_true", default=True)

    parser.add_argument("--clip_content_basis_path", type=str, default="./decomposition/decomposition.pt")

    return parser


def load_everything_once(args):
    device_c = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_b = device_c

    config_file = "third_party/StableCascade/configs/inference/stage_c_3b.yaml"
    with open(config_file, "r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)

    core = WurstCoreCTRACE(config_dict=loaded_config, device=device_c, training=False)

    config_file_b = "third_party/StableCascade/configs/inference/stage_b_3b.yaml"
    with open(config_file_b, "r", encoding="utf-8") as file:
        loaded_config_b = yaml.safe_load(file)

    core_b = WurstCoreB(config_dict=loaded_config_b, device=device_b, training=False)

    extras_pre = core.setup_extras_pre()

    gdf_trace = TRACE(
        schedule=CosineSchedule(clamp_range=[0.0001, 0.9999]),
        input_scaler=VPScaler(),
        target=EpsilonTarget(),
        noise_cond=CosineTNoiseCond(),
        loss_weight=AdaptiveLossWeight(),
    )
    sampling_configs = {
        "cfg": 5,
        "sampler": DDPMSampler(gdf_trace),
        "shift": 1,
        "timesteps": 20,
    }

    extras = core.Extras(
        gdf=gdf_trace,
        sampling_configs=sampling_configs,
        transforms=extras_pre.transforms,
        effnet_preprocess=extras_pre.effnet_preprocess,
        clip_preprocess=extras_pre.clip_preprocess,
    )

    models = core.setup_models(extras)
    models.generator.eval().requires_grad_(False)

    extras_b = core_b.setup_extras_pre()
    models_b = core_b.setup_models(extras_b, skip_clip=True)

    models_b = WurstCoreB.Models(
        **{
            **models_b.to_dict(),
            "tokenizer": models.tokenizer,
            "text_model": copy.deepcopy(models.text_model),
        }
    )
    models_b.generator.eval().requires_grad_(False)

    generator_trace = StageCTRACE()
    for param_name, param in load_or_fail(
        core.config.generator_checkpoint_path
    ).items():
        set_module_tensor_to_device(generator_trace, param_name, "cpu", value=param)

    generator_trace = generator_trace.to(getattr(torch, core.config.dtype)).to(device_c)
    generator_trace = core.load_model(generator_trace, "generator")

    models_trace = core.Models(
        effnet=models.effnet,
        previewer=models.previewer,
        generator=generator_trace,
        generator_ema=models.generator_ema,
        tokenizer=models.tokenizer,
        text_model=models.text_model,
        image_model=models.image_model,
    )
    models_trace.generator.eval().requires_grad_(False)

    core.load_clip_content_basis(args.clip_content_basis_path)

    move_stage_c_all_to_cpu(models_trace)
    move_stage_b_all_to_cpu(models_b)
    hard_clear_cuda()

    state = {
        "device_c": device_c,
        "device_b": device_b,
        "core": core,
        "core_b": core_b,
        "extras": extras,
        "extras_b": extras_b,
        "models": models,
        "models_b": models_b,
        "models_trace": models_trace,
    }
    return state


def run_one_pair(content_name, style_name, args, state):
    device_c = state["device_c"]
    device_b = state["device_b"]
    core = state["core"]
    core_b = state["core_b"]
    extras = state["extras"]
    extras_b = state["extras_b"]
    models = state["models"]
    models_b = state["models_b"]
    models_trace = state["models_trace"]


    ref_style_file = image_path_from_name(args.style_dir, style_name, args.style_ext)
    ref_sub_file = image_path_from_name(
        args.content_dir, content_name, args.content_ext
    )

    if not os.path.exists(ref_style_file):
        raise FileNotFoundError(f"Style image not found: {ref_style_file}")
    if not os.path.exists(ref_sub_file):
        raise FileNotFoundError(f"Content image not found: {ref_sub_file}")

    content_stem = os.path.splitext(os.path.basename(content_name))[0]
    style_stem = os.path.splitext(os.path.basename(style_name))[0]
    name_exp = f"{args.output_prefix}_{content_stem}_{style_stem}"

    batch_size = args.batch_size
    height, width = args.height, args.width
    noise_level = args.noise_level

    stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(
        height, width, batch_size=batch_size
    )

    ref_style_cpu = (
        resize_image(PIL.Image.open(ref_style_file).convert("RGB"))
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
        .contiguous()
    )
    ref_images_cpu = (
        resize_image(PIL.Image.open(ref_sub_file).convert("RGB"))
        .unsqueeze(0)
        .expand(batch_size, -1, -1, -1)
        .contiguous()
    )

    move_stage_b_all_to_cpu(models_b)
    move_stage_c_all_to_gpu(models_trace, device_c)
    hard_clear_cuda()

    sampled_c_cpu = None

    with torch.enable_grad():
        images = ref_images_cpu.to(device_c)
        batch = {"images": images}

        effnet_latents = core.encode_latents(batch, models, extras)
        t = torch.ones(effnet_latents.size(0), device=device_c) * noise_level
        noised = extras.gdf.diffuse(effnet_latents, t=t)[0]

        extras.sampling_configs["cfg"] = args.stage_c_cfg
        extras.sampling_configs["shift"] = args.stage_c_shift
        extras.sampling_configs["timesteps"] = int(
            args.stage_c_total_steps * noise_level
        )
        extras.sampling_configs["t_start"] = noise_level
        extras.sampling_configs["x_init"] = noised

        extras_b.sampling_configs["cfg"] = args.stage_b_cfg
        extras_b.sampling_configs["shift"] = args.stage_b_shift
        extras_b.sampling_configs["timesteps"] = args.stage_b_timesteps
        extras_b.sampling_configs["t_start"] = 1.0

        ref_style = ref_style_cpu.to(device_c)
        ref_images = ref_images_cpu.to(device_c)

        batch_c = {
            "captions": [""] * batch_size,
            "style": ref_style,
            "images": ref_images,
        }

        x0_forward = models_trace.effnet(extras.effnet_preprocess(ref_images))
        x0_style_forward = models_trace.effnet(extras.effnet_preprocess(ref_style))

        conditions = core.get_conditions(
            batch_c,
            models_trace,
            extras,
            is_eval=True,
            is_unconditional=False,
            eval_image_embeds=True,
            eval_subject_style=True,
            eval_csd=False,
        )
        unconditions = core.get_conditions(
            batch_c,
            models_trace,
            extras,
            is_eval=True,
            is_unconditional=True,
            eval_image_embeds=False,
            eval_subject_style=True,
        )

        if args.low_vram:
            move_stage_c_condition_models_to_cpu(models_trace)
            hard_clear_cuda()

        if getattr(models_trace, "previewer", None) is not None:
            pass

        x0_forward = x0_forward.to(device_c)
        x0_style_forward = x0_style_forward.to(device_c)

        sampling_c = extras.gdf.sample(
            models_trace.generator,
            conditions,
            stage_c_latent_shape,
            unconditions,
            device=device_c,
            **extras.sampling_configs,
            x0_style_forward=x0_style_forward,
            num_iter=2,
            extras=extras,
            models=models_trace,
            uncertainty_M=args.uncertainty_M,
        )

        sampled_c = None
        for sampled_c, _, _ in tqdm(
            sampling_c,
            total=extras.sampling_configs["timesteps"],
            desc=f"Stage C {content_stem}-{style_stem}",
        ):
            pass

        if sampled_c is None:
            raise RuntimeError("Stage C sampling did not return any sample.")

        sampled_c_cpu = sampled_c.detach().to("cpu")

        del sampled_c
        del conditions, unconditions
        del x0_forward, x0_style_forward
        del ref_style, ref_images
        del images, batch, batch_c
        del noised, effnet_latents

    move_stage_c_all_to_cpu(models_trace)
    hard_clear_cuda()

    move_stage_b_all_to_gpu(models_b, device_b)
    hard_clear_cuda()

    ref_style_b = ref_style_cpu.to(device_b)
    ref_images_b = ref_images_cpu.to(device_b)

    batch_b = {
        "captions": [""] * batch_size,
        "style": ref_style_b,
        "images": ref_images_b,
    }

    conditions_b = core_b.get_conditions(
        batch_b,
        models_b,
        extras_b,
        is_eval=True,
        is_unconditional=False,
    )
    unconditions_b = core_b.get_conditions(
        batch_b,
        models_b,
        extras_b,
        is_eval=True,
        is_unconditional=True,
    )

    sampled_c = sampled_c_cpu.to(device_b)

    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
    ):
        conditions_b["effnet"] = sampled_c
        unconditions_b["effnet"] = torch.zeros_like(sampled_c, device=device_b)

        sampling_b = extras_b.gdf.sample(
            models_b.generator,
            conditions_b,
            stage_b_latent_shape,
            unconditions_b,
            device=device_b,
            **extras_b.sampling_configs,
        )

        sampled_b = None
        for sampled_b, _, _ in tqdm(
            sampling_b,
            total=extras_b.sampling_configs["timesteps"],
            desc=f"Stage B {content_stem}-{style_stem}",
        ):
            pass

        if sampled_b is None:
            raise RuntimeError("Stage B sampling did not return any sample.")

        sampled = models_b.stage_a.decode(sampled_b).float()

    sampled = torch.cat([sampled.cpu()], dim=0)

    out_dir = os.path.join(args.out_dir, args.output_prefix)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name_exp}.png")
    save_images(sampled, out_path)

    del conditions_b, unconditions_b
    del sampled_c, sampled_c_cpu, sampled_b, sampled
    del ref_style_b, ref_images_b, batch_b
    del ref_style_cpu, ref_images_cpu

    move_stage_b_all_to_cpu(models_b)
    hard_clear_cuda()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    set_seed(42)
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()

    content_names, style_names = collect_names(args)
    pairs = make_pairs(content_names, style_names, args.pair_mode)

    state = load_everything_once(args)

    state["extras"].gdf.setup_csd(device=state["device_c"])

    failed_pairs = []
    for _, (content_name, style_name) in enumerate(pairs, start=1):
        content_stem = os.path.splitext(os.path.basename(content_name))[0]
        style_stem = os.path.splitext(os.path.basename(style_name))[0]
        out_path = os.path.join(
            args.out_dir,
            args.output_prefix,
            f"{args.output_prefix}_{content_stem}_{style_stem}.png",
        )
        if os.path.exists(out_path):
            print(f"Skip existing: {out_path}")
            continue

        try:
            run_one_pair(content_name, style_name, args, state)
        except Exception as e:
            print(
                f"Error processing content='{content_name}', style='{style_name}': {e}"
            )
            traceback.print_exc()
            failed_pairs.append((content_name, style_name, str(e)))
            hard_clear_cuda()
            continue


if __name__ == "__main__":
    main()
