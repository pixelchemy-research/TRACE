import gc
import os
import torch


def collect_names(args):
    content_names = []
    style_names = []

    if args.content_names:
        content_names.extend(args.content_names)
    if args.style_names:
        style_names.extend(args.style_names)

    if not content_names:
        raise ValueError("Missing content names. Use --content_names.")
    if not style_names:
        raise ValueError("Missing style names. Use --style_names.")

    if args.pair_mode == "zip" and len(content_names) != len(style_names):
        raise ValueError(
            f"pair_mode='zip' requires same number of content/style names, "
            f"got {len(content_names)} content and {len(style_names)} style."
        )

    return content_names, style_names


def make_pairs(content_names, style_names, pair_mode):
    if pair_mode == "zip":
        return list(zip(content_names, style_names))

    pairs = []
    for content_name in content_names:
        for style_name in style_names:
            pairs.append((content_name, style_name))
    return pairs


def image_path_from_name(root_dir, name, default_ext):
    has_ext = os.path.splitext(name)[1] != ""
    has_dir = os.path.dirname(name) != ""

    if has_ext or has_dir:
        return name
    return os.path.join(root_dir, f"{name}.{default_ext}")


def hard_clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def move_stage_c_all_to_gpu(models_trace, device):
    if getattr(models_trace, "generator", None) is not None:
        models_trace.generator.to(device)
    if getattr(models_trace, "effnet", None) is not None:
        models_trace.effnet.to(device)
    if getattr(models_trace, "text_model", None) is not None:
        models_trace.text_model.to(device)
    if getattr(models_trace, "image_model", None) is not None:
        models_trace.image_model.to(device)
    if getattr(models_trace, "previewer", None) is not None:
        models_trace.previewer.to(device)


def move_stage_c_all_to_cpu(models_trace):
    if getattr(models_trace, "generator", None) is not None:
        models_trace.generator.to("cpu")
    if getattr(models_trace, "effnet", None) is not None:
        models_trace.effnet.to("cpu")
    if getattr(models_trace, "text_model", None) is not None:
        models_trace.text_model.to("cpu")
    if getattr(models_trace, "image_model", None) is not None:
        models_trace.image_model.to("cpu")
    if getattr(models_trace, "previewer", None) is not None:
        models_trace.previewer.to("cpu")


def move_stage_c_condition_models_to_gpu(models_trace, device):
    if getattr(models_trace, "effnet", None) is not None:
        models_trace.effnet.to(device)
    if getattr(models_trace, "text_model", None) is not None:
        models_trace.text_model.to(device)
    if getattr(models_trace, "image_model", None) is not None:
        models_trace.image_model.to(device)


def move_stage_c_condition_models_to_cpu(models_trace):
    if getattr(models_trace, "effnet", None) is not None:
        models_trace.effnet.to("cpu")
    if getattr(models_trace, "text_model", None) is not None:
        models_trace.text_model.to("cpu")
    if getattr(models_trace, "image_model", None) is not None:
        models_trace.image_model.to("cpu")


def move_stage_b_all_to_gpu(models_b, device):
    if getattr(models_b, "generator", None) is not None:
        models_b.generator.to(device, dtype=torch.bfloat16)
    if getattr(models_b, "stage_a", None) is not None:
        models_b.stage_a.to(device)
    if getattr(models_b, "text_model", None) is not None:
        models_b.text_model.to(device)


def move_stage_b_all_to_cpu(models_b):
    if getattr(models_b, "generator", None) is not None:
        models_b.generator.to("cpu")
    if getattr(models_b, "stage_a", None) is not None:
        models_b.stage_a.to("cpu")
    if getattr(models_b, "text_model", None) is not None:
        models_b.text_model.to("cpu")