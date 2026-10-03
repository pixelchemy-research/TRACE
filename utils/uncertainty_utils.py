import torch
from typing import Tuple, Callable


def normalize_uncertainty(U: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    original_dtype = U.dtype
    U_fp32 = U.float()
    B = U_fp32.shape[0]
    flat = U_fp32.flatten(1)

    tau = torch.quantile(flat, 0.90, dim=1).view(B, 1, 1, 1)

    tau = tau.clamp_min(eps)
    U_norm = torch.tanh(U_fp32 / tau)

    return U_norm.to(original_dtype)


def alpha_bar_from_logsnr(logsnr_t, x_like: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(logsnr_t):
        logsnr_t = torch.tensor(logsnr_t, device=x_like.device, dtype=x_like.dtype)

    alpha_bar = torch.sigmoid(logsnr_t)

    if alpha_bar.dim() == 0:
        alpha_bar = alpha_bar.view(1, 1, 1, 1)
    elif alpha_bar.dim() == 1:
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)

    return alpha_bar.to(device=x_like.device, dtype=x_like.dtype)


@torch.no_grad()
def estimate_pixelwise_uncertainty(
    predict_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_t: torch.Tensor,
    noise_cond,
    logsnr_t,
    undiffuse_fn,
    pred_t: torch.Tensor = None,
    M: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B = x_t.shape[0]

    if pred_t is None:
        pred_t = predict_fn(x_t, noise_cond)

    x0_hat, _ = undiffuse_fn(
        x_t,
        logsnr_t,
        pred_t,
    )

    alpha_bar = alpha_bar_from_logsnr(logsnr_t, x_t)

    sqrt_ab = torch.sqrt(alpha_bar)
    sqrt_1mab = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))

    eps_rand = torch.randn(
        (M,) + x0_hat.shape,
        device=x0_hat.device,
        dtype=x0_hat.dtype,
    )

    x0_hat_m = x0_hat.unsqueeze(0)
    sqrt_ab_m = sqrt_ab.unsqueeze(0)
    sqrt_1mab_m = sqrt_1mab.unsqueeze(0)

    x_t_pert = sqrt_ab_m * x0_hat_m + sqrt_1mab_m * eps_rand

    x_t_pert = x_t_pert.flatten(0, 1)

    def repeat_batch(obj):
        if torch.is_tensor(obj):
            if obj.ndim > 0 and obj.shape[0] == B:
                return (
                    obj.unsqueeze(0)
                    .expand(M, *obj.shape)
                    .reshape(M * B, *obj.shape[1:])
                )
            return obj

        elif isinstance(obj, dict):
            return {k: repeat_batch(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return tuple(repeat_batch(v) for v in obj)

        elif isinstance(obj, list):
            return [repeat_batch(v) for v in obj]

        return obj

    noise_cond_m = repeat_batch(noise_cond)

    pred_pert = predict_fn(
        x_t_pert,
        noise_cond_m,
    )

    if torch.is_tensor(logsnr_t):
        if logsnr_t.ndim > 0 and logsnr_t.shape[0] == B:
            logsnr_t_m = (
                logsnr_t.unsqueeze(0)
                .expand(M, *logsnr_t.shape)
                .reshape(M * B, *logsnr_t.shape[1:])
            )
        else:
            logsnr_t_m = logsnr_t
    else:
        logsnr_t_m = logsnr_t

    _, eps_pert = undiffuse_fn(x_t_pert, logsnr_t_m, pred_pert)

    eps_pert = eps_pert.reshape( M, B, *eps_pert.shape[1:])

    U_t = eps_pert.var( dim=0, unbiased=False)

    return U_t