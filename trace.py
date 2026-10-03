from typing import Any, Dict, Tuple
from gdf import BaseSchedule, DDIMSampler, DDPMSampler, GDF, SimpleSampler
import torch
import torchvision.transforms as T
from train import WurstCoreC
from utils.stage_c_utils import setup_csd
from utils.uncertainty_utils import estimate_pixelwise_uncertainty, normalize_uncertainty

transform = T.ToPILImage()


class TRACE(GDF):
    def setup_csd(self, device):
        self.csd_model = setup_csd(device=device)
        self.cosine_loss = torch.nn.CosineSimilarity(dim=1)

    def sample(
        self,
        model: torch.nn.Module,
        model_inputs: Dict[str, Any],
        shape: Tuple,
        unconditional_inputs: Dict[str, Any] = None,
        sampler: SimpleSampler = None,
        schedule: BaseSchedule = None,
        t_start: float = 1.0,
        t_end: float = 0.0,
        timesteps: float = 20,
        x_init: torch.Tensor = None,
        cfg: float = 3.0,
        cfg_t_stop: int = None,
        cfg_t_start: int = None,
        cfg_rho: float = 0.7,
        sampler_params: Dict[str, Any] = None,
        shift: int = 1,
        device: str = "cpu",
        x0_style_forward: torch.Tensor = None,
        num_iter: int = 2,
        eta: float = 1e-1,
        tau: int = 20,
        lam_style: float = 3.0,
        models: WurstCoreC.Models = None,
        extras: WurstCoreC.Extras = None,
        use_ddim_sampler: bool = False,
        uncertainty_M: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        sampler_params = {} if sampler_params is None else sampler_params
        if sampler is None:
            sampler = DDPMSampler(self)
        if use_ddim_sampler:
            sampler = DDIMSampler(self)
        r_range = torch.linspace(t_start, t_end, timesteps + 1)
        schedule = self.schedule if schedule is None else schedule
        logSNR_range = (
            schedule(r_range, shift=shift)[:, None]
            .expand(-1, shape[0] if x_init is None else x_init.size(0))
            .to(device)
        )

        x = sampler.init_x(shape).to(device) if x_init is None else x_init.clone()
        if cfg is not None:
            if unconditional_inputs is None:
                unconditional_inputs = {
                    k: torch.zeros_like(v) for k, v in model_inputs.items()
                }
            model_inputs = {
                k: (
                    torch.cat([v, v_u], dim=0)
                    if isinstance(v, torch.Tensor)
                    else (
                        [
                            (
                                torch.cat([vi, vi_u], dim=0)
                                if isinstance(vi, torch.Tensor)
                                and isinstance(vi_u, torch.Tensor)
                                else None
                            )
                            for vi, vi_u in zip(v, v_u)
                        ]
                        if isinstance(v, list)
                        else (
                            {
                                vk: torch.cat(
                                    [v[vk], v_u.get(vk, torch.zeros_like(v[vk]))],
                                    dim=0,
                                )
                                for vk in v
                            }
                            if isinstance(v, dict)
                            else None
                        )
                    )
                )
                for (k, v), (k_u, v_u) in zip(
                    model_inputs.items(), unconditional_inputs.items()
                )
            }

        for i in range(0, timesteps):
            noise_cond = self.noise_cond(logSNR_range[i])
            if (
                cfg is not None
                and (cfg_t_stop is None or r_range[i].item() >= cfg_t_stop)
                and (cfg_t_start is None or r_range[i].item() <= cfg_t_start)
            ):
                cfg_val = cfg
                if isinstance(cfg_val, (list, tuple)):
                    assert (
                        len(cfg_val) == 2
                    ), "cfg must be a float or a list/tuple of length 2"
                    cfg_val = cfg_val[0] * r_range[i].item() + cfg_val[1] * (
                        1 - r_range[i].item()
                    )

                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        pred, pred_unconditional = model(
                            torch.cat([x, x], dim=0),
                            noise_cond.repeat(2),
                            **model_inputs,
                        ).chunk(2)
                pred_cfg = torch.lerp(pred_unconditional, pred, cfg_val)
                if cfg_rho > 0:
                    std_pos, std_cfg = pred.std(), pred_cfg.std()
                    pred = cfg_rho * (
                        pred_cfg * std_pos / (std_cfg + 1e-9)
                    ) + pred_cfg * (1 - cfg_rho)
                else:
                    pred = pred_cfg
            else:
                pred = model(x, noise_cond, **model_inputs)
            x0, epsilon = self.undiffuse(x, logSNR_range[i], pred)

            base_batch = x.shape[0]

            def expand_cfg_model_inputs(obj, target_batch):
                if torch.is_tensor(obj):
                    if obj.ndim == 0:
                        return obj

                    if obj.shape[0] == 2 * base_batch:
                        if target_batch == base_batch:
                            return obj

                        if target_batch % base_batch != 0:
                            raise ValueError(
                                f"target_batch={target_batch} is not divisible "
                                f"by base_batch={base_batch}"
                            )

                        repeat_factor = target_batch // base_batch
                        obj_pos, obj_uncond = obj.chunk(2, dim=0)
                        obj_pos = (
                            obj_pos.unsqueeze(0)
                            .expand(repeat_factor, *obj_pos.shape)
                            .reshape(target_batch, *obj_pos.shape[1:])
                        )
                        obj_uncond = (
                            obj_uncond.unsqueeze(0)
                            .expand(repeat_factor, *obj_uncond.shape)
                            .reshape(target_batch, *obj_uncond.shape[1:])
                        )

                        return torch.cat([obj_pos, obj_uncond], dim=0)

                    return obj
                elif isinstance(obj, dict):
                    return {
                        k: expand_cfg_model_inputs(v, target_batch)
                        for k, v in obj.items()
                    }
                elif isinstance(obj, tuple):
                    return tuple(expand_cfg_model_inputs(v, target_batch) for v in obj)
                elif isinstance(obj, list):
                    return [expand_cfg_model_inputs(v, target_batch) for v in obj]
                return obj

            def predict_fn(x_in, noise_cond_in):
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        if (
                            cfg is not None
                            and (cfg_t_stop is None or r_range[i].item() >= cfg_t_stop)
                            and (
                                cfg_t_start is None or r_range[i].item() <= cfg_t_start
                            )
                        ):

                            current_batch = x_in.shape[0]

                            model_inputs_batch = expand_cfg_model_inputs(
                                model_inputs,
                                current_batch,
                            )

                            x_cfg = torch.cat([x_in, x_in], dim=0)

                            noise_cond_cfg = torch.cat(
                                [noise_cond_in, noise_cond_in], dim=0
                            )

                            pred_pos, pred_uncond = model(
                                x_cfg,
                                noise_cond_cfg,
                                **model_inputs_batch,
                            ).chunk(2)

                            pred_cfg = torch.lerp(
                                pred_uncond,
                                pred_pos,
                                cfg_val,
                            )

                            if cfg_rho > 0:
                                std_pos = pred_pos.std()
                                std_cfg = pred_cfg.std()

                                pred_out = cfg_rho * (
                                    pred_cfg * std_pos / (std_cfg + 1e-9)
                                ) + pred_cfg * (1 - cfg_rho)

                            else:
                                pred_out = pred_cfg

                            return pred_out

                        else:
                            return model(x_in, noise_cond_in, **model_inputs)

            U_t = estimate_pixelwise_uncertainty(
                predict_fn=predict_fn,
                x_t=x,
                noise_cond=noise_cond,
                logsnr_t=logSNR_range[i],
                undiffuse_fn=self.undiffuse,
                pred_t=pred,
                M=uncertainty_M,
            )

            if i < tau:
                z0 = x0.clone().detach()
                z0.requires_grad_(True)
                optimizer = torch.optim.Adam([z0], lr=eta * (1.0 - i / timesteps))

                with torch.no_grad():
                    U_ctrl = U_t.detach()
                    if U_ctrl.shape[1] != z0.shape[1]:
                        assert U_ctrl.shape[1] == 1
                        U_ctrl = U_ctrl.expand(
                            -1,
                            z0.shape[1],
                            -1,
                            -1,
                        )
                    U_norm = normalize_uncertainty(U_ctrl)

                org_style = models.previewer(x0_style_forward)
                _, _, style_embeddings2 = self.csd_model(
                    extras.clip_preprocess(org_style)
                )
                style_embeddings2 = style_embeddings2.detach()

                for _ in range(num_iter):
                    z_prev = z0.detach().clone()

                    pred_image = models.previewer(z0)

                    _, _, style_embeddings = self.csd_model(
                        extras.clip_preprocess(pred_image)
                    )

                    style_loss = (
                        (1 - self.cosine_loss(style_embeddings, style_embeddings2))
                        .abs()
                        .mean()
                    )

                    loss = lam_style * style_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        delta = z0 - z_prev
                        sigma = 1.0 + U_norm

                        z_new = z_prev + sigma * delta
                        z0.copy_(z_new)

                x0 = z0.detach()

            x = sampler(
                x, x0, epsilon, logSNR_range[i], logSNR_range[i + 1], **sampler_params
            )
            altered_vars = yield (x0, x, pred)

            if altered_vars is not None:
                cfg = altered_vars.get("cfg", cfg)
                cfg_rho = altered_vars.get("cfg_rho", cfg_rho)
                sampler = altered_vars.get("sampler", sampler)
                model_inputs = altered_vars.get("model_inputs", model_inputs)
                x = altered_vars.get("x", x)
                x_init = altered_vars.get("x_init", x_init)