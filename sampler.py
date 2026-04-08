import tqdm
import torch
import numpy as np
import torch.nn as nn
from cores.trajectory import Trajectory, TrajectoryDMD
from cores.scheduler import get_diffusion_scheduler, DiffusionPFODE, EDMScheduler
from cores.mcmc import MCMCSampler
from typing import Optional, Literal, Callable
from PIL import Image
from typing import Optional, Dict
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import LBFGS
import math
import random

IND_TRAJ = False
##########################################################
def _as_bchw(x: torch.Tensor) -> torch.Tensor:
    # Accept [C,H,W] or [1,C,H,W] or [B,C,H,W]
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected 3D/4D tensor, got {tuple(x.shape)}")

def _stack_particles(x_list):
    xs = [_as_bchw(x) for x in x_list]           # each [B,C,H,W]
    # assume same B for all; concat along particle dim
    stack = torch.stack(xs, dim=0)               # [P,B,C,H,W]
    return stack

def _conv_same(x, k):
    # x: [B,C,H,W], k: [1,1,kh,kw] (applied depthwise)
    B, C, H, W = x.shape
    kh, kw = k.shape[-2:]
    pad = (kw//2, kw//2, kh//2, kh//2)
    xpad = F.pad(x, pad, mode="reflect")
    k = k.to(x.device, x.dtype)
    k = k.expand(C, 1, kh, kw)                   # depthwise
    return F.conv2d(xpad, k, groups=C)

def _gaussian_kernel_2d(ks=5, sigma=1.0, device=None, dtype=None):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / (k.sum() + 1e-12)
    return k.view(1, 1, ks, ks)

def _laplacian_pyramid(x, levels=3, ks=5, sigma=1.0):
    """
    x: [B,C,H,W]
    returns:
      lows: list of Gaussian lowpass at each level
      laps: list of laplacian bands (same spatial size per level before downsample)
    """
    lows, laps = [], []
    cur = x
    k = _gaussian_kernel_2d(ks=ks, sigma=sigma, device=x.device, dtype=x.dtype)
    for _ in range(levels):
        low = _conv_same(cur, k)
        lap = cur - low
        lows.append(low)
        laps.append(lap)
        # downsample low for next level
        cur = F.avg_pool2d(low, kernel_size=2, stride=2, ceil_mode=True)
    lows.append(cur)  # coarsest residual low
    return lows, laps

def _laplacian_reconstruct(lows, laps):
    cur = lows[-1]
    for lvl in reversed(range(len(laps))):
        # upsample to lap size
        target = laps[lvl].shape[-2:]
        cur = F.interpolate(cur, size=target, mode="bilinear", align_corners=False)
        cur = cur + laps[lvl]
    return cur

def _energy_map(x, eps=1e-8):
    # x: [B,C,H,W] -> [B,1,H,W] local gradient energy proxy
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    gx = F.pad(gx, (0,1,0,0))
    gy = F.pad(gy, (0,0,0,1))
    e = (gx**2 + gy**2).mean(dim=1, keepdim=True)
    return torch.sqrt(e + eps)

def _hann_window(h, w, device, dtype):
    wh = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    ww = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    win = (wh[:, None] * ww[None, :])
    return win[None, None, :, :]  # [1,1,H,W]

def _tukey_window(h, w, alpha=0.5, device=None, dtype=None):
    # alpha=0 -> rectangular, alpha=1 -> hann
    def tukey1d(n, a):
        x = torch.linspace(0, 1, n, device=device, dtype=dtype)
        out = torch.ones(n, device=device, dtype=dtype)
        if a <= 0:
            return out
        if a >= 1:
            return torch.hann_window(n, periodic=False, device=device, dtype=dtype)
        t1 = a/2
        t2 = 1 - a/2
        left = x < t1
        right = x > t2
        out[left] = 0.5 * (1 + torch.cos(2*torch.pi*(x[left]/a - 0.5)))
        out[right] = 0.5 * (1 + torch.cos(2*torch.pi*((x[right]-1)/a + 0.5)))
        return out
    wh = tukey1d(h, alpha)
    ww = tukey1d(w, alpha)
    win = wh[:, None] * ww[None, :]
    return win[None, None, :, :]

@torch.no_grad()
def aggregate(
    x0_hat_list,
    agg_type="mean",
    step_counter=0,
    warmup=70,
    alpha_max=0.30,          # for partial aggregation
    alpha_ramp=20,           # steps to ramp alpha after warmup
    # trimmed mean
    trim_ratio=0.25,         # 0.25 -> drop 25% low/high per-pixel (needs P>=4)
    # softmin-to-ref
    ref_mode="best_energy",  # ["best_energy", "index0"]
    tau_sim=0.10,            # similarity temperature for softmin_to_ref
    # laplacian pyramid fusion
    pyr_levels=3,
    pyr_ks=5,
    pyr_sigma=1.0,
    # patch selection
    patch_size=64,
    overlap=0.5,             # 0.. <1
    select_metric="energy",  # ["energy", "l2_to_ref"]
    # optional measurement-consistency weights (pass callables if available)
    A_forward=None,          # function: x->[...], e.g., A(x) with same device
    y_meas=None,             # measurement tensor
    tau_dc=1e-2,             # temperature for data-consistency weights
):
    """
    Returns list length P of tensors shaped like inputs (each [B,C,H,W]).
    Supports inputs as list of [C,H,W] or [B,C,H,W] (B can be 1).

    agg_type options:
      - "none"
      - "mean"
      - "patch_select"          (pick best particle per patch; overlap-add with Hann)
    """
    stack = _stack_particles(x0_hat_list)     # [P,B,C,H,W]
    P, B, C, H, W = stack.shape

    # helper: return in same per-particle format (list of [B,C,H,W])
    def _broadcast(x_agg_bchw):
        return [x_agg_bchw.clone() for _ in range(P)]

    def _alpha():
        if step_counter < warmup:
            return 0.0
        t = min(1.0, (step_counter - warmup) / max(1, alpha_ramp))
        return float(alpha_max * t)

    if agg_type == "none":
        return [stack[p].clone() for p in range(P)]

    if agg_type == "mean":
        x_agg = stack.mean(dim=0)  # [B,C,H,W]
        return _broadcast(x_agg)


    if agg_type == "patch_select":
        if warmup:
            if isinstance(warmup, list):
                # aggregation happens only between warmup[0] and warmup[1]
                if step_counter < warmup[0]:
                    return [stack[p].clone() for p in range(P)]
                elif step_counter > warmup[1]:
                    return [stack[p].clone() for p in range(P)]
            else:
                if step_counter < warmup:
                    return [stack[p].clone() for p in range(P)]

        # Select the "best" particle per patch, then overlap-add with Hann weights.
        stride = max(1, int(round(patch_size * (1 - overlap))))
        
        win = _tukey_window(patch_size, patch_size, alpha=0.3, device=stack.device, dtype=stack.dtype)  # [1,1,ps,ps]

        out = torch.zeros((B, C, H, W), device=stack.device, dtype=stack.dtype)
        wsum = torch.zeros((B, 1, H, W), device=stack.device, dtype=stack.dtype)

        # choose ref for l2_to_ref metric if needed
        if select_metric == "l2_to_ref":
            ref = stack[0]

        for top in range(0, H, stride):
            for left in range(0, W, stride):
                bottom = min(H, top + patch_size)
                right  = min(W, left + patch_size)
                # crop patch size (may be smaller at borders); adapt window
                ph, pw = bottom - top, right - left
                win_cur = win[..., :ph, :pw]  # [1,1,ph,pw]

                # extract patches: [P,B,C,ph,pw]
                patches = stack[:, :, :, top:bottom, left:right]

                if select_metric == "energy":
                    # pick particle with max gradient energy on this patch
                    scores = torch.stack([_energy_map(patches[p]).mean() for p in range(P)], dim=0)  # [P]
                    pbest = int(torch.argmax(scores).item())
                elif select_metric == "l2_to_ref":
                    ref_patch = ref[:, :, top:bottom, left:right]
                    scores = torch.stack([((patches[p] - ref_patch) ** 2).mean() for p in range(P)], dim=0)
                    pbest = int(torch.argmin(scores).item())
                else:
                    raise ValueError(f"Unknown select_metric: {select_metric}")

                patch_best = patches[pbest]  # [B,C,ph,pw]
                out[:, :, top:bottom, left:right] += patch_best * win_cur
                wsum[:, :, top:bottom, left:right] += win_cur

        x_agg = out / (wsum + 1e-12)
        return _broadcast(x_agg)

    raise NotImplementedError(f"Unknown agg_type: {agg_type}")
########################################################################################
def tensor_to_pils(x):
    """
        [B, C, H, W] tensor -> list of pil images
    """
    def norm(x):
        """
            normalize data to [0, 1] range
        """
        return (x * 0.5 + 0.5).clip(0, 1)
    
    # Handle input types
    if isinstance(x, torch.Tensor):
        if x.dim() == 4:  # batch
            tensors = [xi for xi in x]
        elif x.dim() == 3:  # single image
            tensors = [x]
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")
    elif isinstance(x, list):
        tensors = x
        if x[0].dim() == 4:
            tensors = [im[0] for im in tensors]
    else:
        raise TypeError("Input must be a torch.Tensor or list of tensors")

    # Convert each tensor to PIL
    pils = []
    for t in tensors:
        t = norm(t).detach().cpu()
        np_x = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pils.append(Image.fromarray(np_x))
    
    return pils

def plot_all_clean(imgs, img_name="dummy.png"):
    plt.figure(figsize=(16, 4))
    pils = tensor_to_pils(imgs)
    for i, img in enumerate(pils):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(img_name)
    plt.clf()
    plt.close()

def plot_all_heatmaps(hmaps, img_name="heatmaps.png", cmap="jet", vmin=None, vmax=None):
    """
    Plot up to 4 heatmaps in a row with a shared color scale.
    
    Args:
        hmaps: torch tensor [B,H,W] or [B,1,H,W]
        img_name: output filename
        cmap: colormap
        vmin, vmax: numeric range; if None -> inferred from data
    """
    # ensure tensor
    if not isinstance(hmaps, torch.Tensor):
        raise TypeError("hmaps must be a torch.Tensor")

    hmaps = hmaps.detach().cpu()

    # collapse channel if [B,1,H,W]
    if hmaps.dim() == 4 and hmaps.shape[1] == 1:
        hmaps = hmaps[:, 0]   # [B,H,W]
    elif hmaps.dim() != 3:
        raise ValueError("Heatmaps must be shape [B,H,W] or [B,1,H,W]")

    B = min(4, hmaps.shape[0])

    # global min/max if not provided
    if vmin is None: vmin = hmaps.min().item()
    if vmax is None: vmax = hmaps.max().item()

    fig, axes = plt.subplots(1, B, figsize=(16, 4))
    # if only one map -> axes is not iterable
    if B == 1:
        axes = [axes]

    img0 = None
    for i in range(B):
        ax = axes[i]
        img0 = ax.imshow(hmaps[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")

    # Shared colorbar
    cbar = fig.colorbar(img0, ax=axes, shrink=0.8)
    cbar.set_label("Intensity")

    plt.tight_layout()
    plt.savefig(img_name)
    plt.clf()
    plt.close()
    
def w2_barycenter_geomloss(images, weights=None, blur=0.01, n_iter=25):
    """
    Compute W2 barycenter via differentiable Sinkhorn (GeomLoss).
    Args:
        images : torch.Tensor [N,C,H,W] normalized to [0,1]
        weights: optional barycentric weights
        blur   : smoothing scale (~regularization strength)
    Returns:
        barycenter: torch.Tensor [C,H,W]
    """
    N, C, H, W = images.shape
    device = images.device
    weights = torch.ones(N, device=device) / N if weights is None else weights.to(device)

    # Flatten spatial dimensions
    xs = [im.reshape(C, -1).T for im in images]  # each [H*W, C]
    loss_fn = SamplesLoss("sinkhorn", p=2, blur=blur)

    # Initialize barycenter as pixel mean
    bary = images.mean(0, keepdim=False).clone().requires_grad_(True)
    opt = torch.optim.Adam([bary], lr=5e-2)

    for k in range(n_iter):
        opt.zero_grad()
        bary_flat = bary.reshape(C, -1).T
        loss = sum(loss_fn(bary_flat, x) * w for x, w in zip(xs, weights))
        loss.backward()
        opt.step()
        # bary.data.clamp_(0, 1)
    return bary.detach()

def plot_diff_heatmaps(
    before, after,
    img_name="diff_heatmaps.png",
    cmap="bwr",
    mode="signed",   # "signed" or "abs"
    vmin=None,
    vmax=None
):
    """
    Visualize difference between BEFORE and AFTER RGB images [B,3,H,W] as heatmaps.
    
    Diff is collapsed over channel as mean.
    
    Args:
        before, after: [B,3,H,W] torch tensors
        mode: "signed" → after-before, "abs" → |after-before|
        cmap: colormap
        vmin, vmax: explicit heatmap scale (optional)
    """

    if not (isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor)):
        raise TypeError("before and after must be torch.Tensors")

    before = before.detach().cpu()
    after  = after.detach().cpu()

    if before.shape != after.shape:
        raise ValueError("before and after must have same shape")

    if before.dim() != 4 or before.shape[1] != 3:
        raise ValueError("Input must be [B,3,H,W]")

    # Compute per-pixel difference
    diff = after - before         # [B,3,H,W]

    # Reduce across channels → grayscale heatmap
    diff = diff.mean(dim=1)       # [B,H,W]

    if mode == "abs":
        diff = diff.abs()

    B = min(4, diff.shape[0])

    # Auto scaling
    if vmin is None: vmin = diff.min().item()
    if vmax is None: vmax = diff.max().item()

    fig, axes = plt.subplots(1, B, figsize=(16, 4))
    if B == 1:
        axes = [axes]

    img0 = None
    for i in range(B):
        ax = axes[i]
        img0 = ax.imshow(diff[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Diff #{i}")
        ax.axis("off")

    # Shared colorbar
    cbar = fig.colorbar(img0, ax=axes, shrink=0.8)
    cbar.set_label("Difference Intensity")

    plt.tight_layout()
    plt.savefig(img_name)
    plt.clf()
    plt.close()

def get_sampler(**kwargs):
    sampler_name = kwargs.get('name')
    kwargs.pop('name')
    if sampler_name == 'edm_daps':
        return DAPS(**kwargs)
    elif sampler_name == 'edm_ddnm':
        kwargs.pop('mcmc_sampler_config', None)  
        return DDNM(**kwargs)
    elif sampler_name == 'edm_dmd':
        kwargs.pop('mcmc_sampler_config', None) 
        return DMD(**kwargs)
    else:   
        raise NotImplementedError(f"Sampler {sampler_name} with latent={latent} not implemented.")


class DMD(nn.Module):
    def __init__(self, diffusion_scheduler_config, annealing_scheduler_config,
                 sigma_y=0.05, sigma_cut=0.01, num_particles=4):
        super().__init__()
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.sigma_y = sigma_y
        self.sigma_cut = sigma_cut
        self.num_particles = num_particles
        
        print("="*50)
        print("DMD Sampler initialized")
        print(f"Num particles: {self.num_particles}")
        print(f"Sigma_y: {self.sigma_y}")
        print(f"Sigma_cut: {self.sigma_cut}")
        print("="*50)

   
    @torch.no_grad()
    def generate_masks_and_measurements(
        self,
        x_gt, operator, 
        mask_type="quadrant",
        pixel_fraction=0.3,
        patch_size=64,
        overlap=0.25,
        smooth_sigma=5.0,
        device="cuda",
    ):
        """
        Returns A_list, Ap_list, y_list, masks
        where Ai(x)=mask_i*x and yi = operator(Ai(x_true)) in measurement space.
        For generic operator: y_i = operator(masked_img).
        """
        B, C, H, W = x_gt.shape
        masks = []

        if mask_type == "quadrant":
            n_rows = int(math.floor(math.sqrt(self.num_particles)))
            n_cols = math.ceil(self.num_particles / n_rows)
            h_block = H // n_rows
            w_block = W // n_cols

            for r in range(n_rows):
                for c in range(n_cols):
                    y0, x0 = r * h_block, c * w_block
                    y1 = H if r == n_rows - 1 else (r + 1) * h_block
                    x1 = W if c == n_cols - 1 else (c + 1) * w_block
                    mask = torch.zeros_like(x_gt, device=device)
                    mask[..., y0:y1, x0:x1] = 1.0
                    masks.append(mask)

        elif mask_type == "random_pixel":
            for _ in range(self.num_particles):
                mask = (torch.rand((1, 1, H, W), device=device) < pixel_fraction).float()
                masks.append(mask.expand_as(x_gt))

        elif mask_type == "overlap":
            ps = patch_size
            for _ in range(self.num_particles):
                y0 = random.randint(0, max(0, H - ps))
                x0 = random.randint(0, max(0, W - ps))
                mask = torch.zeros((1, 1, H, W), device=device)
                mask[..., y0:y0 + ps, x0:x0 + ps] = 1.0

                if smooth_sigma > 0:
                    kernel_size = int(2 * round(3 * smooth_sigma) + 1)
                    kernel_1d = torch.arange(kernel_size, device=device) - kernel_size // 2
                    gauss_1d = torch.exp(-0.5 * (kernel_1d / smooth_sigma) ** 2)
                    gauss_1d = gauss_1d / gauss_1d.sum()
                    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
                    gauss_2d = gauss_2d[None, None, :, :]
                    mask = F.conv2d(mask, gauss_2d, padding=kernel_size // 2)
                    mask = mask / mask.max().clamp_min(1e-8)

                masks.append(mask.expand_as(x_gt))
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

        def make_op(mask):
            return lambda x: x * mask

        A_list = [make_op(m) for m in masks]
        Ap_list = [make_op(m) for m in masks]

        # measurement per block
        y_list = []
        for m in masks:
            masked_img = m * x_gt
            y_i = operator.measure(masked_img)  # measurement noise is added inside operator
            y_list.append(y_i)

        return A_list, Ap_list, y_list, masks

    # --------------------------------------------------------------------------------
    # EDM-conditioning update: randomized blockwise residual backproj + EDM Lambda
    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def randomized_blockwise_condition(
        self,
        x0_list,                    # list of [B,C,H,W]
        operator,                   # measurement operator with __call__ and upsample
        measurement,                # y (global measurement) OR y_list if already decomposed
        sigma,                      # current EDM sigma
        alpha=1.0,
        mask_type="quadrant", diff_step=0, x_orig=None
    ):

        # convert num_particles,B, C, H, W tersors to list of B,C,H,W tensors
        if isinstance(x0_list, torch.Tensor):
            x0_list = [torch.unsqueeze(x0_list[p],0) for p in range(x0_list.shape[0])]
        
        x_ref = x0_list[0]  # any tensor with correct shape for mask creation
        A_list, Ap_list, y_list, _ = self.generate_masks_and_measurements(
            x_orig, operator,mask_type=mask_type, device=x_ref.device
        )
        
        # randomized pairing of (img_idx, block_idx)
        num_particles = len(x0_list)
        img_indices = list(range(num_particles))
        blk_indices = list(range(num_particles))

        grads = []
        
        for _ in range(num_particles):
            random.shuffle(img_indices)
            grad_acc = 0.0
            for img_idx, block_idx in zip(img_indices, blk_indices):
                Ai, Api, yi = A_list[block_idx], Ap_list[block_idx], y_list[block_idx]
                xi = x0_list[img_idx]

                # forward residual in measurement space
                pred = operator.measure(Ai(xi))
                resid = pred.reshape(pred.shape[0], -1) - yi.reshape(yi.shape[0], -1)
                
                backproj = operator.A_pinv(resid).reshape_as(xi)
                if hasattr(operator, "Lambda_edm"):
                    backproj_scaled = operator.Lambda_edm(backproj, sigma, self.sigma_y).reshape_as(xi)
                else:
                    backproj_scaled = backproj

                grad_i = Api(backproj_scaled)
                grad_acc = grad_acc + grad_i

            grads.append(grad_acc)

        x0_hat_list = [x0 - alpha * g for x0, g in zip(x0_list, grads)]
        return x0_hat_list

    @torch.no_grad()
    def deterministic_blockwise_condition(
        self,
        x0_list,                    
        operator,                  
        measurement,               
        sigma,                      
        alpha=1.0,
        mask_type="quadrant", 
        diff_step=0, 
        x_orig=None
    ):

        # convert num_particles,B, C, H, W tersors to list of B,C,H,W tensors
        if isinstance(x0_list, torch.Tensor):
            x0_list = [torch.unsqueeze(x0_list[p],0) for p in range(x0_list.shape[0])]
        
        x_ref = x0_list[0]  # any tensor with correct shape for mask creation
        A_list, Ap_list, y_list, _ = self.generate_masks_and_measurements(
            x_orig, operator,mask_type=mask_type, device=x_ref.device
        )

        grad_acc = 0.0
        
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(4,5, figsize=(8,16))
        
        # import matplotlib.pyplot as plt
        # pinv_img = operator.A_pinv(measurement).cpu().reshape_as(x_ref)[0].permute(1,2,0)
        # pinv_img = (pinv_img - pinv_img.min()) / (pinv_img.max() - pinv_img.min())
        # plt.imshow(pinv_img)
        # plt.savefig(f"pinv_img_step_{str(diff_step)}.png")
        # plt.clf()
        
        for i in range(self.num_particles):
          
            Ai, Api, yi = A_list[i], Ap_list[i], y_list[i]
            xi = x0_list[i]

            # plot xi and yi for debugging
            # x_img = torch.clamp((xi.cpu()+1.0) / 2.0, 0,1)[0].permute(1,2,0)
            # y_img = operator.A_pinv(yi.reshape(yi.size(0), -1)).reshape_as(xi)
            # y_img = torch.clamp((y_img.cpu()+1.0) / 2.0, 0,1)[0].permute(1,2,0)
            # ax[i,0].imshow(x_img) 
            # ax[i,0].set_title(f"xi Block {i}")
            # ax[i,1].imshow(y_img) 
            # ax[i,1].set_title(f"yi Block {i}")
            
            # forward residual in measurement space
            pred = operator.forward(Ai(xi))
            resid = pred.reshape(pred.shape[0], -1) - yi.reshape(yi.shape[0], -1)
            
            # import matplotlib.pyplot as plt
            # resid = operator.A_pinv(resid.reshape(resid.shape[0], -1)).cpu().reshape_as(xi)
            # resid = torch.clamp((resid.cpu()+1.0) / 2.0, 0,1)[0].permute(1,2,0)
            # plt.imshow(resid)
            # plt.savefig(f"resid_step_{str(diff_step)}.png")
            # plt.clf()
            
            
            backproj = operator.A_pinv(resid).reshape_as(xi)
            backproj_scaled = operator.Lambda_edm(backproj, sigma, self.sigma_y).reshape_as(xi)
            grad_i = Api(backproj_scaled)
            # grad_i = backproj_scaled
            grad_acc += grad_i
            
            # grad_img = torch.clamp((grad_i.cpu()+1.0) / 2.0, 0,1)[0].permute(1,2,0)
            # ax[i,2].imshow(grad_img) 
            # ax[i,2].set_title(f"Grad Block {i}")    
            
            # grad_acc_img = torch.clamp((grad_acc.cpu()+1.0) / 2.0, 0,1)[0].permute(1,2,0)
            # ax[i,3].imshow(grad_acc_img) 
            # ax[i,3].set_title(f"Acc Grad Block {i}")
        
        # else:  
        x0_hat_list = [x0 - alpha * grad_acc for x0 in x0_list]
        # for i in range(self.num_particles):
        #     x0_hat_img = torch.clamp((x0_hat_list[i].cpu()+1.0) / 2.0, 0,1)[0].permute(1,2,0)
        #     ax[i,4].imshow(x0_hat_img) 
        #     ax[i,4].set_title(f"x0_hat Block {i}")
            
        # plt.savefig(f"blockwise_debug_step_{str(diff_step)}.png")
        # plt.clf()
        return x0_hat_list
    
    @torch.no_grad()
    def DDNM_condition(
        self,
        x0_list,                    # list of [B,C,H,W]
        operator,                   # measurement operator with __call__ and upsample
        measurement,                # y (global measurement) OR y_list if already decomposed
        sigma,                      # current EDM sigma
        diff_step=0,
    ):

        # convert num_particles,B, C, H, W tersors to list of B,C,H,W tensors
        if isinstance(x0_list, torch.Tensor):
            x0_list = [torch.unsqueeze(x0_list[p],0) for p in range(x0_list.shape[0])]
        
        # randomized pairing of (img_idx, block_idx)
        num_particles = len(x0_list)
        x0_hat_list = []
        for i in range(num_particles):
            # forward residual in measurement space
            xi = x0_list[i]
            pred = operator.measure(xi)
            resid = pred.reshape(pred.shape[0], -1) - measurement.reshape(measurement.shape[0], -1)
            
            backproj = operator.A_pinv(resid).reshape_as(xi)
            backproj_scaled = operator.Lambda_edm(backproj, sigma, self.sigma_y).reshape_as(xi)
            x0_hat = xi - backproj_scaled
            x0_hat_list.append(x0_hat)
        
        return x0_hat_list
    # --------------------------------------------------------------------------------
    # Aggregation (mean/median/none) same as your code
    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def aggregate(self, x0_hat_list, agg_type="mean", step_counter=0, warmup=10):
        stack = torch.cat(x0_hat_list, dim=0)  # [M,C,H,W]
        M = len(x0_hat_list)

        if agg_type == "mean":
            x_agg = torch.mean(stack, dim=0, keepdim=True)  # [1,C,H,W]
            return [x_agg.clone() for _ in range(M)]

        # if agg_type == "median":
        #     if step_counter < 90:
        #         # explore
        #         return [x0_hat_list[p].clone() for p in range(M)]
        #     else:
        #         x_agg = torch.median(stack, dim=0)[0].unsqueeze(0)
        #         return [x_agg.clone() for _ in range(M)]
        if agg_type == "median":
            x_agg = torch.median(stack, dim=0)[0].unsqueeze(0)
            alpha = 0.0 if step_counter < warmup or step_counter > 90 else min(1.0, (step_counter-warmup)/20) * 0.3
            return [(1-alpha)*x0_hat_list[p] + alpha*x_agg for p in range(M)]
           

        if agg_type == "none":
            return [x0_hat_list[p].clone() for p in range(M)]

        raise NotImplementedError(f"Unknown agg_type: {agg_type}")

    # --------------------------------------------------------------------------------
    # Main sampler: outer sigma loop, inner denoise, condition, EDM-noise forward
    # --------------------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        model,
        x_start,                 # initial latent/image at high sigma
        operator,                # measurement operator
        measurement,             # observed y
        inner_pfode_steps=5,
        alpha=1.0,
        agg_type="median",
        mask_type="quadrant",
        verbose=False,
        evaluator=None, record=False, gt=None
        
    ):
        if record:
            self.trajectory = TrajectoryDMD()
       
        xt_list = x_start[:, 0, :, :, :] # num_particles, C, H, W
        step_counter = 0

        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            sigma_next = self.annealing_scheduler.sigma_steps[step + 1]
            # ---- 1) denoise each particle using PF-ODE with sigma_max = sigma ----
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver="euler")

                # for p in range(num_particles):
                x0_hat = sampler.sample(xt_list, num_steps=inner_pfode_steps)
            
            # ---- 2) conditioning: blockwise update in EDM form ----
            x0y_hat_list = self.deterministic_blockwise_condition(
                x0_list=x0_hat,
                operator=operator,
                measurement=measurement,
                sigma=sigma,
                alpha=alpha,
                mask_type="quadrant", 
                diff_step=step,
                x_orig=gt
            )
            # x0y_hat_list = self.randomized_blockwise_condition(
            #     x0_list=x0_hat,
            #     operator=operator,
            #     measurement=measurement,
            #     sigma=sigma,
            #     alpha=alpha,
            #     mask_type="quadrant", 
            #     diff_step=step,
            #     x_orig=gt
            # )
            

            # ---- 3) aggregation ----
            # if hasattr(operator, "corr_type"):
            #     if operator.corr_type == "box":
            #         x0y_hat_list_agg = aggregate(
            #             x0y_hat_list,
            #             agg_type="mean",
            #             step_counter=step_counter,
            #             patch_size=16,
            #             overlap=0.5,             
            #             select_metric="energy", 
            #             warmup=None,
            #             alpha_max=0.25,
            #             alpha_ramp=25,
            #             pyr_levels=3,
            #         )
            #         # H = W = 256
            #         # center_size = 128
            #         # start = (H - center_size) // 2   # 64
            #         # end   = start + center_size 
            #         # mask = np.zeros((H, W), dtype=np.float32)
            #         # mask[start:end, start:end] = 1.0
            #         # missing = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda()
                    
            #         # known = 1 - missing
            #         # # import matplotlib.pyplot as plt
            #         # # plt.imshow(known[0, 0 , :, :].cpu())
            #         # # plt.savefig("know_pix.png")
            #         # # plt.clf()
            #         # # exit()
            #         # agg_missing = [(x0 * missing) for x0 in x0y_hat_list]
            #         # x0y_hat_list_agg = [
            #         #     known * x0y_hat_list[p] +               # keep known pixels fixed
            #         #     missing * torch.stack(agg_missing, dim=0).mean(dim=0)          # aggregate only missing regions
            #         #     for p in range(self.num_particles)
            #         # ]
            # else:
            # No warmup
            x0y_hat_list_agg = aggregate(
                x0y_hat_list,
                agg_type="patch_select",
                step_counter=step_counter,
                patch_size=16,
                overlap=0.5,             
                select_metric="energy", 
                warmup=None,
                alpha_max=0.25,
                alpha_ramp=25,
                pyr_levels=3,
            )
            
            # # Do something specific for box-inpainting
            # stack = torch.cat(x0y_hat_list_agg)
            # H = W = 256
            # center_size = 128
            # start = (H - center_size) // 2   # 64
            # end   = start + center_size 
            # mask = np.ones((H, W), dtype=np.float32)
            # mask[start:end, start:end] = 0.0
            # missing = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda()
            # known = 1 - missing
            # agg_missing = (stack * missing).mean(dim=0)
            
            # x0y_hat_list_agg = (
            #     known * stack +               # keep known pixels fixed
            #     missing * agg_missing          # aggregate only missing regions
            # )
            # x0y_hat_list_agg = [torch.unsqueeze(x0y_hat_list_agg[p],0) for p in range(self.num_particles)]
           
            # ---- 4) forward diffusion to sigma_next (EDM-shaped noise) ----
            xt_list_next = []
            if step == self.annealing_scheduler.num_steps - 2:
                # last step: no noise
                xt_list_next = [x.clone() for x in x0y_hat_list_agg]
            else:
                for p in range(self.num_particles):
                    z = torch.randn_like(x0y_hat_list_agg[p])
                    if hasattr(operator, "Lambda_noise_edm"):
                        delta = operator.Lambda_noise_edm(
                            z, sigma_next, self.sigma_y, sigma_cut=self.sigma_cut
                        ).reshape_as(x0y_hat_list_agg[p])
                        
                    else:
                        raise NotImplementedError("Operator must implement Lambda_noise_edm() for EDM-shaped noise.")
                    noisy_img = x0y_hat_list_agg[p] + delta
                    
                    xt_list_next.append(noisy_img)
                    # import matplotlib.pyplot as plt
                    # plt.imshow(noisy_img[0].cpu().permute(1,2,0))
                    # plt.savefig("noisy_img.png")
                    
            xt_list = torch.stack(xt_list_next, dim=0)[:, 0, :, :, :]
            
            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator:
                x0hat = torch.unsqueeze(x0_hat[0], dim=0)
                x0y = x0y_hat_list_agg[0]
                with torch.no_grad():
                    gt = gt
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt_list_next, x0y_hat_list_agg, x0_hat, sigma, x0hat_results, x0y_results)
                
            step_counter += 1


        errs = []
        xt_list = [xt_list[p] for p in range(self.num_particles)]
        
        xt = torch.stack(xt_list, dim=0)
        return xt

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        
        # Log particle-wise tensors (Trajectory will normalize list vs stacked)
        for i in range(self.num_particles):
            self.trajectory.add_tensor(f'xt_{i+1}', xt[i].unsqueeze(0) if xt[i].dim() == 3 else xt[i])
            self.trajectory.add_tensor(f'x0y_{i+1}', x0y[i].unsqueeze(0) if x0y[i].dim() == 3 else x0y[i])
            self.trajectory.add_tensor(f'x0hat_{i+1}', x0hat[i].unsqueeze(0) if x0hat[i].dim() == 3 else x0hat[i])

        # Scalars / per-particle values (Trajectory.add_value handles scalar or [P])
        self.trajectory.add_value("sigma", sigma)

        # Metrics/results: either scalar or per-particle; store as-is
        for name, val in x0hat_results.items():
            self.trajectory.add_value(f"x0hat_{name}", val)

        for name, val in x0y_results.items():
            self.trajectory.add_value(f"x0y_{name}", val)
            
    def get_start(self, batch_size, model, sigma_max=70.0):
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        x_s = []
        SEEDS = [42, 66, 78, 90]
        for p in range(self.num_particles):
            # pick random seed for each particle
            # torch.manual_seed(torch.randint(0,10000,(1,)).item())
            torch.manual_seed(SEEDS[p])
            x_s.append(torch.randn(1, batch_size, *in_shape, device=device) * sigma_max)
        return torch.cat(x_s, dim=0)
    

class DDNM(nn.Module):
   
    def __init__(self, diffusion_scheduler_config, annealing_scheduler_config):
        print("Initializing DDNM...")
        super().__init__()
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        
        self.sigma_y = 0.05
    
    # ---------- Minimal CG on a linear operator ----------
    def cg_solve(self, apply_A, b, x0=None, tol=1e-6, maxiter=200):
        """
        Solve A x = b for symmetric positive-definite operator A, with CG.
        apply_A: function(v) -> A v, operates on same shape as b
        b: tensor
        """
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        r = b - apply_A(x)
        p = r.clone()
        rr_old = torch.dot(r.flatten(), r.flatten())
        if rr_old == 0:
            return x
        for _ in range(maxiter):
            Ap = apply_A(p)
            alpha = rr_old / torch.dot(p.flatten(), Ap.flatten()).clamp_min(1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = torch.dot(r.flatten(), r.flatten())
            if rr_new.sqrt() <= tol:
                break
            beta = rr_new / rr_old
            p = r + beta * p
            rr_old = rr_new
        return x

        # ---------- Helpers to flatten/unflatten ----------
        

    # ---------- INPMAT on a general linear image operator ----------
    @torch.no_grad()
    def inpmat_with_operator(self, y_img, operator, eps=1e-8, max_iter=200, rcond=None,
                            return_history=False, cg_tol=1e-6, cg_maxiter=200):
        """
        INPMAT for image degradation y = Φ x using only Φ(.) and Φᵀ(.).

        Args:
        y_img:  (1, C, H_y, W_y) low-res measurement image (output of operator(x_true))
        operator:
            - __call__(x_img) -> Φ x (downsample)   , expects (1, C, H, W)
            - upsample(z_img) -> Φᵀ z (adjoint-ish), expects (1, C, H_y, W_y)
        eps: stopping criterion on ||y - Φ x||_2^2
        max_iter: max INPMAT iterations
        cg_tol, cg_maxiter: accuracy for inner CG solves

        Returns:
        x_hat: (1, C, H, W) high-res estimate
        (optional) hist: list of residual norms
        """

        device = y_img.device
        dtype  = y_img.dtype

        # We need the unknown high-res spatial size. Infer it from the operator:
        # Make a dummy high-res tensor by *upsampling* y once (rough proxy of HR shape).
        x_shape_guess = operator.upsample(y_img).shape  # (1, C, H, W)
        C, H, W = x_shape_guess[1:]

        # Define Phi(.) and Phi^T(.) with correct shapes
        def Phi(x_img):
            return operator(x_img)

        def PhiT(z_img):
            # For INPMAT we need the *adjoint*. If operator.upsample is not the true adjoint,
            # see the note below to swap in an adjoint-preserving resizer.
            return operator.upsample(z_img)
        
        def _flatten_img(x):
            # Expect x in shape (B?, C, H, W) or (C, H, W) or 1D already
            if x.dim() == 1:
                return x, x.shape
            if x.dim() == 3:
                C, H, W = x.shape
                return x.reshape(-1), (C, H, W)
            if x.dim() == 4:
                B, C, H, W = x.shape
                assert B == 1, "This INPMAT variant expects a single image (B=1)."
                return x.reshape(-1), (C, H, W)
            raise ValueError("Unsupported input shape for image.")

        def _unflatten_img(v, shape, device=None, dtype=None):
            C, H, W = shape
            t = v.reshape(1, C, H, W)
            if device: t = t.to(device)
            if dtype:  t = t.to(dtype)
            return t
        
        # Vectorization helpers
        def vec_img(x_img):    # (1,C,H,W) -> (n,)
            v, _ = _flatten_img(x_img)
            return v
        def img_from_vec(v):   # (n,) -> (1,C,H,W)
            return _unflatten_img(v, (C, H, W), device=device, dtype=dtype)

        # Measurement- and signal-domain sizes
        m = y_img.numel()
        n = C * H * W

        # ---- Define Gram operators needed for pseudoinverses without matrices ----
        # For any w in measurement space: (Φ Φᵀ) w  = Φ(Φᵀ w)
        def apply_G_meas(w_img):
            return Phi(PhiT(w_img))

        # For any v in signal space: Φ^† b = Φᵀ (Φ Φᵀ)^{-1} b
        def pinv_times_measurement(b_img):
            # Solve (Φ Φᵀ) u = b  in measurement domain
            u = self.cg_solve(apply_G_meas, b_img, tol=cg_tol, maxiter=cg_maxiter)
            return PhiT(u)

        # ---- Initialization: x0 = Φ^† y ----
        x0_img = pinv_times_measurement(y_img)
        x_img  = x0_img.clone()

        # Initial threshold thr = max |x0|
        thr = x0_img.abs().amax()

        hist = []
        for it in range(max_iter):
            # residual r = y - Φ x
            r_img = y_img - Phi(x_img)
            res_norm2 = torch.dot(vec_img(r_img), vec_img(r_img)).item()
            hist.append(res_norm2)
            if res_norm2 < eps:
                break

            # Binary mask T^i on the HR grid: keep entries with |x| >= thr
            mask = (x_img.abs() >= thr).to(dtype)  # (1,C,H,W), 0/1

            # Define A = Φ T. We need s = A^† y with the *minimum-norm* solution.
            # Use identity: A^† y = Aᵀ (A Aᵀ)^{-1} y, where A Aᵀ = Φ T T Φᵀ = Φ diag(mask) Φᵀ
            def apply_AAT(w_img):
                # w_img -> (A Aᵀ) w_img = Φ( T (Φᵀ w_img) )
                tmp = PhiT(w_img)
                tmp = tmp * mask
                return Phi(tmp)

            z_img = cg_solve(apply_AAT, y_img, tol=cg_tol, maxiter=cg_maxiter)  # solve (A Aᵀ) z = y
            s_img = PhiT(z_img) * mask                                          # s = Aᵀ z = T Φᵀ z

            # x^{i+1} = x0 + (I - Φ^† Φ) s
            # Compute Φ s, then Φ^†(Φ s)
            Phi_s = Phi(s_img)
            Phi_dag_Phi_s = pinv_times_measurement(Phi_s)
            x_img = x0_img + (s_img - Phi_dag_Phi_s)

            # Update threshold: thr <- max(|(I - T^i) x^i|)
            unselected = (mask == 0)
            if unselected.any():
                thr = x_img[unselected].abs().amax()
            else:
                thr = torch.tensor(0.0, device=device, dtype=dtype)

        return (x_img, hist) if return_history else x_img
   

    def condition_on_y_inpmat(self, x, step, operator, measurement):
        # make the conditioning step is immune to the measurement noise
        x0_t_y = self.inpmat_with_operator(measurement, operator, eps=1e-8, max_iter=200, rcond=None,
                            return_history=False, cg_tol=1e-6, cg_maxiter=200)
        # x0_t_y = x - lambda_t * operator.upsample(operator(x) - measurement)
        return x0_t_y
    
    def condition_on_y_vanilla(self, x, step, operator, measurement, sigma):
        lambda_t = 1.0
        # # make the conditioning step is immune to the measurement noise
        residual = operator.measure(x.reshape(x.size(0), -1)) - measurement.reshape(measurement.size(0), -1)
        back_proj = operator.A_pinv(residual).reshape_as(x)
        backproj = operator.Lambda_edm(
                back_proj,
                sigma,
                self.sigma_y).reshape_as(x)
        # x0_t_y = x - operator.upsample(operator(x) - measurement)
        x0_t_y = x - backproj
        return x0_t_y
    
    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        if record:
            self.trajectory = Trajectory()
            
        NUM_DIFF_STEPS = self.diffusion_scheduler_config.get('num_denoising_steps', 50)
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        # pbar = tqdm.trange(0,NUM_DIFF_STEPS-1) if verbose else range(NUM_DIFF_STEPS-1)
        xt = x_start
        
        # EDM scheduler for sampling
        # diffusion_scheduler = EDMScheduler(num_steps=NUM_DIFF_STEPS, sigma_max=70.0, sigma_min=1e-2, timestep='poly-7')
        # time_steps = diffusion_scheduler.get_discrete_time_steps(NUM_DIFF_STEPS)
        
        for step in pbar:
            # 1. reverse diffusion
            sigma = self.annealing_scheduler.sigma_steps[step]
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
                x0hat = sampler.sample(xt, num_steps=5)

            # condition on y using range null space decomposition
            # x0y = self.condition_on_y_inpmat(x0hat, step, operator, measurement)
            x0y = self.condition_on_y_vanilla(x0hat, step, operator, measurement, sigma=sigma)

            # 3. forward diffusion
            # if step != self.annealing_scheduler.num_steps-1:
            #     xt = x0y + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[step + 1]
            # else:
            #     xt = x0y

            sigma_next = self.annealing_scheduler.sigma_steps[step + 1]  # scalar or tensor
            if step != self.annealing_scheduler.num_steps-1:
                z = torch.randn_like(x0y)  # image-space noise
                delta = operator.Lambda_noise_edm(z, sigma_next, self.sigma_y, sigma_cut=0.03).reshape_as(x0y)
                xt = x0y + delta
            else:
                xt = x0y
          
            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, x0hat_results, x0y_results)
        return xt

    def _record(self, xt, x0y, x0hat, x0hat_results, x0y_results):
        """Records the intermediate states during sampling."""

        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """Checks and updates the configurations for the schedulers."""

        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, batch_size, model):
        """
        Generates initial random state tensors from the Gaussian prior.

        Args:
            batch_size (int): Number of initial states to generate.
            model (nn.Module): Diffusion or latent diffusion model.

        Returns:
            torch.Tensor: Random initial tensor.
        """
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        
        x_start = torch.randn(batch_size, *in_shape, device=device) * 70.0 # TODO: hardcoded prior sigma
        return x_start
    
class DAPS(nn.Module):
    """
    Decoupled Annealing Posterior Sampling (DAPS) implementation.

    Combines diffusion models and MCMC updates for posterior sampling from noisy measurements.
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config):
        """
        Initializes the DAPS sampler with the provided scheduler and sampler configurations.

        Args:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            mcmc_sampler_config (dict): Configuration for MCMC sampler.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
                                                                             diffusion_scheduler_config)
        
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.mcmc_sampler = MCMCSampler(**mcmc_sampler_config)
        self.lr = 1e-4
        self.shape = (1, 3, 256, 256)

    def get_lr(self, ratio):
        """
            Computes the learning rate based on the given ratio.
        """
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (0.01 ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr
    
    def consensus_admm(
            self,
            samples, operator, measurement,
            step, sigma, xt,
            rho=1.0, tau=1.0,
            num_admm=10, inner_r=50,
            device="cuda",
            use_over_relax=True,
            alpha=1.8,                     # recommended [1.5, 1.8]
        ):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        samples = samples.to(device)
        N, C, H, W = samples.shape

        errors = ((operator(samples) - measurement)**2).mean(dim=(1,2,3))
        best = torch.argmin(errors)

        q = samples[best].clone()
        # q = torch.median(samples, dim=0)
        
        x = samples.clone().requires_grad_(True)
        u = torch.zeros_like(x)

        # Check how close x is w.r.t q 
        # print(f"Inital consensus distance : {torch.linalg.norm(q-x)}")
        ratio = step / self.annealing_scheduler.num_steps
        lr    = self.get_lr(ratio)

        meas_stacked = measurement

        # parameters from paper
        mu = 10.0
        tau_inc = 2.0
        tau_dec = 2.0

        for t in range(num_admm):

            q_old = q.clone()
            # === Local x update ===
            q_b = q.unsqueeze(0)
            u_b = u
            for _ in range(inner_r):
                grad, loss = operator.gradient(x, meas_stacked, return_loss=True)
                data_term = grad / (tau**2)
                prox_term = (rho) * (x - q_b + u_b)
                x = x - lr * (data_term + prox_term)

            # === Over-relaxation ===
            if use_over_relax:
                x_hat = alpha * x + (1 - alpha) * q_b   # broadcast q_old
            else:
                x_hat = x

            # pred = operator(x_hat)                  # [N,C,H,W]
            # residual = (pred - measurement)**2             # [N,C,H,W]

            # # collapse channels to get per-sample per-pixel score
            # loss_map_lr = residual.mean(dim=1, keepdim=True)   # [N,1,H,W]
            # loss_map = torch.nn.functional.interpolate(loss_map_lr, size=(H, W), mode="bilinear", align_corners=False) 
            # # negative scaled → good samples get higher weight
            # beta = 10.0                                # tune this
            # ll_map = -beta * loss_map                 # [N,1,H,W]

            # # softmax across particles, pixelwise
            # ll_shift = ll_map - ll_map.amax(dim=0, keepdim=True)
            # w = torch.softmax(ll_shift, dim=0)        # [N,1,H,W]

            # q update
            # q = (w * (x_hat + u)).sum(dim=0) / (w.sum(dim=0) + 1e-8)   # [C,H,W]
            errors = ((operator(x_hat) - measurement)**2).mean(dim=(1,2,3))
            best = torch.argmin(errors)

            q = x_hat[best].clone()
            
            # === Consensus update ===
            # q = (x_hat + u).mean(dim=0)

            # === Dual update ===
            u = u + (x_hat - q.unsqueeze(0))

            # print()
            # === Residuals ===
            r = x_hat - q.unsqueeze(0)       # primal residual
            r_norm = r.norm()

            s = rho * (q - q_old)            # dual residual
            s_norm = s.norm()
            # print(f"u norm : {u.norm().item()} primal norm : {r_norm.item()} Dual norm : {s_norm.item()}")
            # === Adaptive ρ ===
            if r_norm > mu * s_norm:
                rho = tau_inc * rho
            elif s_norm > mu * r_norm:
                rho = rho / tau_dec

            # NOTE: This version matches the snippet exactly:
            #   does NOT rescale u with rho change,
            #   because the paper does not include that.
            # If you want textbook scaled ADMM, we would rescale u here.
        # print(f"Final consensus distance : {torch.linalg.norm(q-x)}")
        return q.unsqueeze(0).detach()
    
    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        def add_dim(x):
            return torch.unsqueeze(x, dim=0)
        
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        xt = x_start[:,0,...]
        x0y_prev = None
        psnr_x0y = []
        psnr_x0hat = []
        
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
                x0hat = sampler.sample(xt)

            # 2. MCMC update
            # x0y = self.mcmc_sampler.sample(xt, model, x0hat, operator, measurement, sigma, step / self.annealing_scheduler.num_steps)
            
            x0y = self.consensus_admm(x0hat, operator, 
                                        measurement, 
                                        step, 
                                        sigma,
                                        xt = xt,
                                        rho=1.0, 
                                        tau=0.01, 
                                        num_admm=20,  
                                        inner_r=5, 
                                        ) 
            
            # 3. forward diffusion
            if step != self.annealing_scheduler.num_steps - 2:
                xt_1 = x0y + torch.randn(self.shape, device="cuda") * self.annealing_scheduler.sigma_steps[step + 1]
                xt_2 = x0y + torch.randn(self.shape, device="cuda") * self.annealing_scheduler.sigma_steps[step + 1]
                xt_3 = x0y + torch.randn(self.shape, device="cuda") * self.annealing_scheduler.sigma_steps[step + 1]
                xt_4 = x0y + torch.randn(self.shape, device="cuda") * self.annealing_scheduler.sigma_steps[step + 1]
                xt = torch.cat([xt_1, xt_2, xt_3, xt_4])
            else:
                # print("no noise added...")
                xt = x0y
           
            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat[0].unsqueeze(0))
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
                    psnr_x0hat.append(x0hat_results['psnr'].item())
                    psnr_x0y.append(x0y_results['psnr'].item())
                    
            if record:
                
                self._record(xt[0].unsqueeze(0), x0y, x0hat, sigma, x0hat_results, x0y_results)

        # print("PSNR x0hat:", psnr_x0hat)
        # print("PSNR x0y:", psnr_x0y)
        return xt[0].unsqueeze(0)

    # def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
    #     """Records the intermediate states during sampling."""

    #     self.trajectory.add_tensor(f'xt', xt)
    #     self.trajectory.add_tensor(f'x0y', x0y)
    #     self.trajectory.add_tensor(f'x0hat', x0hat)
    #     self.trajectory.add_value(f'sigma', sigma)
    #     for name in x0hat_results.keys():
    #         self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
    #     for name in x0y_results.keys():
    #         self.trajectory.add_value(f'x0y_{name}', x0y_results[name])
    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """Records the intermediate states during sampling."""

        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        
        num_particles, C, H, W = x0hat.shape
        if num_particles > 1:
            self.trajectory.add_tensor(f'x0hat_1', x0hat[0].unsqueeze(0))
            self.trajectory.add_tensor(f'x0hat_2', x0hat[1].unsqueeze(0))
            self.trajectory.add_tensor(f'x0hat_3', x0hat[2].unsqueeze(0))
            self.trajectory.add_tensor(f'x0hat_4', x0hat[3].unsqueeze(0))
        else:
            self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """Checks and updates the configurations for the schedulers."""

        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        return annealing_scheduler_config, diffusion_scheduler_config

    # def get_start(self, batch_size, model):
    #     """
    #     Generates initial random state tensors from the Gaussian prior.

    #     Args:
    #         batch_size (int): Number of initial states to generate.
    #         model (nn.Module): Diffusion or latent diffusion model.

    #     Returns:
    #         torch.Tensor: Random initial tensor.
    #     """
    #     device = next(model.parameters()).device
    #     in_shape = model.get_in_shape()
    #     x_start = torch.randn(batch_size, *in_shape, device=device) * self.annealing_scheduler.get_prior_sigma()
    #     return x_start

    def get_start(self, batch_size, model):
        """
        Generates initial random state tensors from the Gaussian prior.

        Args:
            batch_size (int): Number of initial states to generate.
            model (nn.Module): Diffusion or latent diffusion model.

        Returns:
            torch.Tensor: Random initial tensor.
        """
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        
        x_start_1 = torch.randn(batch_size, *in_shape, device=device) *  self.annealing_scheduler.get_prior_sigma()
        x_start_2 = torch.randn(batch_size, *in_shape, device=device) *  self.annealing_scheduler.get_prior_sigma()
        x_start_3 = torch.randn(batch_size, *in_shape, device=device) *  self.annealing_scheduler.get_prior_sigma()
        x_start_4 = torch.randn(batch_size, *in_shape, device=device) *  self.annealing_scheduler.get_prior_sigma()
        
        
        return torch.stack([x_start_1, x_start_2, x_start_3, x_start_4])
