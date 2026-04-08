from abc import ABC, abstractmethod
from .resizer import Resizer
import torch.nn.functional as F

import torch
import torch.nn as nn
import scipy
import numpy as np
import yaml
import warnings
from torch.autograd import grad

__OPERATOR__ = {}


def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            if __OPERATOR__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __OPERATOR__[name] = cls
        cls.name = name
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class Operator(ABC):
    """
    Abstract base class for operators in diffusion processes.

    Attributes:
        sigma (float): Standard deviation of measurement noise.
    """
    def __init__(self, sigma=0.05):
        """
        Initializes the operator with a noise standard deviation.

        Args:
            sigma (float, optional): Measurement noise level. Defaults to 0.05.
        """
        self.sigma = sigma

    @abstractmethod
    def __call__(self, x):
        """
        Abstract method: apply operator to input data.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Output after applying the operator.
        """
        pass

    def measure(self, x):
        """
        Measures input data by applying the operator and adding Gaussian noise.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Noisy measurement result.
        """
        y0 = self(x)
        return y0 + self.sigma * torch.randn_like(y0)

    def loss(self, x, y):
        """
        Computes squared-error loss between operator output and observed data.

        Args:
            x (torch.Tensor): Input data tensor.
            y (torch.Tensor): Observed measurement tensor.

        Returns:
            torch.Tensor: Loss values (one per sample).
        """
        return ((self(x) - y) ** 2).flatten(1).sum(-1)

    def gradient(self, x, y, return_loss=False):
        """
        Computes gradient of the loss with respect to input x.

        Args:
            x (torch.Tensor): Input tensor requiring gradient.
            y (torch.Tensor): Observed measurements.
            return_loss (bool, optional): If True, returns both gradient and loss. Defaults to False.

        Returns:
            torch.Tensor or tuple: Gradient tensor (and optionally the loss value).
        """
        x_tmp = x.clone().detach().requires_grad_(True)
        loss = self.loss(x_tmp, y).sum()
        x_grad = torch.autograd.grad(loss, x_tmp)[0]
        if return_loss:
            return x_grad, loss
        return x_grad

    def log_likelihood(self, x, y):
        return -self.loss(x, y) / 2 / self.sigma ** 2

    def likelihood(self, x, y):
        return torch.exp(self.log_likelihood(x, y))


@register_operator(name='down_sampling')
class DownSampling(Operator):
    def __init__(self, resolution=256, scale_factor=4, device='cuda', sigma=0.05):
        super().__init__(sigma)
        in_shape = [1, 3, resolution, resolution]
        self.down_sample = Resizer(in_shape, 1 / scale_factor).to(device)

    def upsample(self, x):
        in_shape = x.shape
        up_sample = Resizer(in_shape, scale_factor=4).to(x.device)
        return up_sample(x)
    
    def __call__(self, x):
        return self.down_sample(x)


def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t + h, l:l + w] = 0

    return mask, t, t + h, l, l + w


class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(32, 32)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                                            mask_shape=(mask_h, mask_w),
                                            image_size=self.image_size,
                                            margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask


@register_operator(name='inpainting')
class Inpainting(Operator):
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None, resolution=256, device='cuda',
                 sigma=0.05):
        super().__init__(sigma)
        self.mask_gen = mask_generator(mask_type, mask_len_range, mask_prob_range, resolution)
        self.mask = None  # [B, 1, H, W]

    def __call__(self, x):
        # print(x.shape)
        if self.mask is None:
            self.mask = self.mask_gen(x)
            self.mask = self.mask[0:1, 0:1, :, :]
        obs = x * self.mask
       
        return obs

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        assert self.mask is not None, "Mask not initialized; call the operator once or regenerate()."
        return y * self.mask

    def upsample(self, y: torch.Tensor) -> torch.Tensor:
        # alias to adjoint to match your existing call sites
        return self.adjoint(y)
    

class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


@register_operator(name='gaussian_blur')
class GaussianBlur(Operator):
    def __init__(self, kernel_size, intensity, device='cuda', sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.conv.requires_grad_(False)

    def __call__(self, data):
        return self.conv(data)


############################################################## Operators with SVD support ##############################################################

class A_functions:
    """
    A class replacing the SVD of a matrix A, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()
    
    def measure(self, vec):
        """
        Multiplies the input vector by A
        """
        # temp = self.Vt(vec)
        # singulars = self.singulars()
        # return self.U(singulars * temp[:, :singulars.shape[0]])
        raise NotImplementedError()
    
    def At(self, vec):
        """
        Multiplies the input vector by A transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def A_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of A
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        
        factors = 1. / singulars
        factors[singulars == 0] = 0.
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def A_pinv_add_eta(self, vec, eta):
        """
        Multiplies the input vector by the pseudo inverse of A with factor eta
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        factors = singulars / (singulars*singulars+eta)
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def AtA_add_eta_inv(self, vec, eta=1e-6):
        """
        Multiplies the input vector by the pseudo inverse of A with loading eta
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        extended_singulars = self.add_zeros(singulars[None,:])
        factors = 1 / (extended_singulars*extended_singulars + eta)

        temp = temp * factors
        return self.V(temp)
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        raise NotImplementedError()

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        raise NotImplementedError()
    
@register_operator(name='avg_pooling')
class SuperResolution(A_functions):
    def __init__(self, channels, img_dim, ratio, device, sigma=0.05): #ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        self.sigma_y = sigma
        A = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(A, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        #reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim**2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio**2-1):
            patches[:, :, :, idx+1] = temp[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1].view(vec.shape[0], self.channels, -1)
        #multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        #extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, :self.channels * self.y_dim**2] = patches[:, :, :, 0].view(vec.shape[0], self.channels * self.y_dim**2)
        for idx in range(self.ratio**2-1):
            recon[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1] = patches[:, :, :, idx+1].view(vec.shape[0], self.channels * self.y_dim**2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        singulars = self.singulars_small
        
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        
        lambda_t = torch.ones(self.ratio ** 2, device=vec.device)
        
        temp = torch.zeros(self.ratio ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.
        
        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y)
            
        lambda_t = lambda_t.reshape(1, 1, 1, -1)

        patches = patches * lambda_t
        
        
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1))
        
        patches = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches = patches.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
        return patches
    
    def Lambda_edm(self, vec, sigma, sigma_y, eps=1e-8):
        """
        EDM-native Lambda for SVD-supported operators.

        Args:
            vec     : Tensor [B, C*H*W] or [B, C, H, W] (image-space residual/backproj)
            sigma   : float or tensor scalar, EDM noise level at this step
            sigma_y : float, measurement noise std (in measurement domain)

        Returns:
            Tensor [B, C*H*W]  (same layout as your original Lambda)
        """
        # ensure tensor scalars
        device = vec.device
        dtype = vec.dtype
        sigma   = torch.as_tensor(sigma, device=device, dtype=dtype)
        sigma_y = torch.as_tensor(sigma_y, device=device, dtype=dtype)

        # reshape vec to BCHW
        if vec.ndim == 2:
            B = vec.shape[0]
            patches = vec.reshape(B, self.channels, self.img_dim, self.img_dim)
        else:
            B = vec.shape[0]
            patches = vec

        # ---- extract patches: [B, C, n_patches, r^2]
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches = patches.contiguous().reshape(B, self.channels, -1, self.ratio**2)

        # ---- rotate into SVD basis: V^T
        patches = torch.matmul(self.Vt_small.to(dtype), patches.reshape(-1, self.ratio**2, 1))
        patches = patches.reshape(B, self.channels, -1, self.ratio**2)

        # ---- build padded singulars: length r^2
        s = torch.zeros(self.ratio**2, device=device, dtype=dtype)
        s[: self.singulars_small.numel()] = self.singulars_small.to(dtype)

        # ---- EDM weighting in SVD coords (measured dims only)
        # w_i(sigma) = s_i^2 / (s_i^2 + (sigma_y/sigma)^2)
        # if s_i = 0 -> w_i = 0
        inv_sigma = 1.0 / torch.clamp(sigma, min=eps)
        r = sigma_y * inv_sigma  # (sigma_y / sigma)
        w = (s**2) / (s**2 + r**2 + eps)

        w = w.view(1, 1, 1, -1)  # broadcast over [B,C,n_patches,r^2]
        patches = patches * w

        # ---- rotate back: V
        patches = torch.matmul(self.V_small.to(dtype), patches.reshape(-1, self.ratio**2, 1))
        patches = patches.reshape(B, self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches = patches.reshape(B, self.channels * self.img_dim**2)

        return patches

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        singulars = self.singulars_small
        
        patches_vec = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches_vec = patches_vec.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches_vec = patches_vec.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        patches_eps = epsilon.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches_eps = patches_eps.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches_eps = patches_eps.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        d1_t = torch.ones(self.ratio ** 2, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.ratio ** 2, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5
        
        temp = torch.zeros(self.ratio ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.
        
        if a != 0 and sigma_y != 0:
            
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index  * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)
             
            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(change_index * (sigma_t ** 2 - a ** 2 * sigma_y ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)
            
            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5
        
        d1_t = d1_t.reshape(1, 1, 1, -1)
        d2_t = d2_t.reshape(1, 1, 1, -1)
        patches_vec = patches_vec * d1_t
        patches_eps = patches_eps * d2_t
        
        patches_vec = torch.matmul(self.V_small, patches_vec.reshape(-1, self.ratio**2, 1))
        
        patches_vec = patches_vec.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches_vec = patches_vec.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_vec = patches_vec.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
        patches_eps = torch.matmul(self.V_small, patches_eps.reshape(-1, self.ratio**2, 1))
        
        patches_eps = patches_eps.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches_eps = patches_eps.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_eps = patches_eps.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
        return patches_vec + patches_eps  
    
    @torch.no_grad()
    def Lambda_noise_edm(
        self,
        z_img: torch.Tensor,   # [B,C,H,W] ~ N(0,I)
        sigma: float,          # EDM noise level
        sigma_y: float,        # measurement noise std (in measurement domain)
        sigma_cut: float = 0.03,   # start collapsing extra noise below this
        dtype=None,
    ) -> torch.Tensor:
        """
        EDM-native shaped noise for SR average pooling.

        Returns delta such that x_sigma = x0 + delta, with correct measurement-aware covariance,
        and delta -> 0 as sigma -> 0.
        """
        assert z_img.ndim == 4, f"expected BCHW, got {z_img.shape}"
        B, C, H, W = z_img.shape
        assert H == self.img_dim and W == self.img_dim and C == self.channels

        device = z_img.device
        if dtype is None:
            dtype = z_img.dtype

        sigma   = torch.as_tensor(sigma, device=device, dtype=dtype)
        sigma_y = torch.as_tensor(sigma_y, device=device, dtype=dtype)

        # ---- Late-stage collapse gate (prevents end-of-chain re-noising) ----
        # gate = 1 for sigma >= sigma_cut, and linearly -> 0 as sigma -> 0
        gate = torch.clamp(sigma / sigma_cut, max=1.0)

        # ---- Extract patches: [B, C, n_patches, r^2] ----
        patches = z_img.to(dtype).unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches = patches.contiguous().reshape(B, C, -1, self.ratio**2)

        # ---- Rotate into SVD basis: V^T ----
        patches = torch.matmul(self.Vt_small.to(dtype), patches.reshape(-1, self.ratio**2, 1))
        patches = patches.reshape(B, C, -1, self.ratio**2)

        # ---- Build singular values padded to r^2 ----
        s = torch.zeros(self.ratio**2, device=device, dtype=dtype)
        s[: self.singulars_small.numel()] = self.singulars_small.to(dtype)

        nz = s > 0
        inv_s = torch.zeros_like(s)
        inv_s[nz] = 1.0 / s[nz]

        # ---- Per-coordinate std (EDM) ----
        std = torch.full_like(s, sigma)       # null-space default
        std[nz] = torch.minimum(std[nz], sigma_y * inv_s[nz])  # measured cap

        # ---- Apply late-stage collapse everywhere ----
        std = std * gate

        std = std.view(1, 1, 1, -1)  # broadcast over [B,C,n_patches,r^2]
        patches = patches * std

        # ---- Rotate back: V ----
        patches = torch.matmul(self.V_small.to(dtype), patches.reshape(-1, self.ratio**2, 1))
        patches = patches.reshape(B, C, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        delta = patches.reshape(B, C, H, W)

        return delta
    
    def measure(self, vec):
        # A * x
        temp = self.Vt(vec)
        singulars = self.singulars()
        y0 = self.U(singulars * temp[:, :singulars.shape[0]])
        y0_noise = y0 + self.sigma_y * torch.randn_like(y0)   
        return y0_noise
    
    def forward(self, vec):
        # A * x
        temp = self.Vt(vec)
        singulars = self.singulars()
        y0 = self.U(singulars * temp[:, :singulars.shape[0]])
        return y0

@register_operator(name='gaussian_deblur_svd') 
class Deblurring(A_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, sigma, channels, img_dim, device, ZERO = 3e-2, sigma_y = 0.05):
        self.img_dim = img_dim
        self.channels = channels
        self.sigma_y = sigma_y
        self.sigma = sigma
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / self.sigma) ** 2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        kernel = kernel / kernel.sum()
        
        print("Deblurring sigma:", self.sigma)
        print("sigma_y:", self.sigma_y)
        #build 1D conv matrix
        A_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                A_small[i, j] = kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(A_small, some=False)
        #ZERO = 3e-2
        self.singulars_small_orig = self.singulars_small.clone()
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars_orig = torch.matmul(self.singulars_small_orig.reshape(img_dim, 1), self.singulars_small_orig.reshape(1, img_dim)).reshape(img_dim**2)
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1), self.singulars_small.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)
        self._singulars_orig = self._singulars_orig[self._perm]

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)
    
    def A_pinv(self, vec):
        temp = self.Ut(vec)
        singulars = self._singulars.repeat(1, 3).reshape(-1)
        
        factors = 1. / singulars
        factors[singulars == 0] = 0.
        
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        temp_vec = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp_vec = self.img_by_mat(temp_vec, self.V_small).reshape(vec.shape[0], self.channels, -1)
        temp_vec = temp_vec[:, :, self._perm].permute(0, 2, 1)

        singulars = self._singulars_orig
        lambda_t = torch.ones(self.img_dim ** 2, device=vec.device)
        temp_singulars = torch.zeros(self.img_dim ** 2, device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                    singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y)

        lambda_t = lambda_t.reshape(1, -1, 1)
        temp_vec = temp_vec * lambda_t

        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = temp_vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        temp_vec = vec.clone().reshape(vec.shape[0], self.channels, -1)
        temp_vec = temp_vec[:, :, self._perm].permute(0, 2, 1)

        temp_eps = epsilon.clone().reshape(vec.shape[0], self.channels, -1)
        temp_eps = temp_eps[:, :, self._perm].permute(0, 2, 1)

        singulars = self._singulars_orig
        d1_t = torch.ones(self.img_dim ** 2, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.img_dim ** 2, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5

        temp_singulars = torch.zeros(self.img_dim ** 2, device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_y ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, -1, 1)
        d2_t = d2_t.reshape(1, -1, 1)

        temp_vec = temp_vec * d1_t
        temp_eps = temp_eps * d2_t

        temp_vec_new = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp_vec_new[:, self._perm, :] = temp_vec
        out_vec = self.mat_by_img(self.V_small, temp_vec_new.permute(0, 2, 1))
        out_vec = self.img_by_mat(out_vec, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)

        temp_eps_new = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp_eps_new[:, self._perm, :] = temp_eps
        out_eps = self.mat_by_img(self.V_small, temp_eps_new.permute(0, 2, 1))
        out_eps = self.img_by_mat(out_eps, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)

        return out_vec + out_eps
    
    @torch.no_grad()
    def Lambda_edm(self, vec, sigma, sigma_y, eps=1e-8):
        """
        EDM-native Lambda for SVD operators.

        Args:
            vec    : Tensor [B, C, H, W] OR [B, C*H*W] (image-space backprojected residual)
            sigma  : scalar EDM noise level (float or tensor)
            sigma_y: measurement noise std in measurement domain
        Returns:
            Tensor shaped like vec (same dtype/device), representing a scaled correction.
        """
        device = vec.device
        dtype  = vec.dtype
        sigma   = torch.as_tensor(sigma, device=device, dtype=dtype).clamp_min(eps)
        sigma_y = torch.as_tensor(sigma_y, device=device, dtype=dtype)

        # Ensure vec is BCHW then flatten later if needed
        flatten_out = False
        if vec.ndim == 2:
            flatten_out = True
            B = vec.shape[0]
            vec_img = vec.reshape(B, self.channels, self.img_dim, self.img_dim)
        else:
            vec_img = vec

        # ---- Go to "V^T basis with perm" (matches your Lambda/Lambda_noise internal basis) ----
        # This yields temp_vec with shape [B, img_dim^2, C] (via permuting)
        temp_vec = self.mat_by_img(self.V_small.transpose(0, 1), vec_img.clone())
        temp_vec = self.img_by_mat(temp_vec, self.V_small).reshape(vec_img.shape[0], self.channels, -1)
        temp_vec = temp_vec[:, :, self._perm].permute(0, 2, 1)  # [B, img_dim^2, C]

        # Singulars for big operator (orig, before ZEROing) padded to img_dim^2
        s = self._singulars_orig.to(device=device, dtype=dtype)
        if s.numel() < self.img_dim**2:
            s_pad = torch.zeros(self.img_dim**2, device=device, dtype=dtype)
            s_pad[:s.numel()] = s
            s = s_pad
        else:
            s = s[:self.img_dim**2]

        # EDM weight in each singular direction
        r = sigma_y / sigma  # scalar
        w = (s**2) / (s**2 + r**2 + eps)  # [img_dim^2]
        w = w.view(1, -1, 1)              # broadcast over channels

        temp_vec = temp_vec * w

        # ---- Map back to image space (invert perm, apply V and V^T) ----
        temp = torch.zeros(vec_img.shape[0], self.img_dim**2, self.channels, device=device, dtype=dtype)
        temp[:, self._perm, :] = temp_vec.reshape(vec_img.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)  # [B, C, img_dim^2]

        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1))  # [B, C, H, W]

        if flatten_out:
            return out.reshape(vec.shape[0], -1)
        return out
    
    @torch.no_grad()
    def Lambda_noise_edm(self, z, sigma, sigma_y, sigma_cut=0.03, eps=1e-8):
        """
        EDM-native shaped noise for SVD operators.

        Args:
            z        : Tensor [B, C, H, W] ~ N(0,1) OR [B, C*H*W]
            sigma    : EDM noise level (scalar)
            sigma_y  : measurement noise std in measurement domain
            sigma_cut: below this sigma, aggressively collapse noise -> 0
        Returns:
            delta with same shape as z (image-space noise to add: x = x0 + delta)
        """
        device = z.device
        dtype  = z.dtype
        sigma   = torch.as_tensor(sigma, device=device, dtype=dtype).clamp_min(0.0)
        sigma_y = torch.as_tensor(sigma_y, device=device, dtype=dtype)

        # collapse gate to avoid tail noise
        gate = torch.clamp(sigma / max(float(sigma_cut), eps), max=1.0)

        # Ensure BCHW
        flatten_out = False
        if z.ndim == 2:
            flatten_out = True
            B = z.shape[0]
            z_img = z.reshape(B, self.channels, self.img_dim, self.img_dim)
        else:
            z_img = z

        # ---- Transform z into the same singular basis used by Lambda_noise ----
        # Note: for noise, we can use V^T basis similarly
        temp_vec = self.mat_by_img(self.V_small.transpose(0, 1), z_img.clone())
        temp_vec = self.img_by_mat(temp_vec, self.V_small).reshape(z_img.shape[0], self.channels, -1)
        temp_vec = temp_vec[:, :, self._perm].permute(0, 2, 1)  # [B, img_dim^2, C]

        # Singulars (padded)
        s = self._singulars_orig.to(device=device, dtype=dtype)
        if s.numel() < self.img_dim**2:
            s_pad = torch.zeros(self.img_dim**2, device=device, dtype=dtype)
            s_pad[:s.numel()] = s
            s = s_pad
        else:
            s = s[:self.img_dim**2]

        # std per direction
        std = torch.full_like(s, sigma)  # default (null)
        nz = s > 0
        std[nz] = torch.minimum(std[nz], sigma_y / (s[nz] + eps))  # cap measured

        # late-stage collapse everywhere
        std = std * gate
        std = std.view(1, -1, 1)  # [1, img_dim^2, 1]

        temp_vec = temp_vec * std

        # ---- Back to image space ----
        temp = torch.zeros(z_img.shape[0], self.img_dim**2, self.channels, device=device, dtype=dtype)
        temp[:, self._perm, :] = temp_vec.reshape(z_img.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)

        delta = self.mat_by_img(self.V_small, temp)
        delta = self.img_by_mat(delta, self.V_small.transpose(0, 1))  # [B,C,H,W]

        if flatten_out:
            return delta.reshape(z.shape[0], -1)
        return delta
    
    def measure(self, vec):
        # A * x
        temp = self.Vt(vec)
        singulars = self.singulars()
        y0 = self.U(singulars * temp[:, :singulars.shape[0]])
        y0_noise = y0 + self.sigma_y * torch.randn_like(y0)   
        return y0_noise
    
    def forward(self, vec):
        # A * x
        temp = self.Vt(vec)
        singulars = self.singulars()
        y0 = self.U(singulars * temp[:, :singulars.shape[0]])
        return y0

@register_operator(name='inpainting_svd')
class Inpainting(A_functions):
    def __init__(self, channels, img_dim, corr_type, corr_ratio, device, sigma_y = 0.05):
        self.channels = channels
        self.img_dim = img_dim
        self.corr_type = corr_type
        self.corr_ratio = corr_ratio
        self.sigma_y = sigma_y
        if corr_type == 'random':
            def generate_random_mask_ratio(H, W, ratio, device='cuda'):
                rand_map = torch.rand((H, W), device=device)
                mask = (rand_map > ratio).float()   
                return mask
            
            ratio = corr_ratio   # e.g., 0.4 for 40% missing
            H = W = img_dim
            mask = generate_random_mask_ratio(H, W, ratio, device=device) 
            mask_flat = mask.reshape(-1)    # 1 = visible, 0 = missing
            missing_r = torch.nonzero(mask_flat == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_r + 2
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0).to(device)
        elif corr_type == 'box':
            H = W = img_dim   # e.g., 2
            mask = np.ones((H, W), dtype=np.float32)
            crop = 128
            start = (H - crop) // 2
            end   = start + crop

            mask[start:end, start:end] = 0.0   # zero out center block
            mask = torch.from_numpy(mask).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_r + 2

            missing = torch.cat([missing_r, missing_g, missing_b], dim=0).to(device)
        
        self._singulars = torch.ones(channels * img_dim**2 - missing.shape[0]).to(device)
        self.missing_indices = missing
        self.kept_indices = torch.Tensor([i for i in range(channels * img_dim**2) if i not in missing]).to(device).long()

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, :self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0]:]
        return out.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):

        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]

        singulars = self._singulars
        lambda_t = torch.ones(temp.size(1), device=vec.device)
        temp_singulars = torch.zeros(temp.size(1), device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                    singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y)

        lambda_t = lambda_t.reshape(1, -1)
        out = out * lambda_t

        result = torch.zeros_like(temp)
        result[:, self.kept_indices] = out[:, :self.kept_indices.shape[0]]
        result[:, self.missing_indices] = out[:, self.kept_indices.shape[0]:]
        return result.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        temp_vec = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out_vec = torch.zeros_like(temp_vec)
        out_vec[:, :self.kept_indices.shape[0]] = temp_vec[:, self.kept_indices]
        out_vec[:, self.kept_indices.shape[0]:] = temp_vec[:, self.missing_indices]

        temp_eps = epsilon.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out_eps = torch.zeros_like(temp_eps)
        out_eps[:, :self.kept_indices.shape[0]] = temp_eps[:, self.kept_indices]
        out_eps[:, self.kept_indices.shape[0]:] = temp_eps[:, self.missing_indices]

        singulars = self._singulars
        d1_t = torch.ones(temp_vec.size(1), device=vec.device) * sigma_t * eta
        d2_t = torch.ones(temp_vec.size(1), device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5

        temp_singulars = torch.zeros(temp_vec.size(1), device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_y ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, -1)
        d2_t = d2_t.reshape(1, -1)
        out_vec = out_vec * d1_t
        out_eps = out_eps * d2_t

        result_vec = torch.zeros_like(temp_vec)
        result_vec[:, self.kept_indices] = out_vec[:, :self.kept_indices.shape[0]]
        result_vec[:, self.missing_indices] = out_vec[:, self.kept_indices.shape[0]:]
        result_vec = result_vec.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

        result_eps = torch.zeros_like(temp_eps)
        result_eps[:, self.kept_indices] = out_eps[:, :self.kept_indices.shape[0]]
        result_eps[:, self.missing_indices] = out_eps[:, self.kept_indices.shape[0]:]
        result_eps = result_eps.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)
        
        return result_vec + result_eps
    
    @torch.no_grad()
    def Lambda_edm(self, vec, sigma, sigma_y, eps=1e-8):
        """
        EDM-native Lambda for inpainting.

        Args:
            vec     : Tensor [B, C*H*W] or [B, C, H, W] (image-space backproj / residual)
            sigma   : scalar EDM noise level
            sigma_y : measurement noise std in measurement space
        Returns:
            Tensor same shape as vec
        """
        device = vec.device
        dtype  = vec.dtype
        sigma   = torch.as_tensor(sigma, device=device, dtype=dtype).clamp_min(eps)
        sigma_y = torch.as_tensor(sigma_y, device=device, dtype=dtype)

        flatten_out = False
        if vec.ndim == 4:
            B = vec.shape[0]
            temp = vec.reshape(B, self.channels, -1).permute(0, 2, 1).reshape(B, -1)
        else:
            B = vec.shape[0]
            flatten_out = True
            temp = vec.clone().reshape(B, -1)

        # EDM weight for singular=1 directions
        r = sigma_y / sigma
        w = 1.0 / (1.0 + r * r)   # scalar in [0,1]

        out = torch.zeros_like(temp)
        # only measured coords (kept pixels) get scaled; missing pixels get 0 update
        out[:, self.kept_indices] = temp[:, self.kept_indices] * w
        out[:, self.missing_indices] = 0.0

        if flatten_out:
            return out
        # back to BCHW
        return out.reshape(B, -1, self.channels).permute(0, 2, 1).reshape(B, self.channels, self.img_dim, self.img_dim)
    
    @torch.no_grad()
    def Lambda_noise_edm(self, z, sigma, sigma_y, sigma_cut=0.03, eps=1e-8):
        """
        EDM-native shaped noise for inpainting.

        Args:
            z         : Tensor [B, C*H*W] or [B, C, H, W] ~ N(0,1)
            sigma     : EDM noise level
            sigma_y   : measurement noise std (measurement space)
            sigma_cut : below this, force noise to collapse strongly (prevents end-stage re-noising)
        Returns:
            delta with same shape as z
        """
        device = z.device
        dtype  = z.dtype
        sigma   = torch.as_tensor(sigma, device=device, dtype=dtype).clamp_min(0.0)
        sigma_y = torch.as_tensor(sigma_y, device=device, dtype=dtype)

        # late-stage collapse (critical)
        gate = torch.clamp(sigma / max(float(sigma_cut), eps), max=1.0)

        flatten_out = False
        if z.ndim == 4:
            B = z.shape[0]
            temp = z.reshape(B, self.channels, -1).permute(0, 2, 1).reshape(B, -1)
        else:
            B = z.shape[0]
            flatten_out = True
            temp = z.clone().reshape(B, -1)

        # std on kept pixels is capped by measurement noise (singular=1)
        sigma_kept = torch.minimum(sigma, sigma_y) * gate
        # std on missing pixels is sigma (null-space), but still collapses via gate
        sigma_missing = sigma * gate

        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, self.kept_indices] * sigma_kept
        out[:, self.missing_indices] = temp[:, self.missing_indices] * sigma_missing

        if flatten_out:
            return out
        return out.reshape(B, -1, self.channels).permute(0, 2, 1).reshape(B, self.channels, self.img_dim, self.img_dim)
    
    def measure(self, vec):
        # A * x
        temp = self.Vt(vec)
        singulars = self.singulars()
        y0 = self.U(singulars * temp[:, :singulars.shape[0]])
        y0_noise = y0 + self.sigma_y * torch.randn_like(y0)   
        return y0_noise
    
    def forward(self, vec):
        # A * x
        temp = self.Vt(vec)
        singulars = self.singulars()
        y0 = self.U(singulars * temp[:, :singulars.shape[0]])
        return y0