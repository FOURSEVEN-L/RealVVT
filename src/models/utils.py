import torch
from einops import rearrange


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def rand_log_normal(
    shape, 
    loc=0., 
    scale=1., 
    device='cpu', 
    dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def add_noise(latents, P_mean, P_std):
    """
    latents shape: BxTxCxHxW
    """
    noise = torch.randn_like(latents)

    sigma = rand_log_normal(
        [latents.shape[0], 1, 1, 1, 1],
        loc=P_mean,
        scale=P_std,
        device=latents.device,
        dtype=latents.dtype,
    )
    noisy_inputs = latents + noise * sigma
    return noisy_inputs, sigma


def pixel2latent(pixel_values, vae):
    """
    pixel shape: BxTxCxHxW
    """
    video_length = pixel_values.shape[1]
    with torch.no_grad():
        # encode each video to avoid OOM
        latents = []
        for pixel_value in pixel_values:
            latent = vae.encode(pixel_value).latent_dist.sample()
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor
    return latents


def encode_image(pixel_values, feature_extractor, image_encoder):
    """
    pixel_values: BxCxHxW, [-1, 1]
    """
    pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
    pixel_values = (pixel_values + 1.0) / 2.0

    # Normalize the image with for CLIP input
    pixel_values = feature_extractor(
        images=pixel_values,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values

    pixel_values = pixel_values.to(image_encoder.device)

    with torch.no_grad():
        image_embeddings = image_encoder(pixel_values).image_embeds
    image_embeddings = image_embeddings.unsqueeze(dim=1)
    return image_embeddings


def get_add_time_ids(
    fps,
    motion_bucket_id,
    noise_aug_strength,
    dtype,
    batch_size,
):
    motion_bucket_id = torch.tensor(motion_bucket_id.repeat(batch_size, 1))
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids
    

def scaling_with_edm_noise(sigma):
    c_skip = 1.0 / (sigma ** 2 + 1.0)
    c_out = -sigma / (sigma ** 2 + 1.0) ** 0.5
    c_in = 1.0 / (sigma ** 2 + 1.0) ** 0.5
    c_noise = torch.Tensor(
        [0.25 * sigma.log() for sigma in sigma]).to(sigma.device)
    return c_skip, c_out, c_in, c_noise



def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
                      dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out