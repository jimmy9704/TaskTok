import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


TITOK_IMAGE_SIZE_ADAPTER_TRAINABLE_LAYER_PREFIXES = ("encoder.patch_embed.", "decoder.ffn.0.")


def replace_encoder_patch_embed(encoder, image_size):
    old_patch = encoder.patch_size
    grid_size = encoder.grid_size
    if image_size % grid_size != 0:
        raise ValueError(f"image_size must be divisible by encoder grid_size={grid_size}")

    new_patch = image_size // grid_size
    if new_patch == old_patch:
        encoder.image_size = image_size
        return

    old = encoder.patch_embed
    new = nn.Conv2d(
        old.in_channels,
        old.out_channels,
        kernel_size=new_patch,
        stride=new_patch,
        bias=old.bias is not None,
    ).to(device=old.weight.device, dtype=old.weight.dtype)

    with torch.no_grad():
        weight = F.interpolate(
            old.weight.float(),
            size=(new_patch, new_patch),
            mode="bicubic",
            align_corners=False,
        )
        weight = weight * (old_patch / new_patch) ** 2
        new.weight.copy_(weight.to(dtype=old.weight.dtype))
        if old.bias is not None:
            new.bias.copy_(old.bias)

    encoder.patch_embed = new
    encoder.image_size = image_size
    encoder.patch_size = new_patch
    encoder.grid_size = grid_size


def replace_decoder_patch_projection(decoder, image_size):
    if decoder.is_legacy:
        raise ValueError("This simple adapter expects non-legacy TiTok decoder output.")

    old_patch = decoder.patch_size
    grid_size = decoder.grid_size
    if image_size % grid_size != 0:
        raise ValueError(f"image_size must be divisible by decoder grid_size={grid_size}")

    new_patch = image_size // grid_size
    if new_patch == old_patch:
        decoder.image_size = image_size
        return

    old = decoder.ffn[0]
    if not isinstance(old, nn.Conv2d) or old.kernel_size != (1, 1):
        raise ValueError("Expected decoder.ffn[0] to be the final 1x1 patch projection.")

    in_channels = old.in_channels
    new = nn.Conv2d(
        in_channels,
        new_patch * new_patch * 3,
        kernel_size=1,
        bias=old.bias is not None,
    ).to(device=old.weight.device, dtype=old.weight.dtype)

    with torch.no_grad():
        weight = old.weight[:, :, 0, 0]
        weight = weight.reshape(old_patch, old_patch, 3, in_channels).permute(2, 3, 0, 1)
        weight = F.interpolate(
            weight.reshape(3 * in_channels, 1, old_patch, old_patch).float(),
            size=(new_patch, new_patch),
            mode="bicubic",
            align_corners=False,
        )
        weight = weight.reshape(3, in_channels, new_patch, new_patch)
        weight = weight.permute(2, 3, 0, 1).reshape(new_patch * new_patch * 3, in_channels, 1, 1)
        new.weight.copy_(weight.to(dtype=old.weight.dtype))

        if old.bias is not None:
            bias = old.bias.reshape(old_patch, old_patch, 3).permute(2, 0, 1).unsqueeze(0)
            bias = F.interpolate(
                bias.float(),
                size=(new_patch, new_patch),
                mode="bicubic",
                align_corners=False,
            )
            bias = bias.squeeze(0).permute(1, 2, 0).reshape(new_patch * new_patch * 3)
            new.bias.copy_(bias.to(dtype=old.bias.dtype))

    decoder.ffn[0] = new
    decoder.ffn[1] = Rearrange(
        "b (p1 p2 c) h w -> b c (h p1) (w p2)",
        p1=new_patch,
        p2=new_patch,
    )
    decoder.image_size = image_size
    decoder.patch_size = new_patch
    decoder.grid_size = grid_size


def adapt_titok_to_image_size(model, image_size):
    replace_encoder_patch_embed(model.encoder, image_size)
    replace_decoder_patch_projection(model.decoder, image_size)
    model.config.dataset.preprocessing.crop_size = image_size
    model.config.model.vq_model.vit_enc_patch_size = model.encoder.patch_size
    model.config.model.vq_model.vit_dec_patch_size = model.decoder.patch_size


def build_reference_encoder_quantizer(model):
    reference_encoder = copy.deepcopy(model.encoder).eval()
    reference_quantize = copy.deepcopy(model.quantize).eval()
    for param in reference_encoder.parameters():
        param.requires_grad_(False)
    for param in reference_quantize.parameters():
        param.requires_grad_(False)
    return reference_encoder, reference_quantize, reference_encoder.image_size


def vq_logits_and_targets(z, quantize, target_indices):
    z_flat = z.float().permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    embedding = quantize.embedding.weight.float()
    if quantize.use_l2_norm:
        z_flat = F.normalize(z_flat, dim=-1)
        embedding = F.normalize(embedding, dim=-1)

    distances = (
        z_flat.pow(2).sum(dim=1, keepdim=True)
        + embedding.pow(2).sum(dim=1).unsqueeze(0)
        - 2 * z_flat @ embedding.t()
    )
    return -distances, target_indices.reshape(-1).long()


def trainable_titok_image_size_adapter_params(model):
    model.eval()
    model.encoder.eval()
    model.quantize.eval()
    model.decoder.eval()
    model.encoder.patch_embed.train()
    model.decoder.ffn[0].train()

    params = []
    trainable_names = []
    for name, param in model.named_parameters():
        trainable = name.startswith(TITOK_IMAGE_SIZE_ADAPTER_TRAINABLE_LAYER_PREFIXES)
        param.requires_grad_(trainable)
        if trainable:
            params.append(param)
            trainable_names.append(name)

    if not params:
        raise RuntimeError(
            f"No trainable params found for {TITOK_IMAGE_SIZE_ADAPTER_TRAINABLE_LAYER_PREFIXES}"
        )
    return params, trainable_names


# Backward-compatible aliases for older local scripts.
TITOK_512_TRAINABLE_LAYER_PREFIXES = TITOK_IMAGE_SIZE_ADAPTER_TRAINABLE_LAYER_PREFIXES
trainable_titok_512_adapter_params = trainable_titok_image_size_adapter_params

# -------------------------
# Short-side based patch extraction (no padding, max overlap)
# -------------------------
def _compute_patch_starts(long_side: int, short_side: int):
    """
    Compute patch start positions along the long side.
    Patches have size short_side x short_side.
    Maximize overlap by distributing patches evenly.
    """
    if long_side <= short_side:
        return [0]
    
    n_patches = math.ceil(long_side / short_side)
    
    if n_patches == 1:
        return [0]
    
    # Distribute patches evenly to maximize overlap
    # Last patch must end at long_side, so last_start = long_side - short_side
    last_start = long_side - short_side
    
    # Evenly distribute starts from 0 to last_start
    starts = []
    for i in range(n_patches):
        start = int(round(i * last_start / (n_patches - 1)))
        starts.append(start)
    
    return starts


def prepare_tasktok_input(image_list, target_size=256):
    """
    Extract short-side-based square patches from each image and resize to target_size.
    
    Args:
        image_list: List of tensors (C, H, W)
        target_size: Size to resize each patch to
        
    Returns:
        patches: (N, C, target_size, target_size) batch tensor
        metas: List of metadata for reconstruction
    """
    all_patches = []
    all_metas = []
    
    for img_idx, img in enumerate(image_list):
        C, H, W = img.shape
        short_side = min(H, W)
        long_side = max(H, W)
        is_vertical = H > W  # True if height is the long side
        
        # Compute patch starts along the long side
        starts = _compute_patch_starts(long_side, short_side)
        
        for patch_idx, start in enumerate(starts):
            if is_vertical:
                # Vertical image: patches along height
                y0, y1 = start, start + short_side
                x0, x1 = 0, short_side
            else:
                # Horizontal image: patches along width
                y0, y1 = 0, short_side
                x0, x1 = start, start + short_side
            
            # Extract patch (no padding)
            patch = img[:, y0:y1, x0:x1]  # (C, short_side, short_side)
            
            # Resize to target_size
            patch_resized = F.interpolate(
                patch.unsqueeze(0),  # (1, C, short_side, short_side)
                size=(target_size, target_size),
                mode="bicubic",
                align_corners=False
            )  # (1, C, target_size, target_size)
            
            all_patches.append(patch_resized)
            all_metas.append({
                "img_idx": img_idx,
                "patch_idx": patch_idx,
                "n_patches": len(starts),
                "y0": y0, "y1": y1,
                "x0": x0, "x1": x1,
                "orig_h": H, "orig_w": W,
                "short_side": short_side,
                "is_vertical": is_vertical,
            })
    
    # Concatenate all patches into a batch
    patches_batch = torch.cat(all_patches, dim=0)  # (N, C, target_size, target_size)
    
    return patches_batch, all_metas


# -------------------------
# Reconstructor (blend avg / hann)
# -------------------------
def _make_2d_hann(h: int, w: int, device, dtype, eps: float = 1e-6):
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    win = (wy[:, None] * wx[None, :]).clamp_min(eps)
    return win


# -------------------------
# Reconstructor for short-side based patches
# -------------------------
@torch.no_grad()
def reconstruct_from_tasktok_input(
    patches: torch.Tensor,
    metas: list,
    target_size: int = 256,
    blend: str = "hann",  # "avg" or "hann"
):
    """
    Reconstruct images from patches created by prepare_tasktok_input.
    
    Args:
        patches: (N, C, target_size, target_size) batch tensor
        metas: List of metadata from prepare_tasktok_input
        target_size: Size of input patches (must match patches shape)
        blend: "avg" or "hann" for overlap blending
        
    Returns:
        List of tensors (C, H, W) - reconstructed images in original size
    """
    N, C, ph, pw = patches.shape
    assert ph == target_size and pw == target_size
    assert len(metas) == N
    
    device, dtype = patches.device, patches.dtype
    
    # Group patches by image index
    img_indices = sorted(set(m["img_idx"] for m in metas))
    
    reconstructed = []
    patch_offset = 0
    
    for img_idx in img_indices:
        # Get all patches for this image
        img_metas = [m for m in metas if m["img_idx"] == img_idx]
        n_patches = len(img_metas)
        
        # Get original image dimensions
        orig_h = img_metas[0]["orig_h"]
        orig_w = img_metas[0]["orig_w"]
        short_side = img_metas[0]["short_side"]
        
        # Create canvas at original size
        canvas = torch.zeros((1, C, orig_h, orig_w), device=device, dtype=dtype)
        weight = torch.zeros((1, 1, orig_h, orig_w), device=device, dtype=dtype)
        
        # Create blending window at short_side x short_side
        if blend == "hann":
            wpatch = _make_2d_hann(short_side, short_side, device, dtype)[None, None]
        elif blend == "avg":
            wpatch = torch.ones((1, 1, short_side, short_side), device=device, dtype=dtype)
        else:
            raise ValueError("blend must be 'avg' or 'hann'")
        
        for i, m in enumerate(img_metas):
            # Get the patch and resize back to original patch size
            p = patches[patch_offset + i: patch_offset + i + 1]  # (1, C, target_size, target_size)
            p_resized = F.interpolate(
                p,
                size=(short_side, short_side),
                mode="bicubic",
                align_corners=False
            )  # (1, C, short_side, short_side)
            
            y0, y1 = m["y0"], m["y1"]
            x0, x1 = m["x0"], m["x1"]
            
            canvas[:, :, y0:y1, x0:x1] += p_resized * wpatch
            weight[:, :, y0:y1, x0:x1] += wpatch
        
        # Normalize by weight
        merged = canvas / weight.clamp_min(1e-8)  # (1, C, orig_h, orig_w)
        reconstructed.append(merged.squeeze(0))  # (C, orig_h, orig_w)
        
        patch_offset += n_patches
    
    return torch.stack(reconstructed, dim=0)
