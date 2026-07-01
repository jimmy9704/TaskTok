import os, sys
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils.filter_warning

import gc
import math
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True 
import random
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from model import SwinIR
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
from argparse import ArgumentParser
from omegaconf import OmegaConf
from accelerate import Accelerator, DataLoaderConfiguration
from torchvision.utils import make_grid, save_image

from utils.common import (
    instantiate_from_config, load_network,
    calculate_psnr_pt
)
from utils.classification import calculate_accuracy
from utils.classification import prepare_environment as prepare_environment_cls
from utils.detection import (
    GroupedBatchSampler, CocoEvaluator, prepare_batch,
    create_aspect_ratio_groups, get_coco_api_from_dataset, batch_to_list,
    collate_fn, _get_iou_types, suppress_stdout
)
from utils.segmentation import convert2color, calculate_mat, compute_iou
from utils.tasktok import (
    adapt_titok_to_image_size,
    prepare_tasktok_input,
    reconstruct_from_tasktok_input,
)

from model.titok.modeling import TiTok
from model.token_predictor import TokenPredictor

def get_next_batch(iterator, loader):
    """Get next batch from iterator, recreate if exhausted."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def masked_token_l1_loss(
    z_refined: torch.Tensor,
    z_gt: torch.Tensor,
    prob_map: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """L1 loss over selected token positions only (hard mask from prob_map)."""
    selected = (prob_map > threshold).to(dtype=z_refined.dtype)  # (1, 1, H, W)
    l1 = F.l1_loss(z_refined, z_gt, reduction="none")

    # Normalize by selected token count (and by batch/channel dimensions).
    denom = (selected.sum() * z_refined.shape[0] * z_refined.shape[1]).clamp_min(1.0)
    return (l1 * selected).sum() / denom


def encode_tasktok_image(
    tasktok: TiTok,
    image: torch.Tensor,
    target_size: int,
    resize_input: bool,
) -> torch.Tensor:
    expected_size = (target_size, target_size)
    if tuple(image.shape[-2:]) != expected_size:
        if not resize_input:
            image = F.interpolate(image, size=expected_size, mode="bicubic", align_corners=False)
        else:
            raise ValueError(
                f"TiTok input must be {expected_size}, got {tuple(image.shape[-2:])}. "
                "Set model.titok.resize_input to false or remove it to use legacy TiTok resizing."
            )
    return tasktok.encoder(pixel_values=image, latent_tokens=tasktok.latent_tokens)


def decode_tasktok_tokens(
    tasktok: TiTok,
    tokens: torch.Tensor,
    output_size: tuple[int, int],
    resize_input: bool,
) -> torch.Tensor:
    output = tasktok.decoder(tokens)
    if tuple(output.shape[-2:]) != tuple(output_size):
        if not resize_input:
            output = F.interpolate(output, size=output_size, mode="bicubic", align_corners=False)
        else:
            raise ValueError(
                f"TiTok output must be {tuple(output_size)}, got {tuple(output.shape[-2:])}. "
                "Use a TiTok checkpoint adapted to the target input size."
            )
    return output


def worker_init_fn(worker_id):
    """Initialize worker with fixed seed for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(args) -> None:
    # setup environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cfg = OmegaConf.load(args.config)

    # Set random seeds for reproducibility
    seed = cfg.train.get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=True),
        mixed_precision=cfg.train.precision
    )
    device = accelerator.device
    dirs, Logging = prepare_environment_cls(__name__, cfg, args, accelerator)
    exp_dir, ckpt_dir, img_dir = dirs["exp"], dirs["ckpt"], dirs["img"]

    # ========== Create and load models ==========
    # SwinIR
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    if cfg.model.pre_restoration:
        swinir.load_state_dict(torch.load(cfg.train.resume_swinir, map_location="cpu"), strict=True)
        Logging(f"Load SwinIR weight from checkpoint: {cfg.train.resume_swinir}")

    # TiTok 
    tasktok = TiTok.from_pretrained(cfg.model.titok.pretrained)
    resize_input = bool(cfg.model.titok.get("resize_input", False))
    tasktok_input_size = 512 if resize_input else tasktok.encoder.image_size
    if resize_input and tasktok_input_size != tasktok.encoder.image_size:
        adapt_titok_to_image_size(tasktok, tasktok_input_size)
        Logging(f"Adapt TiTok input/output size to {tasktok_input_size}")
    elif not resize_input:
        Logging(f"Use legacy TiTok resize path with input size {tasktok_input_size}")
    tasktok.to(device)
    if cfg.train.get("resume_tasktok"):
        tasktok.load_state_dict(torch.load(cfg.train.resume_tasktok, map_location="cpu"), strict=True)
        Logging(f"Load TaskTok weight from checkpoint: {cfg.train.resume_tasktok}")

    # Token Predictor
    token_predictor = TokenPredictor(
        d_token=tasktok.quantize.token_size,
        d_model=cfg.model.token_predictor.params.d_model,
        n_heads=cfg.model.token_predictor.params.n_heads,
        n_layers=cfg.model.token_predictor.params.n_layers,
        n_tokens=tasktok.num_latent_tokens,
    ).to(device)
    if cfg.train.get("resume_token_predictor"):
        token_predictor.load_state_dict(torch.load(cfg.train.resume_token_predictor, map_location="cpu"))
        Logging(f"Load TokenPredictor weight from checkpoint: {cfg.train.resume_token_predictor}")

    # Initialize token_switch from greedy search order
    if cfg.train.get("greedy_token_order"):
        order_data = torch.load(cfg.train.greedy_token_order, map_location="cpu")
        orders = order_data["orders"]  # (3, n_tokens)
        if isinstance(orders, torch.Tensor):
            orders = orders.tolist()
        Logging(f"Initialized token_switch from: {cfg.train.greedy_token_order}")
    else:
        # Use random order if order_data is None
        n_tokens = token_predictor.n_tokens
        orders = [random.sample(range(n_tokens), n_tokens) for _ in range(3)]
        Logging(f"order_data is None, using random order for token_switch initialization")
    
    for task_id in range(3):
        token_predictor.init_token_switch(
            order=orders[task_id] if isinstance(orders[task_id], list) else orders[task_id].tolist(),
            p_min=0.0,
            p_max=1.0,
            noise_std=0.0,
            task_id=task_id,
        )

    # Classification models
    teacher_clsnet = instantiate_from_config(cfg.model.teacher_clsnet)
    teacher_clsnet = load_network(teacher_clsnet, cfg.train.resume_teacher_clsnet, strict=cfg.train.strict_load)
    for p in teacher_clsnet.parameters():
        p.requires_grad = False
    Logging(f"Load Teacher ClassificationNetwork weight from checkpoint: {cfg.train.resume_teacher_clsnet}")

    clsnet = instantiate_from_config(cfg.model.clsnet)
    if cfg.train.get("resume_clsnet"):
        clsnet = load_network(clsnet, cfg.train.resume_clsnet, strict=cfg.train.strict_load)
        Logging(f"Load ClassificationNetwork weight from checkpoint: {cfg.train.resume_clsnet}")

    # Detection models
    teacher_detnet = instantiate_from_config(cfg.model.teacher_detnet)
    teacher_detnet = load_network(teacher_detnet, cfg.train.resume_teacher_detnet, strict=cfg.train.strict_load)
    for p in teacher_detnet.parameters():
        p.requires_grad = False
    Logging(f"Load Teacher DetectionNetwork weight from checkpoint: {cfg.train.resume_teacher_detnet}")

    detnet = instantiate_from_config(cfg.model.detnet)
    if cfg.train.get("resume_detnet"):
        detnet = load_network(detnet, cfg.train.resume_detnet, strict=cfg.train.strict_load)
        Logging(f"Load DetectionNetwork weight from checkpoint: {cfg.train.resume_detnet}")

    # Segmentation models
    teacher_segnet = instantiate_from_config(cfg.model.teacher_segnet)
    teacher_segnet = load_network(teacher_segnet, cfg.train.resume_teacher_segnet, strict=cfg.train.strict_load)
    for p in teacher_segnet.parameters():
        p.requires_grad = False
    Logging(f"Load Teacher SegmentationNetwork weight from checkpoint: {cfg.train.resume_teacher_segnet}")

    segnet = instantiate_from_config(cfg.model.segnet)
    if cfg.train.get("resume_segnet"):
        segnet = load_network(segnet, cfg.train.resume_segnet, strict=cfg.train.strict_load)
        Logging(f"Load SegmentationNetwork weight from checkpoint: {cfg.train.resume_segnet}")

    # ========== Setup trainable parameters ==========
    tasktok.latent_tokens.requires_grad_(False)
    tasktok.encoder.eval()
    tasktok.encoder.requires_grad_(False)
    tasktok.decoder.train()
    tasktok.decoder.requires_grad_(True)

    tasktok_params = [p for p in tasktok.decoder.parameters() if p.requires_grad]
    token_switch_param_ids = {id(p) for p in token_predictor.token_switch_tasks.parameters()}
    token_predictor_params = [p for p in token_predictor.parameters() if p.requires_grad and id(p) not in token_switch_param_ids]

    # ========== Setup optimizers ==========
    lr_token_switch = cfg.train.get("learning_rate_token_switch", cfg.train.learning_rate_tasktok)
    opt_tasktok = torch.optim.AdamW([
        {"params": tasktok_params, "lr": cfg.train.learning_rate_tasktok},
        {"params": token_predictor_params, "lr": cfg.train.get("learning_rate_token_predictor", cfg.train.learning_rate_tasktok)},
        {"params": [token_predictor.token_switch_tasks[0]], "lr": lr_token_switch},  # cls
        {"params": [token_predictor.token_switch_tasks[1]], "lr": lr_token_switch},  # det
        {"params": [token_predictor.token_switch_tasks[2]], "lr": lr_token_switch},  # seg
    ])

    opt_clsnet = torch.optim.SGD(
        clsnet.parameters(), lr=cfg.train.learning_rate_clsnet,
        momentum=0.9, weight_decay=1e-4
    )

    opt_detnet = torch.optim.SGD(
        detnet.parameters(), lr=cfg.train.learning_rate_detnet,
        momentum=0.9, weight_decay=1e-4
    )

    opt_segnet = torch.optim.SGD(
        segnet.parameters(), lr=cfg.train.learning_rate_segnet,
        momentum=0.9, weight_decay=1e-4
    )

    eta_min_value = 1e-7

    ratio = cfg.train.task_sampling_ratio
    cls_max_steps = cfg.train.train_steps * ratio[0] // sum(ratio)
    det_max_steps = cfg.train.train_steps * ratio[1] // sum(ratio)
    seg_max_steps = cfg.train.train_steps * ratio[2] // sum(ratio)

    # Step counters for LambdaLR (mutable containers for closure capture)
    global_step_counter = [0]  # For param_groups 0-1 (tasktok, token_predictor)
    token_switch_step_counts = [0, 0, 0]  # For param_groups 2-4 (token_switch per task)

    # T_max and eta_min per param_group
    # param_groups: 0=tasktok, 1=token_predictor, 2=token_switch_cls, 3=token_switch_det, 4=token_switch_seg
    T_max_list = [
        cfg.train.train_steps,  # tasktok
        cfg.train.train_steps,  # token_predictor
        int(0.05 * cls_max_steps),  # token_switch_cls
        int(0.05 * det_max_steps),  # token_switch_det
        int(0.05 * seg_max_steps),  # token_switch_seg
    ]

    base_lrs = [g["lr"] for g in opt_tasktok.param_groups]

    def cosine_mult(step: int, T_max: int, base_lr: float, eta_min: float) -> float:
        """Compute cosine annealing multiplier for LambdaLR."""
        if T_max <= 0:
            return eta_min / base_lr if base_lr > 0 else 1.0
        t = min(step, T_max)
        lr = eta_min + (base_lr - eta_min) * 0.5 * (1.0 + math.cos(math.pi * t / T_max))
        return lr / base_lr if base_lr > 0 else 1.0

    # Create lr_lambda functions for each param group
    # Groups 0-1: use global_step_counter, Groups 2-4: use per-task token_switch_step_counts
    lr_lambdas = []
    for i in range(5):
        if i < 2:
            lr_lambdas.append(
                lambda _step, idx=i: cosine_mult(global_step_counter[0], T_max_list[idx], base_lrs[idx], eta_min_value)
            )
        else:
            task_idx = i - 2
            lr_lambdas.append(
                lambda _step, idx=i, tidx=task_idx: cosine_mult(
                    token_switch_step_counts[tidx], T_max_list[idx], base_lrs[idx], eta_min_value
                )
            )

    sch_tasktok = torch.optim.lr_scheduler.LambdaLR(opt_tasktok, lr_lambda=lr_lambdas)
    sch_clsnet = torch.optim.lr_scheduler.CosineAnnealingLR(opt_clsnet, T_max=cls_max_steps, eta_min=eta_min_value)
    sch_detnet = torch.optim.lr_scheduler.CosineAnnealingLR(opt_detnet, T_max=det_max_steps, eta_min=eta_min_value)
    sch_segnet = torch.optim.lr_scheduler.CosineAnnealingLR(opt_segnet, T_max=seg_max_steps, eta_min=eta_min_value)

    # ========== Setup datasets ==========
    # Set generator seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # Classification dataset
    cls_dataset = instantiate_from_config(cfg.dataset.cls.train)
    cls_loader = DataLoader(
        dataset=cls_dataset, batch_size=cfg.train.batch_size_cls,
        num_workers=cfg.train.num_workers, shuffle=True, drop_last=True,
        worker_init_fn=worker_init_fn, generator=generator
    )
    Logging(f"Classification training dataset contains {len(cls_dataset):,} images")

    cls_val_dataset = instantiate_from_config(cfg.dataset.cls.val)
    cls_val_loader = DataLoader(
        dataset=cls_val_dataset, batch_size=cfg.val.batch_size_cls,
        num_workers=cfg.val.num_workers, shuffle=False, drop_last=False,
        worker_init_fn=worker_init_fn
    )
    Logging(f"Classification validation dataset contains {len(cls_val_dataset):,} images")

    # Detection dataset
    with suppress_stdout():
        det_dataset = instantiate_from_config(cfg.dataset.det.train)
    train_sampler = torch.utils.data.RandomSampler(det_dataset, generator=generator)
    group_ids = create_aspect_ratio_groups(det_dataset, k=cfg.train.aspect_ratio_group_factor, Logging=Logging)
    batch_sampler = GroupedBatchSampler(train_sampler, group_ids, cfg.train.batch_size_det)
    det_loader = DataLoader(
        dataset=det_dataset, batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers, pin_memory=True, collate_fn=collate_fn,
        worker_init_fn=worker_init_fn
    )
    batch_transform = None
    if cfg.dataset.get("batch_transform"):
        batch_transform = instantiate_from_config(cfg.dataset.batch_transform)
    Logging(f"Detection training dataset contains {len(det_dataset):,} images")

    with suppress_stdout():
        det_val_dataset = instantiate_from_config(cfg.dataset.det.val)
    det_val_loader = DataLoader(
        dataset=det_val_dataset, batch_size=cfg.val.batch_size_det,
        num_workers=cfg.val.num_workers, shuffle=False, pin_memory=True, collate_fn=collate_fn,
        worker_init_fn=worker_init_fn
    )
    Logging(f"Detection validation dataset contains {len(det_val_dataset):,} images")

    # Segmentation dataset
    seg_dataset = instantiate_from_config(cfg.dataset.seg.train)
    seg_loader = DataLoader(
        dataset=seg_dataset, batch_size=cfg.train.batch_size_seg,
        num_workers=cfg.train.num_workers, shuffle=True, drop_last=True,
        worker_init_fn=worker_init_fn, generator=generator
    )
    Logging(f"Segmentation training dataset contains {len(seg_dataset):,} images")

    seg_val_dataset = instantiate_from_config(cfg.dataset.seg.val)
    seg_val_loader = DataLoader(
        dataset=seg_val_dataset, batch_size=cfg.val.batch_size_seg,
        num_workers=cfg.val.num_workers, shuffle=False, drop_last=False,
        worker_init_fn=worker_init_fn
    )
    Logging(f"Segmentation validation dataset contains {len(seg_val_dataset):,} images")

    # ========== Prepare models ==========
    swinir.eval().to(device)
    tasktok.train().to(device)
    teacher_clsnet.eval().to(device)
    clsnet.train().to(device)
    teacher_detnet.eval().to(device)
    detnet.train().to(device)
    teacher_segnet.eval().to(device)
    segnet.train().to(device)

    # Prepare models, optimizers, schedulers, and dataloaders
    (swinir, tasktok, token_predictor, teacher_clsnet, clsnet, teacher_detnet, detnet,
     teacher_segnet, segnet,
     opt_tasktok, opt_clsnet, opt_detnet, opt_segnet,
     sch_tasktok, sch_clsnet, sch_detnet, sch_segnet,
     cls_loader, cls_val_loader, det_loader, det_val_loader, seg_loader, seg_val_loader) = accelerator.prepare(
        swinir, tasktok, token_predictor, teacher_clsnet, clsnet, teacher_detnet, detnet,
        teacher_segnet, segnet,
        opt_tasktok, opt_clsnet, opt_detnet, opt_segnet,
        sch_tasktok, sch_clsnet, sch_detnet, sch_segnet,
        cls_loader, cls_val_loader, det_loader, det_val_loader, seg_loader, seg_val_loader
    )

    pure_detnet = accelerator.unwrap_model(detnet)
    frozen_det_parameters = [n for n, p in detnet.named_parameters() if not p.requires_grad]

    # Setup COCO evaluator
    Logging("Preparing COCO API for detection evaluation...")
    with suppress_stdout():
        coco = get_coco_api_from_dataset(det_val_dataset)
    iou_types = _get_iou_types(pure_detnet)

    # ========== Training setup ==========
    global_step = 0
    max_steps = cfg.train.train_steps
    epoch = 0

    loss_records = dict(
        ce_cls=[], ce_seg=[], det=[],
        token_selected_cls=[], token_selected_det=[], token_selected_seg=[],
    )
    loss_avg_records = {k: 0.0 for k in loss_records.keys()}

    weight_tok = cfg.train.get("weight_tok", 0.0)
    weight_mask = cfg.train.get("weight_mask", 0.0)
    best_val_acc1 = float("-inf")
    best_val_mAP = float("-inf")
    best_val_miou = float("-inf")

    writer = None
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)

    Logging(f"Loss weights -> tok: {weight_tok}, mask: {weight_mask}")

    # Initialize iterators
    cls_iter = iter(cls_loader)
    det_iter = iter(det_loader)
    seg_iter = iter(seg_loader)

    # Store latest images for each task (for logging)
    latest_cls_images = None  # (gt, lq, pre_res, res)
    latest_det_images = None  # (gt, lq, pre_res, res)
    latest_seg_images = None  # (gt, lq, pre_res, res, mask, pred)

    accelerator.wait_for_everyone()

    # ========== Training loop ==========
    Logging(f"Training for {max_steps} steps with joint classification, detection, and segmentation...")
    pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="step", total=max_steps)

    while global_step < max_steps:
        # Randomly sample task_id: 0 = classification, 1 = detection
        task_id = random.choices(
            range(len(cfg.train.task_sampling_ratio)),
            weights=cfg.train.task_sampling_ratio,
            k=1
        )[0]


        if task_id == 0:
            # ==================== CLASSIFICATION ====================
            cls_batch, cls_iter = get_next_batch(cls_iter, cls_loader)
            gt, lq, label, _ = cls_batch
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            bs = gt.size(0)

            # Train tasktok with classification
            with accelerator.autocast():
                tasktok.train(), token_predictor.train(), clsnet.eval()
                for p in clsnet.parameters():
                    p.requires_grad = False

                # pre-restoration
                pre_res = lq
                if cfg.model.pre_restoration:
                    with torch.no_grad():
                        pre_res = swinir(lq)

                # task tokenization
                with torch.no_grad(): # gt token
                    z_gt = encode_tasktok_image(tasktok, gt, tasktok_input_size, resize_input)
                    
                z = encode_tasktok_image(tasktok, pre_res, tasktok_input_size, resize_input)    
                z_pred, z_refined, token_selected_sum, prob_map, token_logit = token_predictor(z, task_id=task_id)
                res = decode_tasktok_tokens(tasktok, z_pred, gt.shape[-2:], resize_input)

                loss_tok = masked_token_l1_loss(z_refined, z_gt, prob_map)
                loss_mask = prob_map.mean()

                # HLF loss for classification
                _, feat_gt = clsnet(gt, return_feat=True)
                _, feat_res = clsnet(res, return_feat=True)
                with torch.no_grad():
                    _, teacher_feat_gt = teacher_clsnet(gt, return_feat=True)
                _, teacher_feat_res = teacher_clsnet(res, return_feat=True)
                loss_hlf = (F.l1_loss(teacher_feat_res, teacher_feat_gt, reduction="mean") +
                            F.l1_loss(feat_res, feat_gt, reduction="mean")) * cfg.train.weight_hlf

            opt_tasktok.zero_grad()
            total_tasktok_loss = loss_hlf + loss_tok * weight_tok + loss_mask * weight_mask
            accelerator.backward(total_tasktok_loss)
            opt_tasktok.step()
            # Update step counters for LambdaLR scheduler
            global_step_counter[0] = global_step + 1
            token_switch_step_counts[task_id] += 1
            sch_tasktok.step()

            # Train clsnet
            with accelerator.autocast():
                tasktok.eval(), token_predictor.eval(), clsnet.train()
                for p in clsnet.parameters():
                    p.requires_grad = True

                pred, feat_student = clsnet(torch.cat((res.detach(), gt[:bs//2]), dim=0), return_feat=True)
                label_cat = torch.cat([label, label[:bs//2]], dim=0)
                loss_ce = F.cross_entropy(pred, label_cat, reduction="mean") * cfg.train.weight_ce

                with torch.no_grad():
                    _, feat_teacher = teacher_clsnet(torch.cat((gt, gt[:bs//2]), dim=0), return_feat=True)
                loss_fm = F.l1_loss(feat_student, feat_teacher, reduction="mean") * cfg.train.weight_fm

            opt_clsnet.zero_grad()
            clsnet_loss = loss_ce + loss_fm
            accelerator.backward(clsnet_loss)
            opt_clsnet.step()
            del clsnet_loss
            if opt_clsnet.param_groups[0]['lr'] > eta_min_value:
                sch_clsnet.step()

            # Record losses
            loss_records["ce_cls"].append(loss_ce.item())
            loss_records["token_selected_cls"].append((token_selected_sum).item())
            
            prob_mean = prob_map.mean()

            # Store latest images for logging (only keep CPU copies to save GPU memory)
            # Do this before cleanup to avoid UnboundLocalError
            if (global_step + 1) % cfg.train.image_every == 0 or global_step == 0:
                latest_cls_images = (gt.detach().cpu(), lq.detach().cpu(), 
                                    pre_res.detach().cpu() if cfg.model.pre_restoration else None,
                                    res.detach().cpu())
            else:
                latest_cls_images = None

            # Clean up intermediate tensors to prevent memory leak
            del z, z_gt
            del z_pred, z_refined, token_logit
            del pre_res
            del feat_gt, feat_res, teacher_feat_gt, teacher_feat_res, feat_student, feat_teacher
            del pred, label_cat

            task_str = "CLS"

        elif task_id == 1:
            # ==================== DETECTION ====================
            det_batch, det_iter = get_next_batch(det_iter, det_loader)
            gt_list, lq_list, gt_batch, lq_batch, annot_list, _, bs = prepare_batch(det_batch, device, batch_transform)

            # Train tasktok with detection
            with accelerator.autocast():
                tasktok.train(), token_predictor.train(), detnet.eval()
                for p in detnet.parameters():
                    p.requires_grad = False

                # pre-restoration
                pre_res_batch = lq_batch
                if cfg.model.pre_restoration:
                    with torch.no_grad():
                        pre_res_batch = swinir(lq_batch)

                # task tokenization
                with torch.no_grad(): # gt token
                    z_gt = encode_tasktok_image(tasktok, gt_batch, tasktok_input_size, resize_input)

                z = encode_tasktok_image(tasktok, pre_res_batch, tasktok_input_size, resize_input)
                z_pred, z_refined, token_selected_sum, prob_map, token_logit = token_predictor(z, task_id=task_id)
                res_batch = decode_tasktok_tokens(tasktok, z_pred, gt_batch.shape[-2:], resize_input)
                res_list = batch_to_list(res_batch, gt_list)
                
                loss_tok = masked_token_l1_loss(z_refined, z_gt, prob_map)
                loss_mask = prob_map.mean()

                # HLF loss for detection
                _, _, feat_gt = detnet(gt_list, return_feat=True)
                _, _, feat_res = detnet(res_list, return_feat=True)
                with torch.no_grad():
                    _, _, teacher_feat_gt = teacher_detnet(gt_list, return_feat=True)
                _, _, teacher_feat_res = teacher_detnet(res_list, return_feat=True)
                k1, k2 = [k for k in feat_gt['features']][-3:-1]
                loss_hlf = (F.l1_loss(feat_res['features'][k1], feat_gt['features'][k1], reduction="mean") * 0.5 +
                            F.l1_loss(feat_res['features'][k2], feat_gt['features'][k2], reduction="mean") * 0.5 +
                            F.l1_loss(teacher_feat_res['features'][k1], teacher_feat_gt['features'][k1], reduction="mean") * 0.5 +
                            F.l1_loss(teacher_feat_res['features'][k2], teacher_feat_gt['features'][k2], reduction="mean") * 0.5) * cfg.train.weight_hlf

            opt_tasktok.zero_grad()
            total_tasktok_loss = loss_hlf + loss_tok * weight_tok + loss_mask * weight_mask
            accelerator.backward(total_tasktok_loss)
            opt_tasktok.step()
            # Clear loss computation graph
            del total_tasktok_loss
            # Update step counters for LambdaLR scheduler
            global_step_counter[0] = global_step + 1
            token_switch_step_counts[task_id] += 1
            sch_tasktok.step()

            # Train detnet
            with accelerator.autocast():
                tasktok.eval(), token_predictor.eval(), detnet.train()
                for n, p in detnet.named_parameters():
                    if n not in frozen_det_parameters:
                        p.requires_grad = True

                res_list_detached = [r.detach() for r in res_list]
                _, loss_dict, feat_student = detnet(res_list_detached + gt_list[:bs//2], annot_list + annot_list[:bs//2], return_feat=True)
                loss_det = sum(loss_dict.values()) * cfg.train.weight_det

                with torch.no_grad():
                    _, _, feat_teacher = teacher_detnet(gt_list + gt_list[:bs//2], annot_list + annot_list[:bs//2], return_feat=True)
                loss_fm = (F.l1_loss(feat_student['features']['0'], feat_teacher['features']['0'], reduction="mean") * 0.5 +
                           F.l1_loss(feat_student['features']['1'], feat_teacher['features']['1'], reduction="mean") * 0.5) * cfg.train.weight_fm

            opt_detnet.zero_grad()
            accelerator.backward(loss_det + loss_fm)
            opt_detnet.step()
            if opt_detnet.param_groups[0]['lr'] > eta_min_value:
                sch_detnet.step()

            # Record losses
            loss_records["det"].append(loss_det.item())
            loss_records["token_selected_det"].append((token_selected_sum).item())
            
            prob_mean = prob_map.mean()

            # Store latest images for logging (only keep CPU copies to save GPU memory)
            # Do this before cleanup to avoid UnboundLocalError
            if (global_step + 1) % cfg.train.image_every == 0 or global_step == 0:
                latest_det_images = (gt_batch.detach().cpu(), lq_batch.detach().cpu(), 
                                    pre_res_batch.detach().cpu() if cfg.model.pre_restoration else None,
                                    res_batch.detach().cpu())
            else:
                latest_det_images = None

            # Clean up intermediate tensors to prevent memory leak
            del z, z_gt
            del z_pred, z_refined, token_logit
            del pre_res_batch
            del feat_gt, feat_res, teacher_feat_gt, teacher_feat_res, feat_student, feat_teacher
            del res_list, res_list_detached, loss_dict
            task_str = "DET"

        elif task_id == 2:
            # ==================== SEGMENTATION ====================
            seg_batch, seg_iter = get_next_batch(seg_iter, seg_loader)
            gt, lq, mask, _ = seg_batch
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            mask = mask.long().to(device)
            bs = gt.size(0)

            # Train tasktok with segmentation
            with accelerator.autocast():
                tasktok.train(), token_predictor.train(), segnet.eval()
                for p in segnet.parameters():
                    p.requires_grad = False

                # pre-restoration
                pre_res = lq
                if cfg.model.pre_restoration:
                    with torch.no_grad():
                        pre_res = swinir(lq)

                # task tokenization
                with torch.no_grad(): # gt token
                    z_gt = encode_tasktok_image(tasktok, gt, tasktok_input_size, resize_input)

                z = encode_tasktok_image(tasktok, pre_res, tasktok_input_size, resize_input)
                z_pred, z_refined, token_selected_sum, prob_map, token_logit = token_predictor(z, task_id=task_id)
                res = decode_tasktok_tokens(tasktok, z_pred, gt.shape[-2:], resize_input)

                loss_tok = masked_token_l1_loss(z_refined, z_gt, prob_map)
                loss_mask = prob_map.mean()

                # HLF loss for segmentation
                _, feat_gt = segnet(gt, return_feat=True)
                _, feat_res = segnet(res, return_feat=True)
                with torch.no_grad():
                    _, teacher_feat_gt = teacher_segnet(gt, return_feat=True)
                _, teacher_feat_res = teacher_segnet(res, return_feat=True)
                loss_hlf = (F.l1_loss(teacher_feat_res["C5"], teacher_feat_gt["C5"], reduction="mean") +
                            F.l1_loss(feat_res["C5"], feat_gt["C5"], reduction="mean")) * cfg.train.weight_hlf

            opt_tasktok.zero_grad()
            total_tasktok_loss = loss_hlf + loss_tok * weight_tok + loss_mask * weight_mask
            accelerator.backward(total_tasktok_loss)
            opt_tasktok.step()
            # Clear loss computation graph
            del total_tasktok_loss
            # Update step counters for LambdaLR scheduler
            global_step_counter[0] = global_step + 1
            token_switch_step_counts[task_id] += 1
            sch_tasktok.step()

            # Train segnet
            with accelerator.autocast():
                tasktok.eval(), token_predictor.eval(), segnet.train()
                for p in segnet.parameters():
                    p.requires_grad = True

                pred, feat_student = segnet(torch.cat((res.detach(), gt[:bs//2]), dim=0), return_feat=True)
                loss_ce = F.cross_entropy(pred["out"], torch.cat((mask, mask[:bs//2]), dim=0), ignore_index=255) * cfg.train.weight_ce

                with torch.no_grad():
                    _, feat_teacher = teacher_segnet(torch.cat((gt, gt[:bs//2]), dim=0), return_feat=True)
                loss_fm = F.l1_loss(feat_student["C5"], feat_teacher["C5"], reduction="mean") * cfg.train.weight_fm

            opt_segnet.zero_grad()
            segnet_loss = loss_ce + loss_fm
            accelerator.backward(segnet_loss)
            opt_segnet.step()
            del segnet_loss
            if opt_segnet.param_groups[0]['lr'] > eta_min_value:
                sch_segnet.step()

            # Record losses
            loss_records["ce_seg"].append(loss_ce.item())
            loss_records["token_selected_seg"].append((token_selected_sum).item())
            
            prob_mean = prob_map.mean()

            # Store latest images for logging (only keep CPU copies to save GPU memory)
            pred_out_cpu = None
            if (global_step + 1) % cfg.train.image_every == 0 or global_step == 0:
                pred_out_cpu = pred["out"][:bs].detach().cpu()
                latest_seg_images = (gt.detach().cpu(), lq.detach().cpu(), 
                                    pre_res.detach().cpu() if cfg.model.pre_restoration else None,
                                    res.detach().cpu(), mask.detach().cpu(), pred_out_cpu)
            else:
                latest_seg_images = None
            
            # Clean up intermediate tensors to prevent memory leak
            del z, z_gt
            del z_pred, z_refined, token_logit
            del pre_res
            del feat_gt, feat_res, teacher_feat_gt, teacher_feat_res, feat_student, feat_teacher
            del pred, pred_out_cpu
            task_str = "SEG"

        accelerator.wait_for_everyone()
        global_step += 1
        pbar.update(1)
        pbar.set_description(f"Step: {global_step:07d}, Task: {task_str}, Token: {token_selected_sum.item():.1f}")

        # Log training loss
        if global_step % cfg.train.log_every == 0 or args.debug:
            for key in loss_records.keys():
                if len(loss_records[key]) > 0:
                    # For token_selected, use last value; for others, use average
                    if key.startswith("token_selected_"):
                        loss_avg_records[key] = loss_records[key][-1]
                    else:
                        loss_tensor = torch.tensor(loss_records[key], device=device)
                        loss_avg_records[key] = accelerator.gather(
                            loss_tensor.unsqueeze(0)
                        ).mean().item()
                        del loss_tensor
                    loss_records[key].clear()
            torch.cuda.empty_cache()

            if accelerator.is_local_main_process:
                loss_summary = f"[{global_step:05d}/{max_steps:05d}] Training loss: ( " + \
                    ", ".join([f"{key}: {value:.4f}" for key, value in loss_avg_records.items() if value > 0]) + " )"
                Logging(loss_summary, print=False)
                for key in ["ce_cls", "ce_seg", "det"]:
                    value = loss_avg_records.get(key, 0.0)
                    if value > 0:
                        writer.add_scalar(f"loss/{key}", value, global_step)

                # Token selected count per sample (last value in the logging window)
                for task_name in ["cls", "det", "seg"]:
                    key = f"token_selected_{task_name}"
                    if loss_avg_records.get(key, 0) > 0:
                        writer.add_scalar(f"train/token_selected_{task_name}", loss_avg_records[key], global_step)

        # Save checkpoint
        if global_step % cfg.train.ckpt_every == 0 or args.debug:
            if accelerator.is_local_main_process:
                torch.save(tasktok.state_dict(), f"{ckpt_dir}/tasktok_{global_step:07d}.pt")
                torch.save(token_predictor.state_dict(), f"{ckpt_dir}/token_predictor_{global_step:07d}.pt")
                torch.save(clsnet.state_dict(), f"{ckpt_dir}/clsnet_{global_step:07d}.pt")
                torch.save(detnet.state_dict(), f"{ckpt_dir}/detnet_{global_step:07d}.pt")
                torch.save(segnet.state_dict(), f"{ckpt_dir}/segnet_{global_step:07d}.pt")

        # save images (both cls and det from latest iterations)
        if global_step % cfg.train.image_every == 0 or global_step == 1 or (args.debug):
            with torch.no_grad():
                N = 4
                if accelerator.is_local_main_process:
                    # Save classification images (if available)
                    if latest_cls_images is not None:
                        cls_gt, cls_lq, cls_pre_res, cls_res = latest_cls_images
                        # Move to GPU temporarily for processing
                        cls_gt = cls_gt.to(device)
                        cls_lq = cls_lq.to(device)
                        cls_pre_res = cls_pre_res.to(device) if cls_pre_res is not None else None
                        cls_res = cls_res.to(device)
                        for name, image in [
                            ("cls_gt", cls_gt[:N]), ("cls_lq", cls_lq[:N]),
                            ("cls_pre_restored", cls_pre_res[:N] if cls_pre_res is not None else cls_lq[:N]), 
                            ("cls_restored", cls_res[:N]),
                        ]:
                            grid_image = make_grid(image, nrow=4)
                            # writer.add_image(f"image/{name}", grid_image, global_step)
                            img_name = '{:06d}_{}.png'.format(global_step, name)
                            save_image(grid_image, os.path.join(img_dir, img_name))
                        del cls_gt, cls_lq, cls_pre_res, cls_res
                    
                    # Save detection images (if available)
                    if latest_det_images is not None:
                        det_gt, det_lq, det_pre_res, det_res = latest_det_images
                        # Move to GPU temporarily for processing
                        det_gt = det_gt.to(device)
                        det_lq = det_lq.to(device)
                        det_pre_res = det_pre_res.to(device) if det_pre_res is not None else None
                        det_res = det_res.to(device)
                        for name, image in [
                            ("det_gt", det_gt[:N]), ("det_lq", det_lq[:N]),
                            ("det_pre_restored", det_pre_res[:N] if det_pre_res is not None else det_lq[:N]), 
                            ("det_restored", det_res[:N]),
                        ]:
                            grid_image = make_grid(image, nrow=4)
                            # writer.add_image(f"image/{name}", grid_image, global_step)
                            img_name = '{:06d}_{}.png'.format(global_step, name)
                            save_image(grid_image, os.path.join(img_dir, img_name))
                        del det_gt, det_lq, det_pre_res, det_res
                    
                    # Save segmentation images (if available)
                    if latest_seg_images is not None:
                        seg_gt, seg_lq, seg_pre_res, seg_res, seg_mask, seg_pred = latest_seg_images
                        # Move to GPU temporarily for processing
                        seg_gt = seg_gt.to(device)
                        seg_lq = seg_lq.to(device)
                        seg_pre_res = seg_pre_res.to(device) if seg_pre_res is not None else None
                        seg_res = seg_res.to(device)
                        seg_mask = seg_mask.to(device)
                        seg_pred = seg_pred.to(device) if seg_pred is not None else None
                        for name, image in [
                            ("seg_gt", seg_gt[:N]), ("seg_lq", seg_lq[:N]),
                            ("seg_pre_restored", seg_pre_res[:N] if seg_pre_res is not None else seg_lq[:N]), 
                            ("seg_restored", seg_res[:N]),
                        ]:
                            grid_image = make_grid(image, nrow=4)
                            # writer.add_image(f"image/{name}", grid_image, global_step)
                            img_name = '{:06d}_{}.png'.format(global_step, name)
                            save_image(grid_image, os.path.join(img_dir, img_name))
                        # Save mask and prediction as colored images
                        seg_mask_color = convert2color(seg_mask[:N])  # (N, 3, H, W) - already normalized to [0, 1]
                        seg_pred_color = convert2color(seg_pred[:N].argmax(dim=1)) if seg_pred is not None else None  # (N, 3, H, W) - already normalized to [0, 1]
                        seg_mask_color = seg_mask_color.float()  # Convert from float64 to float32
                        if seg_pred_color is not None:
                            seg_pred_color = seg_pred_color.float()  # Convert from float64 to float32
                        for name, image in [("seg_mask", seg_mask_color), ("seg_pred", seg_pred_color)]:
                            if image is not None:
                                grid_image = make_grid(image, nrow=4)
                                # writer.add_image(f"image/{name}", grid_image, global_step)
                                img_name = '{:06d}_{}.png'.format(global_step, name)
                                save_image(grid_image, os.path.join(img_dir, img_name))
                        del seg_gt, seg_lq, seg_pre_res, seg_res, seg_mask, seg_pred, seg_mask_color, seg_pred_color
                    
                # Clear saved images to free GPU memory
                latest_cls_images = None
                latest_det_images = None
                latest_seg_images = None
                torch.cuda.empty_cache()
                gc.collect()

        # Validation
        if global_step % cfg.val.val_every == 0 or args.debug:
            tasktok.eval(), token_predictor.eval(), clsnet.eval(), detnet.eval(), segnet.eval()

            # Classification validation
            val_acc1 = []
            val_cls_psnr = []
            val_token_selected_cls = []
            cls_val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process,
                                unit="batch", total=len(cls_val_loader), leave=False, desc="CLS Validation")
            for val_gt, val_lq, val_label, _ in cls_val_loader:
                val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
                val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)

                with torch.no_grad():
                    val_pre_res = val_lq
                    if cfg.model.pre_restoration:
                        val_pre_res = swinir(val_lq)

                    z = encode_tasktok_image(tasktok, val_pre_res, tasktok_input_size, resize_input)
                    z_pred, _, token_selected_sum, _, _ = token_predictor(z, task_id=0)
                    val_res = decode_tasktok_tokens(tasktok, z_pred, val_lq.shape[-2:], resize_input)

                    val_pred = clsnet(val_res, normalize=True)
                    val_psnr_batch = calculate_psnr_pt(val_res, val_gt, crop_border=0)
                    val_psnr_batch = accelerator.gather_for_metrics(val_psnr_batch)
                    val_token_selected_batch = accelerator.gather_for_metrics(
                        token_selected_sum.unsqueeze(0)
                    )
                    val_pred, val_label = accelerator.gather_for_metrics((val_pred, val_label))
                    if accelerator.is_local_main_process:
                        val_acc1 += [calculate_accuracy(val_pred, val_label, topk=(1, 5))[0]] * val_gt.size(0)
                        val_cls_psnr += [v.item() for v in val_psnr_batch]
                        val_token_selected_cls += [v.item() for v in val_token_selected_batch]

                cls_val_pbar.update(1)
            cls_val_pbar.close()

            # Detection validation
            coco_evaluator = CocoEvaluator(coco, iou_types)
            val_token_selected_det = []
            det_val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process,
                                unit="batch", total=len(det_val_loader), leave=False, desc="DET Validation")
            for val_batch in det_val_loader:
                val_gt_list, val_lq_list, _, val_lq_batch, val_annot_list, _, val_bs = prepare_batch(val_batch, device)
                assert val_bs == 1

                with torch.no_grad():
                    val_pre_res_batch = val_lq_batch
                    if cfg.model.pre_restoration:
                        val_pre_res_batch = swinir(val_lq_batch)

                # task tokenization
                    z = encode_tasktok_image(tasktok, val_pre_res_batch, tasktok_input_size, resize_input)
                    z_pred, _, token_selected_sum, _, _ = token_predictor(z, task_id=1)
                    val_res_batch = decode_tasktok_tokens(tasktok, z_pred, val_lq_batch.shape[-2:], resize_input)
                    val_res_list = batch_to_list(val_res_batch, val_gt_list)

                    val_pred_list, _ = detnet(val_res_list)
                    val_pred_list = [{k: v.cpu() for k, v in t.items()} for t in val_pred_list]

                    val_token_selected_batch = accelerator.gather_for_metrics(
                        (token_selected_sum / val_bs).unsqueeze(0)
                    )
                    res = {annot["image_id"]: pred for annot, pred in zip(val_annot_list, val_pred_list)}
                    if accelerator.is_local_main_process:
                        val_token_selected_det += [v.item() for v in val_token_selected_batch]
                    coco_evaluator.update(res)
                    accelerator.wait_for_everyone()

                det_val_pbar.update(1)
            det_val_pbar.close()

            coco_evaluator.synchronize_between_processes()
            with suppress_stdout():
                coco_evaluator.accumulate()

            # Segmentation validation
            num_classes = cfg.model.segnet.params.num_classes
            confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
            val_token_selected_seg = []
            seg_val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process,
                                unit="batch", total=len(seg_val_loader), leave=False, desc="SEG Validation")
            for val_gt, val_lq, val_mask, _ in seg_val_loader:
                val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
                val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
                val_mask = val_mask.long().to(device)

                with torch.no_grad():
                    # prepare input for tasktok patch size(512x512)
                    val_lq_patch, val_patch_meta = prepare_tasktok_input(val_lq, target_size=512)

                    val_pre_res = val_lq_patch
                    if cfg.model.pre_restoration:
                        val_pre_res = swinir(val_lq_patch)

                    z = encode_tasktok_image(tasktok, val_pre_res, tasktok_input_size, resize_input)
                    z_pred, _, token_selected_sum, _, _ = token_predictor(z, task_id=2)
                    val_res_patches = decode_tasktok_tokens(tasktok, z_pred, val_lq_patch.shape[-2:], resize_input)
                    val_res = reconstruct_from_tasktok_input(val_res_patches, val_patch_meta, target_size=512)
                    val_pred = segnet(val_res)
                    val_pred = val_pred["out"].argmax(dim=1)
                    val_mat = calculate_mat(val_mask.flatten(), val_pred.flatten(), n=num_classes).unsqueeze(0)
                    val_mat = accelerator.gather_for_metrics(val_mat)
                    val_token_selected_batch = accelerator.gather_for_metrics(
                        (token_selected_sum / val_gt.size(0)).unsqueeze(0)
                    )
                    if accelerator.is_local_main_process:
                        confmat += val_mat.sum(0)
                        val_token_selected_seg += [v.item() for v in val_token_selected_batch]

                seg_val_pbar.update(1)
            seg_val_pbar.close()

            val_miou = compute_iou(confmat).mean().item() * 100

            if accelerator.is_local_main_process:
                avg_val_acc1 = torch.tensor(val_acc1).mean().item() if val_acc1 else 0
                avg_val_cls_psnr = torch.tensor(val_cls_psnr).mean().item() if val_cls_psnr else 0
                det_results = coco_evaluator.summarize(Logging=Logging)
                avg_val_miou = val_miou
                avg_val_token_selected_cls = torch.tensor(val_token_selected_cls).mean().item() if val_token_selected_cls else 0
                avg_val_token_selected_det = torch.tensor(val_token_selected_det).mean().item() if val_token_selected_det else 0
                avg_val_token_selected_seg = torch.tensor(val_token_selected_seg).mean().item() if val_token_selected_seg else 0

                for tag, val in [
                    ("val/cls_acc1", avg_val_acc1),
                    ("val/cls_psnr", avg_val_cls_psnr),
                    ("val/det_mAP@[0.5:0.95]", det_results["mAP@[0.5:0.95]"]),
                    ("val/seg_miou", avg_val_miou),
                    ("val/token_selected_cls", avg_val_token_selected_cls),
                    ("val/token_selected_det", avg_val_token_selected_det),
                    ("val/token_selected_seg", avg_val_token_selected_seg),
                ]:
                    Logging(f"{tag}: {val:.4f}")
                    writer.add_scalar(tag, val, global_step)

                if avg_val_acc1 > best_val_acc1:
                    best_val_acc1 = avg_val_acc1
                    Logging(f"New best val acc1: {avg_val_acc1:.4f} at step {global_step}")
                    torch.save(tasktok.state_dict(), f"{ckpt_dir}/tasktok_best_cls.pt")
                    torch.save(token_predictor.state_dict(), f"{ckpt_dir}/token_predictor_best_cls.pt")
                    torch.save(clsnet.state_dict(), f"{ckpt_dir}/clsnet_best.pt")

                if det_results["mAP@[0.5:0.95]"] > best_val_mAP:
                    best_val_mAP = det_results["mAP@[0.5:0.95]"]
                    Logging(f"New best val mAP: {best_val_mAP:.4f} at step {global_step}")
                    torch.save(tasktok.state_dict(), f"{ckpt_dir}/tasktok_best_det.pt")
                    torch.save(token_predictor.state_dict(), f"{ckpt_dir}/token_predictor_best_det.pt")
                    torch.save(detnet.state_dict(), f"{ckpt_dir}/detnet_best.pt")

                if avg_val_miou > best_val_miou:
                    best_val_miou = avg_val_miou
                    Logging(f"New best val mIoU: {avg_val_miou:.4f} at step {global_step}")
                    torch.save(tasktok.state_dict(), f"{ckpt_dir}/tasktok_best_seg.pt")
                    torch.save(token_predictor.state_dict(), f"{ckpt_dir}/token_predictor_best_seg.pt")
                    torch.save(segnet.state_dict(), f"{ckpt_dir}/segnet_best.pt")

            tasktok.train(), token_predictor.train(), clsnet.train(), detnet.train(), segnet.train()
            
            # Explicitly delete validation variables to free memory
            del val_acc1, val_cls_psnr, val_token_selected_cls, val_token_selected_det, val_token_selected_seg
            del coco_evaluator, confmat
            torch.cuda.empty_cache()
            gc.collect()

        accelerator.wait_for_everyone()

    pbar.close()

    # Save final models
    if accelerator.is_local_main_process:
        torch.save(tasktok.state_dict(), f"{ckpt_dir}/tasktok_last.pt")
        torch.save(token_predictor.state_dict(), f"{ckpt_dir}/token_predictor_last.pt")
        torch.save(clsnet.state_dict(), f"{ckpt_dir}/clsnet_last.pt")
        torch.save(detnet.state_dict(), f"{ckpt_dir}/detnet_last.pt")
        torch.save(segnet.state_dict(), f"{ckpt_dir}/segnet_last.pt")
        Logging("Done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    main(args)
