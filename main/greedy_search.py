import os, sys
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils.filter_warning

import torch
from tqdm import tqdm
from utils.common import (
    instantiate_from_config, load_network
    )
from utils.classification import prepare_environment as prepare_environment_cls
from utils.detection import (
    CocoEvaluator, prepare_batch, get_coco_api_from_dataset,
    collate_fn, _get_iou_types, suppress_stdout, batch_to_list
)
from utils.segmentation import calculate_mat, compute_iou
from torch.nn import functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from argparse import ArgumentParser
from omegaconf import OmegaConf
from accelerate import Accelerator, DataLoaderConfiguration

from model.titok.modeling import TiTok


# ==================== Evaluation Functions ====================

def evaluate_cls_accuracy(
    z_test: torch.Tensor,
    all_lq: torch.Tensor,
    all_labels: torch.Tensor,
    tasktok,
    clsnet,
    device,
    batch_size: int = 32,
) -> float:
    """
    Evaluate classification accuracy with given latent tokens.
    
    Args:
        z_test: [N, D, 1, num_tokens] - latent tokens to decode
        all_lq: [N, C, H, W] - LQ images 
        all_labels: [N] - ground truth labels
        tasktok: TiTok model
        clsnet: Classification network
        device: torch device
        batch_size: batch size for evaluation
    
    Returns:
        accuracy: float (0-100)
    """
    num_samples = z_test.shape[0]
    correct = 0
    total = 0
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            z_batch = z_test[start_idx:end_idx].to(device)
            lq_batch = all_lq[start_idx:end_idx].to(device)
            labels_batch = all_labels[start_idx:end_idx].to(device)
            
            # Decode
            tasktok_outputs = tasktok.decoder(z_batch)
            res = F.interpolate(tasktok_outputs, scale_factor=2.0, mode='bicubic', align_corners=False)
            
            # Classification
            pred = clsnet(res, normalize=True)
            pred_labels = pred.argmax(dim=1)
            
            correct += (pred_labels == labels_batch).sum().item()
            total += labels_batch.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_det_map(
    z_test: torch.Tensor,
    det_data: dict,
    tasktok,
    detnet,
    coco_gt,
    iou_types,
    device,
    Logging=None,
) -> float:
    """
    Evaluate detection mAP@[0.5:0.95] with given latent tokens.
    
    Args:
        z_test: [N, D, 1, num_tokens] - latent tokens to decode
        det_data: dict with 'lq_list', 'gt_list', 'annot_list'
        tasktok: TiTok model
        detnet: Detection network
        coco_gt: COCO API object
        iou_types: list of IoU types
        device: torch device
    
    Returns:
        mAP: float (0-100)
    """
    coco_evaluator = CocoEvaluator(coco_gt, iou_types)
    
    with torch.no_grad():
        for idx in range(z_test.shape[0]):
            z_batch = z_test[idx:idx+1].to(device)
            lq_batch = det_data['lq_list'][idx].to(device)
            gt_item = det_data['gt_list'][idx].to(device)
            annot = det_data['annot_list'][idx]
            
            # Decode
            tasktok_outputs = tasktok.decoder(z_batch)
            res_batch = F.interpolate(tasktok_outputs, scale_factor=2.0, mode='bicubic', align_corners=False)
            res_list = batch_to_list(res_batch, [gt_item])
            
            # Detection (use decoded images, not GT!)
            pred_list, _ = detnet(res_list)
            pred_list = [{k: v.cpu() for k, v in t.items()} for t in pred_list]
            
            res = {annot["image_id"]: pred_list[0]}
            coco_evaluator.update(res)
    
    coco_evaluator.synchronize_between_processes()
    with suppress_stdout():
        coco_evaluator.accumulate()
    
    # Summarize and get results (with optional logging)
    results = coco_evaluator.summarize(Logging=Logging)
    
    return results.get("mAP@[0.5:0.95]", 0.0) if results else 0.0


def evaluate_seg_miou(
    z_test: torch.Tensor,
    seg_data: dict,
    tasktok,
    segnet,
    num_classes: int,
    device,
    batch_size: int = 4,
) -> float:
    """
    Evaluate segmentation mIoU with given latent tokens.
    
    Args:
        z_test: [N, D, 1, num_tokens] - latent tokens to decode
        seg_data: dict with 'lq', 'masks'
        tasktok: TiTok model
        segnet: Segmentation network
        num_classes: number of classes
        device: torch device
        batch_size: batch size for evaluation
    
    Returns:
        mIoU: float (0-100)
    """
    num_samples = z_test.shape[0]
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            z_batch = z_test[start_idx:end_idx].to(device)
            lq_batch = seg_data['lq'][start_idx:end_idx].to(device)
            mask_batch = seg_data['masks'][start_idx:end_idx].to(device)
            
            # Decode
            tasktok_outputs = tasktok.decoder(z_batch)
            res = F.interpolate(tasktok_outputs, scale_factor=2.0, mode='bicubic', align_corners=False)
            
            # Segmentation
            pred = segnet(res)
            pred = pred["out"].argmax(dim=1)
            
            # Update confusion matrix
            confmat += calculate_mat(mask_batch.flatten(), pred.flatten(), n=num_classes)
    
    miou = compute_iou(confmat).mean().item() * 100
    return miou


# ==================== Greedy Search ====================

def greedy_search_single_task(
    task_name: str,
    all_z_lr: torch.Tensor,
    all_z_gt: torch.Tensor,
    evaluate_fn,
    Logging,
    accelerator,
) -> tuple:
    """
    Greedy search for a single task (GT → LR direction).
    """
    num_tokens = all_z_lr.shape[3]
    
    # Evaluate oracle (all GT tokens)
    oracle_metric = evaluate_fn(all_z_gt)
    Logging(f"[{task_name.upper()}] Oracle (all GT): {oracle_metric:.2f}")
    
    # Evaluate baseline (all LR tokens)
    baseline_metric = evaluate_fn(all_z_lr)
    Logging(f"[{task_name.upper()}] Baseline (all LR): {baseline_metric:.2f}")
    
    # Greedy search
    selected_order = []
    metrics_at_step = []
    replaced_mask = torch.zeros(num_tokens, dtype=torch.bool)
    z_current = all_z_gt.clone()
    
    for step in range(num_tokens):
        best_metric = -1
        best_token_idx = -1
        
        remaining_tokens = (~replaced_mask).nonzero(as_tuple=True)[0].tolist()
        
        step_pbar = tqdm(
            remaining_tokens,
            disable=not accelerator.is_local_main_process,
            desc=f"[{task_name.upper()}] Step {step + 1}/{num_tokens}",
            leave=False
        )
        
        for token_idx in step_pbar:
            # Create test z: replace this GT token with LR
            z_test = z_current.clone()
            z_test[:, :, :, token_idx] = all_z_lr[:, :, :, token_idx]
            
            # Evaluate
            metric = evaluate_fn(z_test)
            
            if metric > best_metric:
                best_metric = metric
                best_token_idx = token_idx
            
            step_pbar.set_postfix({"best": best_token_idx, "metric": f"{best_metric:.2f}"})
        
        step_pbar.close()
        
        # Permanently replace this token
        replaced_mask[best_token_idx] = True
        z_current[:, :, :, best_token_idx] = all_z_lr[:, :, :, best_token_idx]
        selected_order.append(best_token_idx)
        metrics_at_step.append(best_metric)
        
        Logging(f"[{task_name.upper()}] Step {step + 1}: Replaced token {best_token_idx}, metric = {best_metric:.2f}")
    
    return selected_order, metrics_at_step, oracle_metric, baseline_metric


def main(args) -> None:
    # Setup environment
    cfg = OmegaConf.load(args.config)
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=True),
        mixed_precision=cfg.test.precision
    )
    device = accelerator.device
    dirs, Logging = prepare_environment_cls(__name__, cfg, args, accelerator)
    exp_dir = dirs["exp"]
    
    # ========== Load Models ==========
    Logging("Loading models...")
    
    # TiTok
    tasktok = TiTok.from_pretrained(cfg.model.titok.pretrained)
    if cfg.test.get("resume_tasktok"):
        tasktok.load_state_dict(torch.load(cfg.test.resume_tasktok, map_location="cpu"), strict=True)
        Logging(f"Loaded TaskTok from: {cfg.test.resume_tasktok}")
    tasktok.eval().to(device)
    
    # Classification network
    clsnet = instantiate_from_config(cfg.model.clsnet)
    if cfg.test.get("resume_clsnet"):
        clsnet = load_network(clsnet, cfg.test.resume_clsnet, strict=True)
        Logging(f"Loaded ClsNet from: {cfg.test.resume_clsnet}")
    clsnet.eval().to(device)
    
    # Detection network
    detnet = instantiate_from_config(cfg.model.detnet)
    if cfg.test.get("resume_detnet"):
        detnet = load_network(detnet, cfg.test.resume_detnet, strict=True)
        Logging(f"Loaded DetNet from: {cfg.test.resume_detnet}")
    detnet.eval().to(device)
    
    # Segmentation network
    segnet = instantiate_from_config(cfg.model.segnet)
    if cfg.test.get("resume_segnet"):
        segnet = load_network(segnet, cfg.test.resume_segnet, strict=True)
        Logging(f"Loaded SegNet from: {cfg.test.resume_segnet}")
    segnet.eval().to(device)
    
    # Prepare models
    tasktok, clsnet, detnet, segnet = accelerator.prepare(tasktok, clsnet, detnet, segnet)
    
    # Setup seeds
    torch.manual_seed(args.seed)
    dataloader_gen = torch.Generator()
    dataloader_gen.manual_seed(args.seed)
    
    results = {
        "orders": [],
        "task_names": ["cls", "det", "seg"],
        "metrics": {},
        "oracle": {},
        "baseline": {},
    }
    
    # ==================== Classification ====================
    Logging("\n" + "=" * 60)
    Logging("CLASSIFICATION TASK")
    Logging("=" * 60)
    
    cls_dataset = instantiate_from_config(cfg.dataset.cls.train)
    cls_loader = DataLoader(
        dataset=cls_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers, shuffle=True, drop_last=False,
        generator=dataloader_gen,
    )
    Logging(f"CLS dataset: {len(cls_dataset)} images")
    
    # Collect tokens
    Logging("Collecting CLS tokens...")
    cls_z_lr, cls_z_gt, cls_lq, cls_labels = [], [], [], []
    n_collected = 0
    for gt, lq, label, _ in tqdm(cls_loader, desc="CLS collect", disable=not accelerator.is_local_main_process):
        gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
        lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
        
        with torch.no_grad():
            downsampled_lq = F.interpolate(lq, scale_factor=0.5, mode='bicubic', align_corners=False)
            z_lr = tasktok.encoder(pixel_values=downsampled_lq, latent_tokens=tasktok.latent_tokens)
            
            downsampled_gt = F.interpolate(gt, scale_factor=0.5, mode='bicubic', align_corners=False)
            z_gt = tasktok.encoder(pixel_values=downsampled_gt, latent_tokens=tasktok.latent_tokens)
        
        cls_z_lr.append(z_lr.cpu())
        cls_z_gt.append(z_gt.cpu())
        cls_lq.append(lq.cpu())
        cls_labels.append(label.cpu())
        
        n_collected += gt.size(0)
        if n_collected >= args.n_samples:
            break
    
    cls_z_lr = torch.cat(cls_z_lr, dim=0)[:args.n_samples]
    cls_z_gt = torch.cat(cls_z_gt, dim=0)[:args.n_samples]
    cls_lq = torch.cat(cls_lq, dim=0)[:args.n_samples]
    cls_labels = torch.cat(cls_labels, dim=0)[:args.n_samples]
    Logging(f"Collected {cls_z_lr.shape[0]} CLS samples")
    
    # Greedy search
    def cls_evaluate_fn(z_test):
        return evaluate_cls_accuracy(z_test, cls_lq, cls_labels, tasktok, clsnet, device, cfg.test.batch_size)
    
    cls_order, cls_metrics, cls_oracle, cls_baseline = greedy_search_single_task(
        "cls", cls_z_lr, cls_z_gt, cls_evaluate_fn, Logging, accelerator
    )
    results["orders"].append(cls_order)
    results["metrics"]["cls"] = cls_metrics
    results["oracle"]["cls"] = cls_oracle
    results["baseline"]["cls"] = cls_baseline
    
    # ==================== Detection ====================
    Logging("\n" + "=" * 60)
    Logging("DETECTION TASK")
    Logging("=" * 60)
    
    with suppress_stdout():
        det_dataset = instantiate_from_config(cfg.dataset.det.train)
    det_loader = DataLoader(
        dataset=det_dataset, batch_size=1,
        num_workers=cfg.test.num_workers, shuffle=True, pin_memory=True, collate_fn=collate_fn,
        generator=dataloader_gen,
    )
    Logging(f"DET dataset: {len(det_dataset)} images")
    
    # Get COCO API
    with suppress_stdout():
        coco_gt = get_coco_api_from_dataset(det_dataset)
    pure_detnet = accelerator.unwrap_model(detnet)
    iou_types = _get_iou_types(pure_detnet)
    
    # Collect tokens
    Logging("Collecting DET tokens...")
    det_z_lr, det_z_gt = [], []
    det_data = {'lq_list': [], 'gt_list': [], 'annot_list': []}
    n_collected = 0
    for batch in tqdm(det_loader, desc="DET collect", disable=not accelerator.is_local_main_process):
        gt_list, lq_list, gt_batch, lq_batch, annot_list, _, bs = prepare_batch(batch, device)
        
        with torch.no_grad():
            downsampled_lq = F.interpolate(lq_batch, scale_factor=0.5, mode='bicubic', align_corners=False)
            z_lr = tasktok.encoder(pixel_values=downsampled_lq, latent_tokens=tasktok.latent_tokens)
            
            downsampled_gt = F.interpolate(gt_batch, scale_factor=0.5, mode='bicubic', align_corners=False)
            z_gt = tasktok.encoder(pixel_values=downsampled_gt, latent_tokens=tasktok.latent_tokens)
        
        det_z_lr.append(z_lr.cpu())
        det_z_gt.append(z_gt.cpu())
        det_data['lq_list'].append(lq_batch[0].cpu())
        det_data['gt_list'].append(gt_list[0].cpu())
        det_data['annot_list'].append(annot_list[0])
        
        n_collected += bs
        if n_collected >= args.n_samples:
            break
    
    det_z_lr = torch.cat(det_z_lr, dim=0)[:args.n_samples]
    det_z_gt = torch.cat(det_z_gt, dim=0)[:args.n_samples]
    det_data['lq_list'] = det_data['lq_list'][:args.n_samples]
    det_data['gt_list'] = det_data['gt_list'][:args.n_samples]
    det_data['annot_list'] = det_data['annot_list'][:args.n_samples]
    Logging(f"Collected {det_z_lr.shape[0]} DET samples")
    
    # Greedy search
    def det_evaluate_fn(z_test):
        return evaluate_det_map(z_test, det_data, tasktok, detnet, coco_gt, iou_types, device, Logging=Logging)
    
    det_order, det_metrics, det_oracle, det_baseline = greedy_search_single_task(
        "det", det_z_lr, det_z_gt, det_evaluate_fn, Logging, accelerator
    )
    results["orders"].append(det_order)
    results["metrics"]["det"] = det_metrics
    results["oracle"]["det"] = det_oracle
    results["baseline"]["det"] = det_baseline
    
    # ==================== Segmentation ====================
    Logging("\n" + "=" * 60)
    Logging("SEGMENTATION TASK")
    Logging("=" * 60)
    
    seg_dataset = instantiate_from_config(cfg.dataset.seg.train)
    seg_loader = DataLoader(
        dataset=seg_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers, shuffle=True, drop_last=False,
        generator=dataloader_gen,
    )
    num_classes = cfg.model.segnet.params.num_classes
    Logging(f"SEG dataset: {len(seg_dataset)} images, {num_classes} classes")
    
    # Collect tokens
    Logging("Collecting SEG tokens...")
    seg_z_lr, seg_z_gt = [], []
    seg_data = {'lq': [], 'masks': []}
    n_collected = 0
    for gt, lq, mask, _ in tqdm(seg_loader, desc="SEG collect", disable=not accelerator.is_local_main_process):
        gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
        lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
        mask = mask.long().to(device)
        
        with torch.no_grad():
            downsampled_lq = F.interpolate(lq, scale_factor=0.5, mode='bicubic', align_corners=False)
            z_lr = tasktok.encoder(pixel_values=downsampled_lq, latent_tokens=tasktok.latent_tokens)
            
            downsampled_gt = F.interpolate(gt, scale_factor=0.5, mode='bicubic', align_corners=False)
            z_gt = tasktok.encoder(pixel_values=downsampled_gt, latent_tokens=tasktok.latent_tokens)
        
        seg_z_lr.append(z_lr.cpu())
        seg_z_gt.append(z_gt.cpu())
        seg_data['lq'].append(lq.cpu())
        seg_data['masks'].append(mask.cpu())
        
        n_collected += gt.size(0)
        if n_collected >= args.n_samples:
            break
    
    seg_z_lr = torch.cat(seg_z_lr, dim=0)[:args.n_samples]
    seg_z_gt = torch.cat(seg_z_gt, dim=0)[:args.n_samples]
    seg_data['lq'] = torch.cat(seg_data['lq'], dim=0)[:args.n_samples]
    seg_data['masks'] = torch.cat(seg_data['masks'], dim=0)[:args.n_samples]
    Logging(f"Collected {seg_z_lr.shape[0]} SEG samples")
    
    # Greedy search
    def seg_evaluate_fn(z_test):
        return evaluate_seg_miou(z_test, seg_data, tasktok, segnet, num_classes, device, cfg.test.batch_size)
    
    seg_order, seg_metrics, seg_oracle, seg_baseline = greedy_search_single_task(
        "seg", seg_z_lr, seg_z_gt, seg_evaluate_fn, Logging, accelerator
    )
    results["orders"].append(seg_order)
    results["metrics"]["seg"] = seg_metrics
    results["oracle"]["seg"] = seg_oracle
    results["baseline"]["seg"] = seg_baseline
    
    # ==================== Save Results ====================
    Logging("\n" + "=" * 60)
    Logging("SAVING RESULTS")
    Logging("=" * 60)
    
    # Convert orders to tensor
    orders_tensor = torch.tensor(results["orders"], dtype=torch.long)  # (3, n_tokens)
    
    # Save .pt file
    output_path = args.output if args.output else os.path.join(exp_dir, "greedy_token_order.pt")
    torch.save({
        "orders": orders_tensor,
        "task_names": results["task_names"],
    }, output_path)
    Logging(f"Saved orders to: {output_path}")
    Logging(f"Orders shape: {orders_tensor.shape}")
    
    # Save .txt file
    txt_path = output_path.replace(".pt", ".txt")
    with open(txt_path, 'w') as f:
        f.write("# Multi-task Greedy Token Replacement Order (GT → LR)\n")
        f.write(f"# n_samples: {args.n_samples}, seed: {args.seed}\n")
        f.write("=" * 60 + "\n\n")
        
        for task_name in results["task_names"]:
            idx = results["task_names"].index(task_name)
            order = results["orders"][idx]
            oracle = results["oracle"][task_name]
            baseline = results["baseline"][task_name]
            
            f.write(f"[{task_name.upper()}]\n")
            f.write(f"Oracle: {oracle:.2f}, Baseline: {baseline:.2f}\n")
            f.write(f"Order (least → most important): {order[:10]}...{order[-10:]}\n")
            f.write(f"Full order: {order}\n\n")
    
    Logging(f"Saved summary to: {txt_path}")
    
    # Print summary
    Logging("\n" + "=" * 60)
    Logging("SUMMARY")
    Logging("=" * 60)
    for task_name in results["task_names"]:
        idx = results["task_names"].index(task_name)
        order = results["orders"][idx]
        oracle = results["oracle"][task_name]
        baseline = results["baseline"][task_name]
        Logging(f"[{task_name.upper()}] Oracle: {oracle:.2f}, Baseline: {baseline:.2f}")
        Logging(f"  Least important 5: {order[:5]}")
        Logging(f"  Most important 5:  {order[-5:]}")
    
    accelerator.wait_for_everyone()
    Logging("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples per task")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output .pt file path")
    parser.add_argument("--save-img", action='store_true')
    args = parser.parse_args()
    main(args)
