import os, sys
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import utils.filter_warning

import csv
import torch
import numpy as np
from tqdm import tqdm
from model import SwinIR
from torch.nn import functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torchvision.utils import save_image

from utils.common import (
    instantiate_from_config, load_network,
    calculate_psnr_pt, calculate_lpips_pt
)
from utils.classification import calculate_accuracy
from utils.detection import (
    CocoEvaluator, prepare_batch,
    get_coco_api_from_dataset, batch_to_list,
    collate_fn, _get_iou_types, draw_box, suppress_stdout
)
from utils.segmentation import convert2color, calculate_mat, compute_iou
from utils.tasktok import (
    adapt_titok_to_image_size,
    prepare_tasktok_input,
    reconstruct_from_tasktok_input,
)

from model.titok.modeling import TiTok
from model.token_predictor import TokenPredictor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


# VOC class names for segmentation
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor'
]


def setup_output_dir(output_dir: str, task: str) -> dict:
    """Create output directories for test results."""
    dirs = {
        "root": output_dir,
        "images": os.path.join(output_dir, task, "images"),
        "results": os.path.join(output_dir, task),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def resolve_ckpt_dir(cfg) -> str:
    ckpt_dir = cfg.test.get("ckpt_dir")
    if ckpt_dir is None:
        ckpt_dir = os.path.join(os.path.dirname(cfg.test.output_dir), "checkpoints")
        print(f"ckpt_dir is not set. Falling back to: {ckpt_dir}")
    return ckpt_dir


def resolve_resume_path(cfg, key: str, fallback_filename: str) -> str:
    resume_path = cfg.test.get(key)
    if resume_path is None:
        resume_path = os.path.join(resolve_ckpt_dir(cfg), fallback_filename)
        print(f"{key} is not set. Falling back to: {resume_path}")
    return resume_path


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


def load_models(cfg, device, task: str = "all"):
    """Load common models and only the task-specific models needed for testing."""
    models = {}
    load_all = task == "all"
    
    # SwinIR
    swinir = instantiate_from_config(cfg.model.swinir)
    if cfg.model.pre_restoration:
        swinir.load_state_dict(torch.load(cfg.test.resume_swinir, map_location="cpu"), strict=True)
        print(f"Load SwinIR from: {cfg.test.resume_swinir}")
    swinir.eval().to(device)
    models["swinir"] = swinir
    
    # TiTok
    resume_tasktok = resolve_resume_path(cfg, "resume_tasktok", "tasktok_last.pt")
    tasktok = TiTok.from_pretrained(cfg.model.titok.pretrained)
    resize_input = bool(cfg.model.titok.get("resize_input", False))
    tasktok_input_size = 512 if resize_input else tasktok.encoder.image_size
    if resize_input and tasktok_input_size != tasktok.encoder.image_size:
        adapt_titok_to_image_size(tasktok, tasktok_input_size)
        print(f"Adapt TiTok input/output size to {tasktok_input_size}")
    elif not resize_input:
        print(f"Use legacy TiTok resize path with input size {tasktok_input_size}")
    tasktok.load_state_dict(torch.load(resume_tasktok, map_location="cpu"), strict=True)
    print(f"Load TaskTok from: {resume_tasktok}")
    tasktok.eval().to(device)
    models["tasktok"] = tasktok
    models["tasktok_input_size"] = tasktok_input_size
    models["resize_input"] = resize_input
    
    # Token Predictor
    resume_token_predictor = resolve_resume_path(cfg, "resume_token_predictor", "token_predictor_last.pt")
    token_predictor = TokenPredictor(
        d_token=tasktok.quantize.token_size,
        d_model=cfg.model.token_predictor.params.d_model,
        n_heads=cfg.model.token_predictor.params.n_heads,
        n_layers=cfg.model.token_predictor.params.n_layers,
        n_tokens=tasktok.num_latent_tokens,
    ).to(device)
    token_predictor.load_state_dict(torch.load(resume_token_predictor, map_location="cpu"))
    print(f"Load TokenPredictor from: {resume_token_predictor}")
    token_predictor.eval()
    models["token_predictor"] = token_predictor
    
    # Classification network
    if load_all or task == "cls":
        resume_clsnet = resolve_resume_path(cfg, "resume_clsnet", "clsnet_last.pt")
        clsnet = instantiate_from_config(cfg.model.clsnet)
        clsnet = load_network(clsnet, resume_clsnet, strict=True)
        print(f"Load ClassificationNetwork from: {resume_clsnet}")
        clsnet.eval().to(device)
        models["clsnet"] = clsnet
    
    # Detection network
    if load_all or task == "det":
        resume_detnet = resolve_resume_path(cfg, "resume_detnet", "detnet_last.pt")
        detnet = instantiate_from_config(cfg.model.detnet)
        detnet = load_network(detnet, resume_detnet, strict=True)
        print(f"Load DetectionNetwork from: {resume_detnet}")
        detnet.eval().to(device)
        models["detnet"] = detnet
    
    # Segmentation network
    if load_all or task == "seg":
        resume_segnet = resolve_resume_path(cfg, "resume_segnet", "segnet_last.pt")
        segnet = instantiate_from_config(cfg.model.segnet)
        segnet = load_network(segnet, resume_segnet, strict=True)
        print(f"Load SegmentationNetwork from: {resume_segnet}")
        segnet.eval().to(device)
        models["segnet"] = segnet
    
    return models


def test_classification(cfg, models, device, output_dir):
    """Test classification task."""
    print("\n" + "="*50)
    print("Testing Classification")
    print("="*50)
    
    dirs = setup_output_dir(output_dir, "cls")
    
    # Load dataset
    cls_val_dataset = instantiate_from_config(cfg.dataset.cls.val)
    cls_val_loader = DataLoader(
        dataset=cls_val_dataset, batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers, shuffle=False, drop_last=False
    )
    print(f"Classification test dataset: {len(cls_val_dataset)} images")
    
    swinir = models["swinir"]
    tasktok = models["tasktok"]
    token_predictor = models["token_predictor"]
    clsnet = models["clsnet"]
    tasktok_input_size = models["tasktok_input_size"]
    resize_input = models["resize_input"]
    
    results = []
    val_acc1_list = []
    val_acc5_list = []
    val_psnr_list = []
    val_lpips_list = []
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type='squeeze', reduction='none'
    ).to(device).eval()
    
    pbar = tqdm(cls_val_loader, desc="CLS Testing")
    for batch_idx, (val_gt, val_lq, val_label, val_path) in enumerate(pbar):
        val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
        val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
        
        with torch.no_grad():
            val_pre_res = val_lq
            if cfg.model.pre_restoration:
                val_pre_res = swinir(val_lq)
            
            z = encode_tasktok_image(tasktok, val_pre_res, tasktok_input_size, resize_input)
            z_pred, _, _, _, _ = token_predictor(z, task_id=0)
            val_res = decode_tasktok_tokens(tasktok, z_pred, val_lq.shape[-2:], resize_input)
            
            val_pred = clsnet(val_res, normalize=True)
            acc1, acc5 = calculate_accuracy(val_pred.cpu(), val_label.cpu(), topk=(1, 5))
            val_acc1_list.append(acc1)
            val_acc5_list.append(acc5)

            val_psnr_batch = calculate_psnr_pt(val_res, val_gt, crop_border=0).detach().cpu()
            val_psnr_list.extend([v.item() for v in val_psnr_batch])

            val_lpips_batch = calculate_lpips_pt(
                val_res, val_gt, net_lpips=lpips_metric, crop_border=0, img_range=1.0
            ).detach().flatten().cpu()
            val_lpips_list.extend([v.item() for v in val_lpips_batch])
            
            # Per-sample results
            pred_class = val_pred.argmax(dim=1)
            for i in range(val_pred.size(0)):
                sample_idx = batch_idx * cfg.test.batch_size + i
                results.append({
                    "path": val_path[i] if isinstance(val_path, list) else str(sample_idx),
                    "gt_label": val_label[i].item(),
                    "pred_label": pred_class[i].item(),
                    "correct": (pred_class[i] == val_label[i]).item(),
                    "confidence": torch.softmax(val_pred[i], dim=0)[pred_class[i]].item(),
                    "psnr": val_psnr_batch[i].item(),
                    "lpips": val_lpips_batch[i].item()
                })
            
            # Save images
            if cfg.test.save_images:
                for i in range(val_gt.size(0)):
                    sample_idx = batch_idx * cfg.test.batch_size + i
                    gt_label = int(val_label[i].item())
                    pred_label = int(pred_class[i].item())
                    status = "CORRECT" if pred_label == gt_label else "WRONG"
                    sample_name = f"sample_{sample_idx:06d}_gt{gt_label:04d}_pred{pred_label:04d}_{status}"
                    save_image(val_lq[i], os.path.join(dirs["images"], f"{sample_name}_lq.png"))
                    save_image(val_res[i], os.path.join(dirs["images"], f"{sample_name}_restored.png"))
                    # save_image(val_pre_res[i], os.path.join(dirs["images"], f"{sample_name}_pre_restored.png"))

                    save_image(val_gt[i], os.path.join(dirs["images"], f"{sample_name}_gt.png"))
        
        pbar.set_postfix({
            "acc1": f"{np.mean(val_acc1_list):.2f}%",
            "psnr": f"{np.mean(val_psnr_list):.2f}",
            "lpips": f"{np.mean(val_lpips_list):.4f}"
        })
    
    # Calculate final metrics
    avg_acc1 = np.mean(val_acc1_list)
    avg_acc5 = np.mean(val_acc5_list)
    avg_psnr = np.mean(val_psnr_list)
    avg_lpips = np.mean(val_lpips_list)
    
    print(f"\nClassification Results:")
    print(f"  Top-1 Accuracy: {avg_acc1:.2f}%")
    print(f"  Top-5 Accuracy: {avg_acc5:.2f}%")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  LPIPS: {avg_lpips:.4f}")
    
    # Save results
    if cfg.test.save_csv:
        csv_path = os.path.join(dirs["results"], "results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["path", "gt_label", "pred_label", "correct", "confidence", "psnr", "lpips"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"  Results saved to: {csv_path}")
    
    # Save summary
    summary_path = os.path.join(dirs["results"], "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Classification Test Results\n")
        f.write(f"="*50 + "\n")
        f.write(f"Dataset: {len(cls_val_dataset)} images\n")
        f.write(f"Top-1 Accuracy: {avg_acc1:.2f}%\n")
        f.write(f"Top-5 Accuracy: {avg_acc5:.2f}%\n")
        f.write(f"PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"LPIPS: {avg_lpips:.4f}\n")
    print(f"  Summary saved to: {summary_path}")
    
    return {"acc1": avg_acc1, "acc5": avg_acc5, "psnr": avg_psnr, "lpips": avg_lpips}


def test_detection(cfg, models, device, output_dir):
    """Test detection task."""
    print("\n" + "="*50)
    print("Testing Detection")
    print("="*50)
    
    dirs = setup_output_dir(output_dir, "det")
    box_dir = os.path.join(dirs["images"], "boxes")
    os.makedirs(box_dir, exist_ok=True)
    is_coco = bool(cfg.dataset.get("is_coco"))
    vis_score_threshold = float(cfg.test.get("det_vis_score_threshold", 0.8))
    
    # Load dataset
    with suppress_stdout():
        det_val_dataset = instantiate_from_config(cfg.dataset.det.val)
    det_val_loader = DataLoader(
        dataset=det_val_dataset, batch_size=1,
        num_workers=cfg.test.num_workers, shuffle=False, pin_memory=True, collate_fn=collate_fn
    )
    print(f"Detection test dataset: {len(det_val_dataset)} images")
    
    swinir = models["swinir"]
    tasktok = models["tasktok"]
    token_predictor = models["token_predictor"]
    detnet = models["detnet"]
    tasktok_input_size = models["tasktok_input_size"]
    resize_input = models["resize_input"]
    
    # Setup COCO evaluator
    print("Preparing COCO API...")
    with suppress_stdout():
        coco = get_coco_api_from_dataset(det_val_dataset)
    iou_types = _get_iou_types(detnet)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    
    val_psnr_list = []
    
    pbar = tqdm(det_val_loader, desc="DET Testing")
    for batch_idx, val_batch in enumerate(pbar):
        val_gt_list, val_lq_list, _, val_lq_batch, val_annot_list, val_path_list, val_bs = prepare_batch(val_batch, device)
        
        with torch.no_grad():
            val_pre_res_batch = val_lq_batch
            if cfg.model.pre_restoration:
                val_pre_res_batch = swinir(val_lq_batch)
            
            z = encode_tasktok_image(tasktok, val_pre_res_batch, tasktok_input_size, resize_input)
            z_pred, _, _, _, _ = token_predictor(z, task_id=1)
            val_res_batch = decode_tasktok_tokens(tasktok, z_pred, val_lq_batch.shape[-2:], resize_input)
            val_res_list = batch_to_list(val_res_batch, val_gt_list)
            
            val_pred_list, _ = detnet(val_res_list)
            val_pred_list = [{k: v.cpu() for k, v in t.items()} for t in val_pred_list]
            
            # Calculate PSNR
            val_gt, val_res = val_gt_list[0].unsqueeze(0), val_res_list[0].unsqueeze(0)
            val_psnr = calculate_psnr_pt(val_res, val_gt, crop_border=0).item()
            val_psnr_list.append(val_psnr)
            
            # Update COCO evaluator
            res = {annot["image_id"]: pred for annot, pred in zip(val_annot_list, val_pred_list)}
            coco_evaluator.update(res)
            
            # Save images
            if cfg.test.save_images:
                basename = os.path.splitext(os.path.basename(val_path_list[0]))[0] if val_path_list else f"sample_{batch_idx:04d}"
                save_image(val_lq_list[0], os.path.join(dirs["images"], f"{basename}_lq.png"))
                save_image(val_res_list[0], os.path.join(dirs["images"], f"{basename}_restored.png"))
                save_image(val_gt_list[0], os.path.join(dirs["images"], f"{basename}_gt.png"))
                val_annot_cpu = {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in val_annot_list[0].items()
                }
                pred_overlay = draw_box(
                    val_res_list[0], val_pred_list[0], is_coco=is_coco,
                    score_threshold=vis_score_threshold, fontsize=0.7, split_acc=True
                )
                gt_overlay = draw_box(
                    val_gt_list[0], val_annot_cpu, is_coco=is_coco,
                    score_threshold=0.0, fontsize=0.7, split_acc=False
                )
                save_image(pred_overlay, os.path.join(box_dir, f"{basename}_pred_box.png"))
                save_image(gt_overlay, os.path.join(box_dir, f"{basename}_gt_box.png"))
        
        pbar.set_postfix({"psnr": f"{np.mean(val_psnr_list):.2f}"})
    
    # Calculate final metrics
    coco_evaluator.synchronize_between_processes()
    with suppress_stdout():
        coco_evaluator.accumulate()
    
    # Get COCO results
    print("\nDetection Results:")
    coco_evaluator.summarize()  # This prints the results
    
    # Extract mAP from coco_eval stats
    det_results = {}
    coco_eval = coco_evaluator.coco_eval.get("bbox")
    if coco_eval is not None and hasattr(coco_eval, "stats"):
        det_results["mAP@[0.5:0.95]"] = float(coco_eval.stats[0]) * 100
        det_results["mAP@0.5"] = float(coco_eval.stats[1]) * 100
    else:
        det_results["mAP@[0.5:0.95]"] = 0.0
        det_results["mAP@0.5"] = 0.0
    
    avg_psnr = np.mean(val_psnr_list)
    
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  mAP@[0.5:0.95]: {det_results['mAP@[0.5:0.95]']:.2f}%")
    print(f"  mAP@0.5: {det_results['mAP@0.5']:.2f}%")
    
    # Save summary
    summary_path = os.path.join(dirs["results"], "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Detection Test Results\n")
        f.write(f"="*50 + "\n")
        f.write(f"Dataset: {len(det_val_dataset)} images\n")
        f.write(f"PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"mAP@[0.5:0.95]: {det_results['mAP@[0.5:0.95]']:.2f}%\n")
        f.write(f"mAP@0.5: {det_results['mAP@0.5']:.2f}%\n")
    print(f"  Summary saved to: {summary_path}")
    
    return {"psnr": avg_psnr, **det_results}


def test_segmentation(cfg, models, device, output_dir):
    """Test segmentation task."""
    print("\n" + "="*50)
    print("Testing Segmentation")
    print("="*50)
    
    dirs = setup_output_dir(output_dir, "seg")
    
    # Load dataset
    seg_val_dataset = instantiate_from_config(cfg.dataset.seg.val)
    seg_val_loader = DataLoader(
        dataset=seg_val_dataset, batch_size=1,
        num_workers=cfg.test.num_workers, shuffle=False, drop_last=False
    )
    print(f"Segmentation test dataset: {len(seg_val_dataset)} images")
    
    swinir = models["swinir"]
    tasktok = models["tasktok"]
    token_predictor = models["token_predictor"]
    segnet = models["segnet"]
    tasktok_input_size = models["tasktok_input_size"]
    resize_input = models["resize_input"]
    
    num_classes = cfg.model.segnet.params.num_classes
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    
    pbar = tqdm(seg_val_loader, desc="SEG Testing")
    for batch_idx, (val_gt, val_lq, val_mask, val_path) in enumerate(pbar):
        val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float().to(device)
        val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
        val_mask = val_mask.long().to(device)
        
        with torch.no_grad():
            # Prepare input for tasktok (patch-based processing)
            val_lq_patch, val_patch_meta = prepare_tasktok_input(val_lq, target_size=512)
            
            val_pre_res = val_lq_patch
            if cfg.model.pre_restoration:
                val_pre_res = swinir(val_lq_patch)
            
            z = encode_tasktok_image(tasktok, val_pre_res, tasktok_input_size, resize_input)
            z_pred, _, _, _, _ = token_predictor(z, task_id=2)
            val_res_patches = decode_tasktok_tokens(tasktok, z_pred, val_lq_patch.shape[-2:], resize_input)
            val_res = reconstruct_from_tasktok_input(val_res_patches, val_patch_meta, target_size=512)
            
            val_pred = segnet(val_res)
            val_pred_class = val_pred["out"].argmax(dim=1)
            
            # Update confusion matrix
            confmat += calculate_mat(val_mask.flatten(), val_pred_class.flatten(), n=num_classes)
            
            # Save images
            if cfg.test.save_images:
                save_image(val_lq[0], os.path.join(dirs["images"], f"sample_{batch_idx:04d}_lq.png"))
                save_image(val_res[0], os.path.join(dirs["images"], f"sample_{batch_idx:04d}_restored.png"))
                save_image(val_gt[0], os.path.join(dirs["images"], f"sample_{batch_idx:04d}_gt.png"))

                # Save colored mask and prediction
                mask_color = convert2color(val_mask).float()
                pred_color = convert2color(val_pred_class).float()
                save_image(mask_color[0], os.path.join(dirs["images"], f"sample_{batch_idx:04d}_mask.png"))
                save_image(pred_color[0], os.path.join(dirs["images"], f"sample_{batch_idx:04d}_pred.png"))
    
    # Calculate IoU for each class
    iou_per_class = compute_iou(confmat)
    miou = iou_per_class.mean().item() * 100
    
    print(f"\nSegmentation Results:")
    print(f"  mIoU: {miou:.2f}%")
    print(f"\n  Per-class IoU:")
    for i, (cls_name, iou) in enumerate(zip(VOC_CLASSES, iou_per_class)):
        print(f"    {cls_name:15s}: {iou.item()*100:.2f}%")
    
    # Save results
    if cfg.test.save_csv:
        csv_path = os.path.join(dirs["results"], "class_iou.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["class", "iou"])
            for cls_name, iou in zip(VOC_CLASSES, iou_per_class):
                writer.writerow([cls_name, f"{iou.item()*100:.2f}"])
        print(f"  Class IoU saved to: {csv_path}")
    
    # Save summary
    summary_path = os.path.join(dirs["results"], "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Segmentation Test Results\n")
        f.write(f"="*50 + "\n")
        f.write(f"Dataset: {len(seg_val_dataset)} images\n")
        f.write(f"mIoU: {miou:.2f}%\n\n")
        f.write(f"Per-class IoU:\n")
        for cls_name, iou in zip(VOC_CLASSES, iou_per_class):
            f.write(f"  {cls_name:15s}: {iou.item()*100:.2f}%\n")
    print(f"  Summary saved to: {summary_path}")
    
    return {"miou": miou, "iou_per_class": iou_per_class.cpu().numpy()}


def main(args):
    """Main test function."""
    # Setup
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Config: {args.config}")
    print(f"Task: {args.task}")
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = cfg.test.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load models
    print("\nLoading models...")
    models = load_models(cfg, device, task=args.task)
    
    # Run tests
    results = {}
    
    if args.task in ["cls", "all"]:
        if "clsnet" in models:
            results["cls"] = test_classification(cfg, models, device, output_dir)
        else:
            print("Warning: Classification network not loaded. Skipping CLS test.")
    
    if args.task in ["det", "all"]:
        if "detnet" in models:
            results["det"] = test_detection(cfg, models, device, output_dir)
        else:
            print("Warning: Detection network not loaded. Skipping DET test.")
    
    if args.task in ["seg", "all"]:
        if "segnet" in models:
            results["seg"] = test_segmentation(cfg, models, device, output_dir)
        else:
            print("Warning: Segmentation network not loaded. Skipping SEG test.")
    
    # Print final summary
    print("\n" + "="*50)
    print("Test Complete!")
    print("="*50)
    
    if "cls" in results:
        print(
            f"Classification - Top-1: {results['cls']['acc1']:.2f}%, "
            f"Top-5: {results['cls']['acc5']:.2f}%, "
            f"PSNR: {results['cls']['psnr']:.2f} dB, LPIPS: {results['cls']['lpips']:.4f}"
        )
    if "det" in results:
        print(f"Detection - mAP@[0.5:0.95]: {results['det']['mAP@[0.5:0.95]']:.2f}%")
    if "seg" in results:
        print(f"Segmentation - mIoU: {results['seg']['miou']:.2f}%")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--task", type=str, default="all", choices=["cls", "det", "seg", "all"],
                        help="Task to test: cls, det, seg, or all")
    args = parser.parse_args()
    main(args)
