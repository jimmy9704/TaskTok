# dataset generation
python datasets/val_data_generation/gen_cls-dataset.py --config datasets/val_data_generation/config/cls/imagenet-deg-mxb.yaml  # imagenet for cls
python datasets/val_data_generation/gen_seg-dataset.py --config datasets/val_data_generation/config/seg/pascalvoc-deg-mxb.yaml  # VOC2012 for segmentation
python datasets/val_data_generation/gen_det-dataset.py --config datasets/val_data_generation/config/det/pascalvoc-deg-mxb.yaml  # VOC2012 for detection

# greedy search
CUDA_VISIBLE_DEVICES=0 python main/greedy_search.py --config configs/greedy_search_bl64.yaml --n_samples 300

# training
##64
CUDA_VISIBLE_DEVICES=0 accelerate launch main/train_tasktok.py --config configs/tasktok_bl64.yaml
##256
CUDA_VISIBLE_DEVICES=0 accelerate launch main/train_tasktok.py --config configs/tasktok_sl256.yaml

### inference
CUDA_VISIBLE_DEVICES=0 accelerate launch main/test_tasktok.py --config configs/tasktok_bl64_test.yaml
