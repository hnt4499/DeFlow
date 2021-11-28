# From datasets/
# python create_DeFlow_train_dataset.py -source ./AIM-RWSR/train-clean-images/ -target ./AIM-RWSR/train-clean-images/ -scales 16
python create_DeFlow_train_dataset.py -source ./AIM-RWSR/train-clean-images/1x/ -target ./AIM-RWSR/train-clean-images/ -scales 2

mkdir -p BST/processed/task1/train-noisy-images
python create_DeFlow_train_dataset.py -source BST/task1/Input/ -target BST/processed/task1/train-noisy-images/ -scales 0.25 0.5 1 2 4 8 16 32

mkdir -p BST/processed/task1/val-gt-clean
# python create_DeFlow_train_dataset.py -source BST/task1/Ground\ Truth/ -target BST/processed/task1/val-gt-clean/ -scales 2 8
python create_DeFlow_train_dataset.py -source BST/task1/Ground\ Truth/ -target BST/processed/task1/val-gt-clean/ -scales 1 16

mkdir -p BST/processed/task1/val-input-noisy
# python create_DeFlow_train_dataset.py -source BST/task1/Input/ -target BST/processed/task1/val-input-noisy/ -scales 1 4
python create_DeFlow_train_dataset.py -source BST/task1/Input/ -target BST/processed/task1/val-input-noisy/ -scales 8

# From codes/
python compute_dataset_statistics.py ../datasets/AIM-RWSR/train-clean-images/4x/
python compute_dataset_statistics.py ../datasets/AIM-RWSR/train-clean-images/16x/

python compute_dataset_statistics.py ../datasets/BST/processed/task1/train-noisy-images/1x/
python compute_dataset_statistics.py ../datasets/BST/processed/task1/train-noisy-images/4x/

# From codes/
python train.py -opt confs/1_bst_task1.yml
python train.py -opt confs/2_bst_task1.yml
python train.py -opt confs/3_bst_task1.yml --model_path ../trained_models/DeFlow_models/DeFlow-AIM-RWSR-100k.pth
python train.py -opt confs/4_bst_task1.yml --model_path ../trained_models/DeFlow_models/DeFlow-AIM-RWSR-100k.pth

CUDA_VISIBLE_DEVICES=-1 python validate.py -opt DeFlow-AIM-RWSR.yml -model_path ../trained_models/DeFlow_models/DeFlow-AIM-RWSR-100k.pth -crop_size 256 -n_max 5

CUDA_VISIBLE_DEVICES=0 python translate.py -opt 1_bst_task1.yml -model_path ../experiments/1_bst_task1/models/latest_G.pth -source_dir ../datasets/BST/task1/Ground\ Truth/ -out_dir ../translate/1_bst_task1 && \
    CUDA_VISIBLE_DEVICES=0 python translate.py -opt 2_bst_task1.yml -model_path ../experiments/2_bst_task1/models/9400_G.pth -source_dir ../datasets/BST/task1/Ground\ Truth/ -out_dir ../translate/2_bst_task1 && \
    CUDA_VISIBLE_DEVICES=0 python translate.py -opt 3_bst_task1.yml -model_path ../experiments/3_bst_task1/models/latest_G.pth -source_dir ../datasets/BST/task1/Ground\ Truth/ -out_dir ../translate/3_bst_task1 && \
    CUDA_VISIBLE_DEVICES=0 python translate.py -opt 4_bst_task1.yml -model_path ../experiments/4_bst_task1/models/latest_G.pth -source_dir ../datasets/BST/task1/Ground\ Truth/ -out_dir ../translate/4_bst_task1 && \
    CUDA_VISIBLE_DEVICES=0 python translate.py -opt 5_bst_task1.yml -model_path ../experiments/5_bst_task1/models/latest_G.pth -source_dir ../datasets/BST/task1/Ground\ Truth/ -out_dir ../translate/5_bst_task1 && \
    CUDA_VISIBLE_DEVICES=0 python translate.py -opt 6_bst_task1.yml -model_path ../experiments/6_bst_task1/models/latest_G.pth -source_dir ../datasets/BST/task1/Ground\ Truth/ -out_dir ../translate/6_bst_task1 && \
    CUDA_VISIBLE_DEVICES=0 python translate.py -opt 7_bst_task1.yml -model_path ../experiments/7_bst_task1/models/latest_G.pth -source_dir ../datasets/BST/task1/Ground\ Truth/ -out_dir ../translate/7_bst_task1 && \
    CUDA_VISIBLE_DEVICES=1 python translate.py -opt 8_bst_task1.yml -model_path ../experiments/8_bst_task1/models/latest_G.pth -source_dir ../datasets/BST/task1/Ground\ Truth/ -out_dir ../translate/8_bst_task1 && \
    CUDA_VISIBLE_DEVICES=1 python translate.py -opt 9_bst_task1.yml -model_path ../experiments/9_bst_task1/models/latest_G.pth -source_dir ../datasets/BST/task1/Ground\ Truth/ -out_dir ../translate/9_bst_task1 && \
    CUDA_VISIBLE_DEVICES=0 python translate.py -opt 10_bst_task1.yml -model_path ../experiments/10_bst_task1/models/latest_G.pth -source_dir ../datasets/BST/task1/Ground\ Truth/ -out_dir ../translate/10_bst_task1 && \
    CUDA_VISIBLE_DEVICES=0 python translate.py -opt 11_bst_task1.yml -model_path ../experiments/11_bst_task1/models/latest_G.pth -source_dir ../datasets/BST/task1/Ground\ Truth/ -out_dir ../translate/11_bst_task1



################################
# Validate metrics
################################

# From codes/
CUDA_VISIBLE_DEVICES=2 python validate.py -opt 8_bst_task1.yml -model_path ../experiments/8_bst_task1/models/latest_G.pth -crop_size 256
CUDA_VISIBLE_DEVICES=0 python validate.py -opt 12_bst_task1.yml -model_path ../experiments/12_bst_task1/models/65000_G.pth -crop_size 256
CUDA_VISIBLE_DEVICES=1 python validate.py -opt 13_bst_task1.yml -model_path ../experiments/13_bst_task1/models/65000_G.pth -crop_size 256
CUDA_VISIBLE_DEVICES=3 python validate.py -opt 14_bst_task1.yml -model_path ../experiments/14_bst_task1/models/50000_G.pth -crop_size 256
CUDA_VISIBLE_DEVICES=1 python validate.py -opt 15_bst_task1.yml -model_path ../experiments/15_bst_task1/models/60000_G.pth -crop_size 256



################################
# Train super-resolution models
################################

# From codes/; generate data for super-resolution model
CUDA_VISIBLE_DEVICES=0 python translate.py -opt 8_bst_task1.yml -model_path ../experiments/8_bst_task1/models/latest_G.pth -source_dir ../datasets/AIM-RWSR/train-clean-images/2x/ -out_dir ../datasets/AIM-RWSR/train-clean-images/2x_degraded/ --no_orig
CUDA_VISIBLE_DEVICES=0 python translate.py -opt 12_bst_task1.yml -model_path ../experiments/12_bst_task1/models/65000_G.pth -source_dir ../datasets/AIM-RWSR/train-clean-images/2x/ -out_dir ../datasets/AIM-RWSR/train-clean-images/2x_degraded_12/ --no_orig
CUDA_VISIBLE_DEVICES=0 python translate.py -opt 13_bst_task1.yml -model_path ../experiments/13_bst_task1/models/65000_G.pth -source_dir ../datasets/AIM-RWSR/train-clean-images/2x/ -out_dir ../datasets/AIM-RWSR/train-clean-images/2x_degraded_13/ --no_orig
CUDA_VISIBLE_DEVICES=0 python translate.py -opt 14_bst_task1.yml -model_path ../experiments/14_bst_task1/models/50000_G.pth -source_dir ../datasets/AIM-RWSR/train-clean-images/2x/ -out_dir ../datasets/AIM-RWSR/train-clean-images/2x_degraded_14/ --no_orig
CUDA_VISIBLE_DEVICES=0 python translate.py -opt 15_bst_task1.yml -model_path ../experiments/15_bst_task1/models/60000_G.pth -source_dir ../datasets/AIM-RWSR/train-clean-images/2x/ -out_dir ../datasets/AIM-RWSR/train-clean-images/2x_degraded_15/ --no_orig


# From Real-SR/codes
CUDA_VISIBLE_DEVICES=0 python3 train.py -opt ../../ESRGAN_confs/train/8_bst_task1.yml
CUDA_VISIBLE_DEVICES=1 python3 train.py -opt ../../ESRGAN_confs/train/12_bst_task1.yml
CUDA_VISIBLE_DEVICES=3 python3 train.py -opt ../../ESRGAN_confs/train/13_bst_task1.yml

# From Real-SR/codes; to test
CUDA_VISIBLE_DEVICES=1 python test.py -opt ../../ESRGAN_confs/test/8_bst_task1.yml