python create_DeFlow_train_dataset.py -source BST/task2/ -target BST/processed/task2/train-noisy-images  -scales 0.25 0.5 1 2 4 8 16 32

###############################
##### Train
###############################


# From codes/
CUDA_VISIBLE_DEVICES=3 python train.py -opt confs/task2/1_bst_task2.yml --model_path ../experiments/8_bst_task1/models/latest_G.pth
CUDA_VISIBLE_DEVICES=0 python train.py -opt confs/task2/2_bst_task2.yml
CUDA_VISIBLE_DEVICES=0 python train.py -opt confs/task2/3_bst_task2.yml
CUDA_VISIBLE_DEVICES=2,3 python train_double.py -opt confs/task2/4_bst_task2_double.yml


################################
# Validate metrics
# Note that validating metrics does not make much sense here
# as we don't have groundtruth clean images for the provided
# set of noisy images
################################


CUDA_VISIBLE_DEVICES=0 python validate.py -opt task2/1_bst_task2.yml -model_path ../experiments/1_bst_task2/models/latest_G.pth -crop_size 256
CUDA_VISIBLE_DEVICES=0 python validate.py -opt task2/2_bst_task2.yml -model_path ../experiments/2_bst_task2/models/latest_G.pth -crop_size 256
CUDA_VISIBLE_DEVICES=0 python validate.py -opt task2/3_bst_task2.yml -model_path ../experiments/3_bst_task2/models/latest_G.pth -crop_size 256
CUDA_VISIBLE_DEVICES=3 python validate.py -opt task2/4_bst_task2_double.yml -model_path ../experiments/4_bst_task2/models/30000_G_0.pth -crop_size 256


################################
# Train super-resolution models
################################

# From codes/; generate data for the super-resolution model
CUDA_VISIBLE_DEVICES=0 python translate.py -opt task2/1_bst_task2.yml -model_path ../experiments/1_bst_task2/models/latest_G.pth -source_dir ../datasets/AIM-RWSR/train-clean-images/2x/ -out_dir ../datasets/AIM-RWSR/train-clean-images/task2/2x_degraded_1/ --no_orig
CUDA_VISIBLE_DEVICES=0 python translate.py -opt task2/2_bst_task2.yml -model_path ../experiments/2_bst_task2/models/latest_G.pth -source_dir ../datasets/AIM-RWSR/train-clean-images/2x/ -out_dir ../datasets/AIM-RWSR/train-clean-images/task2/2x_degraded/ --no_orig
CUDA_VISIBLE_DEVICES=1 python translate.py -opt task2/3_bst_task2.yml -model_path ../experiments/3_bst_task2/models/latest_G.pth -source_dir ../datasets/AIM-RWSR/train-clean-images/2x/ -out_dir ../datasets/AIM-RWSR/train-clean-images/task2/2x_degraded_3/ --no_orig
CUDA_VISIBLE_DEVICES=3 python translate.py -opt task2/4_bst_task2_double.yml -model_path ../experiments/4_bst_task2/models/30000_G_0.pth -source_dir ../datasets/AIM-RWSR/train-clean-images/2x/ -out_dir ../datasets/AIM-RWSR/train-clean-images/task2/2x_degraded_4_0/ --no_orig
CUDA_VISIBLE_DEVICES=3 python translate.py -opt task2/4_bst_task2_double.yml -model_path ../experiments/4_bst_task2/models/30000_G_1.pth -source_dir ../datasets/AIM-RWSR/train-clean-images/2x/ -out_dir ../datasets/AIM-RWSR/train-clean-images/task2/2x_degraded_4_1/ --no_orig

# From Real-SR/codes
CUDA_VISIBLE_DEVICES=0 python3 train.py -opt ../../ESRGAN_confs/train/task2/1_bst_task2.yml --model_path ../experiments/8_bst_task1/models/latest_G.pth
CUDA_VISIBLE_DEVICES=1 python3 train.py -opt ../../ESRGAN_confs/train/task2/2_bst_task2.yml
CUDA_VISIBLE_DEVICES=2 python3 train.py -opt ../../ESRGAN_confs/train/task2/3_bst_task2.yml
CUDA_VISIBLE_DEVICES=5 python3 train.py -opt ../../ESRGAN_confs/train/task2/4_bst_task2.yml

# Test
CUDA_VISIBLE_DEVICES=1 python test.py -opt ../../ESRGAN_confs/test/task2/1_bst_task2.yml
CUDA_VISIBLE_DEVICES=1 python test.py -opt ../../ESRGAN_confs/test/task2/2_bst_task2.yml
CUDA_VISIBLE_DEVICES=1 python test.py -opt ../../ESRGAN_confs/test/task2/3_bst_task2.yml
CUDA_VISIBLE_DEVICES=1 python test.py -opt ../../ESRGAN_confs/test/task2/4_bst_task2.yml

# Test (qualitative analysis)
CUDA_VISIBLE_DEVICES=0 python translate.py -opt task2/1_bst_task2.yml -model_path ../experiments/1_bst_task2/models/latest_G.pth -source_dir ../datasets/BST/processed/task2/train-noisy-images/1x/ -out_dir ../translated/1_bst_task2
CUDA_VISIBLE_DEVICES=0 python translate.py -opt task2/2_bst_task2.yml -model_path ../experiments/2_bst_task2/models/latest_G.pth -source_dir ../datasets/BST/processed/task2/train-noisy-images/1x/ -out_dir ../translated/2_bst_task2
CUDA_VISIBLE_DEVICES=1 python translate.py -opt task2/3_bst_task2.yml -model_path ../experiments/3_bst_task2/models/latest_G.pth -source_dir ../datasets/BST/processed/task2/train-noisy-images/1x/ -out_dir ../translated/3_bst_task2
CUDA_VISIBLE_DEVICES=2 python translate.py -opt task2/4_bst_task2_double.yml -model_path ../experiments/4_bst_task2/models/ -source_dir ../datasets/BST/processed/task2/train-noisy-images/1x/ -out_dir ../translated/4_bst_task2
