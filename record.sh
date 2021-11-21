# From datasets/
python create_DeFlow_train_dataset.py -source ./AIM-RWSR/train-clean-images/ -target ./AIM-RWSR/train-clean-images/ -scales 16

mkdir -p BST/processed/task1/train-noisy-images
python create_DeFlow_train_dataset.py -source BST/task1/Input/ -target BST/processed/task1/train-noisy-images/ -scales 1 4 16

mkdir -p BST/processed/task1/val-gt-clean
python create_DeFlow_train_dataset.py -source BST/task1/Ground\ Truth/ -target BST/processed/task1/val-gt-clean/ -scales 2 8

mkdir -p BST/processed/task1/val-input-noisy
python create_DeFlow_train_dataset.py -source BST/task1/Input/ -target BST/processed/task1/val-input-noisy/ -scales 1 4

# From codes/
python compute_dataset_statistics.py ../datasets/AIM-RWSR/train-clean-images/4x/
python compute_dataset_statistics.py ../datasets/AIM-RWSR/train-clean-images/16x/

python compute_dataset_statistics.py ../datasets/BST/processed/task1/train-noisy-images/1x/
python compute_dataset_statistics.py ../datasets/BST/processed/task1/train-noisy-images/4x/

# From codes/
python train.py -opt confs/1_bst_task1.yml
python train.py -opt confs/2_bst_task1.yml