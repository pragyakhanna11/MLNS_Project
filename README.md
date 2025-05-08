# MLNS_Project

## Data Preparation
``python scripts/prepare_data.py --input data/trajectories_x.txt --output-dir prepared_data --normalize standard --filter --window-size 20 --stride 5``
## Base Model Training
``python scripts/train_base.py -i prepared_data\processed_trajectories.npy --output-dir results/base_model --epochs 500 --batch-size 32``
