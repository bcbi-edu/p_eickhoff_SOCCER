# To encode the commentary
# python encode_dataset.py --subset "train" --save_base ./features --dataset_base ../../data
# python encode_dataset.py --subset "val" --save_base ./features --dataset_base ../../data
# python encode_dataset.py --subset "test" --save_base ./features --dataset_base ../../data

# For model training and inference
python train.py \
        --run_inference \
        --dataset_path  ./features \
        --log_save_path  ./log \
        --model_save_path  ./models/multi-class/all.pth \
        --result_save_path  ./results/all/

# To evaluate the test set
python evaluation.py --subset "test" 
