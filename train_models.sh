#!/bin/bash
# Define a string variable with a value
Models="SSD-MobileNet-V2-320 SSD-MobileNet-V1-FPN-640 SSD-MobileNet-V2-FPNlite-320 SSD-MobileNet-V2-FPNlite-640 SSD-ResNet50-V1-FPN-640 SSD-ResNet50-V1-FPN-1024 SSD-ResNet101-V1-FPN-640 SSD-ResNet101-V1-FPN-1024 SSD-ResNet152-V1-FPN-640 SSD-ResNet152-V1-FPN-1024"

# Iterate the string variable using for loop
for model in $Models; do
    echo Training $model
    python research/object_detection/model_main_tf2.py --pipeline_config_path="${model}_pipeline_file.config" --model_dir=trained_models/training_$model --alsologtostderr --num_train_steps=4000 --sample_1_of_n_eval_examples=1 --num_eval_steps=100
done