#!/bin/bash
Models="SSD-MobileNet-V2-320 SSD-MobileNet-V1-FPN-640 SSD-MobileNet-V2-FPNlite-320 SSD-MobileNet-V2-FPNlite-640 SSD-ResNet50-V1-FPN-640 SSD-ResNet50-V1-FPN-1024 SSD-ResNet101-V1-FPN-640 SSD-ResNet101-V1-FPN-1024 SSD-ResNet152-V1-FPN-640 SSD-ResNet152-V1-FPN-1024"

# Iterate the string variable using for loop
for model in $Models; do
    echo Evaluating $model
    python research/deeplab/eval.py --pipeline_config_path="${model}_pipeline_file.config" --checkpoint_dir="trained_models/training_${model}" --eval_dir="eval/eval_${model}" --logtostderr    
done