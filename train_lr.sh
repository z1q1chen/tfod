#!/bin/bash
# Define a string variable with a value
models="SSD-MobileNet-V2-320 \
        SSD-MobileNet-V1-FPN-640 \
        SSD-MobileNet-V2-FPNlite-320 \
        SSD-MobileNet-V2-FPNlite-640 \
        SSD-ResNet50-V1-FPN-640 \
        SSD-ResNet50-V1-FPN-1024 \
        SSD-ResNet101-V1-FPN-640 \
        SSD-ResNet101-V1-FPN-1024 \
        SSD-ResNet152-V1-FPN-640 \
        SSD-ResNet152-V1-FPN-1024"

# Iterate the string variable using for loop
for model in $models; do
    for lr in 0.04 0.004 0.08 0.008 0.012
    echo Training $model
    python research/object_detection/model_main_tf2.py \
        --pipeline_config_path="models/${model}/pipeline_lr_${lr}.config" \
        --model_dir="models/${model}/model_lr_${lr}" \
        --num_train_steps=1000 \
        --sample_1_of_n_eval_examples=1 \
        --num_eval_steps=100 \
        --alsologtostderr
done