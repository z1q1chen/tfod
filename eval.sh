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
    echo Training $model
    python research/object_detection/model_main_tf2.py \
        --pipeline_config_path="models/${model}/pipeline.config" \
        --model_dir="models/${model}/model" \
        --checkpoint_dir="models/${model}/checkpoint" \
        --alsologtostderr
done