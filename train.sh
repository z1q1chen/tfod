#!/bin/sh
# PIPELINE_FILE='research/deploy/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8/pipeline_file.config'
# NUM_STEPS=40000
# NUM_EVAL_STEPS=500
# MODEL_DIR='training'
# python research/object_detection/model_main_tf2.py \
#     --pipeline_config_path=$pipeline_file \
#     --model_dir=$MODEL_DIR \
#     --alsologtostderr \
#     --num_train_steps=$NUM_STEPS \
#     --sample_1_of_n_eval_examples=1 \
#     --num_eval_steps=$NUM_EVAL_STEPS


python research/object_detection/model_main_tf2.py --pipeline_config_path=pipeline_file.config --model_dir=training --alsologtostderr --num_train_steps=40000 --sample_1_of_n_eval_examples=1 --num_eval_steps=500
python research/object_detection/model_main_tf2.py --pipeline_config_path=pipeline_file.config --model_dir=training --checkpoint_dir=training 

MODEL_NAME='ssd_mobilenet_v2_320x320_coco17_tpu-8'
python research/object_detection/model_main_tf2.py --pipeline_config_path=research/deploy/$MODEL_NAME/pipeline_file.config --model_dir=training_$MODEL_NAME --alsologtostderr --num_train_steps=40000 --sample_1_of_n_eval_examples=1 --num_eval_steps=500