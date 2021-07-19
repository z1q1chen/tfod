#!/bin/sh
conda install protobuf -y
pip install absl-py
cd research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
mkdir deploy
python -m pip install .
cd ..
pip install imageio
pip install wget
mkdir data
mkdir training
cd data
curl -L "https://app.roboflow.com/ds/no8u8m3Ifw?key=itfwqk4p5f" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

python train_setup.py

python research/object_detection/model_main_tf2.py --pipeline_config_path=pipeline_file.config --model_dir=training --alsologtostderr --num_train_steps=40000 --sample_1_of_n_eval_examples=1 --num_eval_steps=500
