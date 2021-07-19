#!/bin/sh
conda install protobuf -y
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
