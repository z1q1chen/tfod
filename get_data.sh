conda install protobuf -y
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
pip install imageio
pip install wget
mkdir data
mkdir research/deploy
cd data
curl -L "https://app.roboflow.com/ds/no8u8m3Ifw?key=itfwqk4p5f" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip