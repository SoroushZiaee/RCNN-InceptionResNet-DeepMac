# RCNN-InceptionResNet-DeepMac

Suppose we are developing a new labeling tool to annotate masks in a video. Labeling all the frames of a video with great accuracy takes a lot of time and cost. In order to make the annotation process faster, we need to use semi-automated or automated labeling methods.

In this repository, I have tried to make a pipeline using the mask_rcnn_inception_resnet_v2 combined with DEEPMAC pretrained model to detect and mask vehicles on the highway.

In this project we need to use tensorflow version 2.2.0 so, first downgrade our tensorflow to specified version with the following command

```
pip install -U --pre -q tensorflow=="2.2.0"
```

Next install the following requirements: python

```

pip install -q tensorflow-object-detection-api
pip install -q imageio-ffmpeg
```

Afterward, We need to clone the tensorflow model for our furture analysis with following scripts:

```

cd models/research/
protoc object_detection/protos/\*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

### Results

- main video

https://user-images.githubusercontent.com/32066123/196052631-19c956d6-3707-4e2b-b65e-e2aaea23b902.mp4

- output video

https://user-images.githubusercontent.com/32066123/196052638-cb479153-ea19-4395-89ec-8c97f0189df9.mp4

### References

- [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf)
- [The surprising impact of mask-head architecture on novel class segmentation](https://arxiv.org/pdf/2104.00613.pdf)

### Open source repositories on this grand

- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
