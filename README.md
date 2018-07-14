# keras_examples
Simple Keras examples of deep learning inspired by the excellent book "Deep Learning with Python" from Francois Chollet, father of Keras.

All scripts are self-containing in the sense that it uses only data available through Keras. It will need a working installation of Keras package together with one of the supported back-ends (e.g. Tensorflow or Theano) and, for convolutional neural networks, a NVIDIA GPU (e.g. GTX 1080 Titan).

The scripts are all located in the "py" folder:

- binary_simple.py: Shows how to create a binary classifier
- multiclass_simple.py: Similar for a multiclass response
- regression_simple.py: Same for one numeric output aka regression
- mnist_simple.py: Simple (non-convolutional) net to classify hand-written digits 
- mnist_conv.py: Similar than above but now with convolutional layers
- mnist_conv_augmentation.py: Similar than above but now with image data augmentation through "flow"
- dog_detector.py: Similar than above but for a 120 class kaggle competition with dog breeds
