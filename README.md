# conv_neural_net
A simple conv_neural_net.

A little capstone project I made. Convolutional neural networks are cool, but a pain to build from scratch. 

The current structure is the following

- Image
- Convolutional layer
- Leaky RELU
- Max Pooling
- Flatten
- Leaky RELU
- Hidden layer
- Leaky RELU
- Hidden layer
- Softmax

And Tada! You get a result! 

The structure will be reworked for better learning. It is not that optimized, so it might end up resource hogging.

## Interactive training demo

Run the Flask app from the project root with `python app.py`, then open
`http://127.0.0.1:5000/demo`. The demo can train either model version on a
random subset of the bundled MNIST training data. Choose the sample count,
validation sample count, epochs, batch size, and learning rate; the page reports
training and validation loss, validation accuracy, and shows an input image
alongside its convolution, activation, and pooling feature maps. Validation
uses a separate subset of the bundled MNIST test data.

The demo uses the same CuPy-based model operations as the training scripts and
requires a working CuPy/CUDA setup. It does not overwrite the saved model
weights. Training runs are limited to 2,048 images; V1 supports up to 200
epochs, while V2 remains limited to 20 epochs so runs stay interactive.
