from PIL import Image
import gzip
import os
#import numpy as np
import cupy as cp
import warnings
#import cv2


from src.front_prop_old import front_propagation
from src.back_prop_old import back_propagation

if __name__ == "__main__":
    f = gzip.open("training_data/train-images-idx3-ubyte.gz", "r")

    image_size = 28
    num_images = 60000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = cp.frombuffer(buf, dtype=cp.uint8).astype(cp.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    data = data / 255.0

    f = gzip.open("training_data/train-labels-idx1-ubyte.gz", "r")

    f.read(8)
    buf = f.read(num_images)
    labels = cp.frombuffer(buf, dtype=cp.uint8)


    def kaiming_uniform_init(shape, fan_in):
        limit = cp.sqrt(6.0 / fan_in)
        weights = cp.random.uniform(low=-limit, high=limit, size=shape)
        return weights

    # epochs = 10
    # batch_size = 32


    load = ""
    load_bool = -1

    while load_bool == -1:
        load = input("Load from save? (Y/n) \n")
        if load == "y" or load == "Y" or load == "Yes" or load == "yes":
            dense_layer_2 = cp.load("Saves/trained_dense_2.npy")
            dense_layer_1 = cp.load("Saves/trained_dense_1.npy")
            bias_out = cp.load("Saves/trained_bias_2.npy")
            bias_hidden = cp.load("Saves/trained_bias_1.npy")
            conv_filter = cp.load("Saves/trained_filter.npy")
            load_bool = 1
        elif load == "n" or load == "N" or load == "No" or load == "no":
            dense_layer_2 = kaiming_uniform_init((128, 10), 128)
            dense_layer_1 = kaiming_uniform_init((169, 128), 169)
            conv_filter = kaiming_uniform_init((3, 3), 9)
            bias_hidden = kaiming_uniform_init((1, 128), 1)
            bias_out = kaiming_uniform_init((1, 10), 1)
            load_bool = 1

    batch_start = 0




    settings = ""
    load_settings = -1

    while load_settings == -1:
        settings = input("Use default settings? (Y/n) \n")

        if settings == "y" or settings == "Y" or settings == "Yes" or settings == "yes":
            learning_rate = 0.05
            epochs = 20
            batch_size = 512
            load_settings = 1
        elif settings == "n" or settings == "N" or settings == "No" or settings == "no":
            learning_rate =   float(input("Learning rate (float) =    "))
            epochs =            int(input("Epochs (int) =             "))
            batch_size =        int(input("Batch size (int) =         "))
            load_settings = 1
            


    """
    image = np.asarray(data[1]).squeeze()
    flattened_img = front_propagation.filter_image(image)
    flattened_img.show()
    """

    # """
    front = front_propagation
    back = back_propagation

    for epoch in range(epochs):
        total_loss = 0
        num_iterations = 0

        for i in range(0, len(data), batch_size):
            batch_start = i
            batch_end = min(batch_start + batch_size, len(data))
            batch_size_actual = batch_end - batch_start
            batch_labels = labels[i:batch_end]
            # batch_images = cp.asarray(
            #     [cp.asarray(data[j]).squeeze() for j in range(i, batch_end)]
            # )
            batch_images = data[i:batch_end].squeeze(-1)
            flattened_img_arr = []
            # batch_flat = cp.array(
            #     [batch_images[k].flatten() for k in range(batch_size_actual)]
            # )  # Shape: (batch_size, 784)

            #for j in range(batch_start, batch_end):
                #image = cp.asarray(data[j]).squeeze() * 255
                #image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_CUBIC)

                # image = np.clip(image, 0, 255).astype(np.uint8)
                # image = Image.fromarray(image)
                # image = image.resize((96, 96), Image.Resampling.BICUBIC)
                # image = image.convert("L")

            image = batch_images

            batch_conv = front.filter_image(image, conv_filter, "L")
            # image1 = Image.fromarray(image)       Debugging
            # image1.show()                         Debugging

            batch_out_RELU_1 = front.leaky_RELU(batch_conv, 0.001)
            # image1 = Image.fromarray(image)       Debugging
            # image1.show()                         Debugging

            batch_pooled = front.max_pooling(batch_out_RELU_1, "L")
            # image1 = Image.fromarray(image)       Debugging
            # image1.show()                         Debuggings
            # array = cp.array(image).flatten()
            # flattened_img_arr.append(array.ravel())

            batch_flat = batch_pooled.reshape(batch_size_actual, -1)

            batch_RELU_2 = front.leaky_RELU(batch_flat, 0.01)

            hidden_out_dl1 = front.forward_pass(batch_RELU_2, dense_layer_1, bias_hidden)

            hidden_out_RELU_3 = front.leaky_RELU(hidden_out_dl1, 0.001)

            logits = front.forward_pass(hidden_out_RELU_3, dense_layer_2, bias_out)

            # Batch softmax
            # batch_preds = cp.array(
            #     [back.softmax_function(logits[k]) for k in range(batch_size_actual)]
            # )  # (batch_size, 10)
            
            exp_logits = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))

            batch_preds = exp_logits / cp.sum(exp_logits, axis=1, keepdims = True)


            # Batch one-hot labels
            # batch_one_hot = cp.array(
            #     [
            #         back.get_one_hot(int(batch_labels[k]), 10)
            #         for k in range(batch_size_actual)
            #     ]
            # )  # (batch_size, 10)

            batch_one_hot = cp.eye(10)[batch_labels]

            # Batch error
            batch_error = batch_preds - batch_one_hot  # (batch_size, 10)

            # Backward pass on batch
            grad_W_out, grad_b_out, _ = back.backpropagate(
                dense_layer_2, hidden_out_RELU_3, batch_error, batch_size_actual
            )

            RELU_3_error = (batch_error @ dense_layer_2.T) * back.deriv_leaky_RELU(hidden_out_dl1, 0.001)
            
            grad_W_hidden, grad_b_hidden, _ = back.backpropagate(
                dense_layer_1, batch_RELU_2, RELU_3_error, batch_size_actual
            )

            RELU_2_error = (RELU_3_error @ dense_layer_1.T).reshape(batch_size_actual, 13, 13) * back.deriv_leaky_RELU(batch_pooled, 0.001)

            max_pooling_error = cp.repeat(cp.repeat(RELU_2_error, 2, axis=1), 2, axis=2) * back.deriv_max_pooling(batch_out_RELU_1)

            RELU_1_error = max_pooling_error * back.deriv_leaky_RELU(batch_conv, 0.001)

            # input_error = back.deriv_conv_layer(RELU_1_error, conv_filter)

            grad_conv_filter, conv_filter_error = back.deriv_conv_layer(batch_images, conv_filter, batch_conv, RELU_1_error)



            # Update the weight matrix
            dense_layer_2, bias_out = back.learning(
                grad_W_out, grad_b_out, dense_layer_2, bias_out, learning_rate
            )
            dense_layer_1, bias_hidden = back.learning(
                grad_W_hidden, grad_b_hidden, dense_layer_1, bias_hidden, learning_rate
            )

            # Update the convolutional filter
            placeholder = cp.zeros((0))
            conv_filter, placeholder = back.learning(
                grad_conv_filter, placeholder, conv_filter, placeholder, learning_rate
            )


            batch_loss_all = back.cross_entropy_loss(batch_preds, batch_labels)
            batch_loss = cp.mean(batch_loss_all)

            total_loss += batch_loss * batch_size_actual

        avg_loss = total_loss / len(data)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        cp.save("Saves/trained_dense_2.npy", dense_layer_2)
        cp.save("Saves/trained_dense_1.npy", dense_layer_1)
        cp.save("Saves/trained_bias_2.npy", bias_out)
        cp.save("Saves/trained_bias_1.npy", bias_hidden)
        cp.save("Saves/trained_filter.npy", conv_filter)