from PIL import Image
import gzip
import os
#import numpy as np
import cupy as cp
import warnings
#import cv2


from src.front_prop import front_propagation
from src.back_prop import back_propagation






if __name__ == "__main__":
    f = gzip.open("training_data/train-images-idx3-ubyte.gz", "r")

    image_size = 28
    num_images = 60000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    training_data = cp.frombuffer(buf, dtype=cp.uint8).astype(cp.float32)
    training_data = training_data.reshape(num_images, image_size, image_size, 1)
    training_data = training_data / 255.0

    f = gzip.open("training_data/train-labels-idx1-ubyte.gz", "r")

    f.read(8)
    buf = f.read(num_images)
    training_labels = cp.frombuffer(buf, dtype=cp.uint8)

    permutation = cp.random.permutation(len(training_data))

    training_data = training_data[permutation, :, :]
    training_labels = training_labels[permutation]



    f = gzip.open("t10k-images-idx3-ubyte.gz", "r")

    image_size = 28
    num_images = 10000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    validation_data = cp.frombuffer(buf, dtype=cp.uint8).astype(cp.float32)
    validation_data = validation_data.reshape(num_images, image_size, image_size, 1)
    validation_data = validation_data / 255.0

    f = gzip.open("t10k-labels-idx1-ubyte.gz", "r")

    f.read(8)
    buf = f.read(num_images)
    validation_labels = cp.frombuffer(buf, dtype=cp.uint8)

    permutation = cp.random.permutation(len(validation_data))
    
    validation_data = validation_data[permutation, :, :]
    validation_labels = validation_labels[permutation]


    def kaiming_uniform_init(shape, fan_in):
        limit = cp.sqrt(6.0 / fan_in)
        weights = cp.random.uniform(low=-limit, high=limit, size=shape)
        return weights

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

            


    """
    image = np.asarray(data[1]).squeeze()
    flattened_img = front_propagation.filter_image(image)
    flattened_img.show()
    """

    # """
    front = front_propagation
    back = back_propagation

    training = -1

    while training == -1:
        training_start = input("Run trainning for model? (Y/n) \n")

        if training_start == "Y" or training_start == "y" or training_start == "Yes" or training_start == "yes":

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
                    load_settings = 0

            print("Running...")

            for epoch in range(epochs):
                total_loss = 0
                num_iterations = 0

                for i in range(0, len(training_data), batch_size):
                    batch_start = i
                    batch_end = min(batch_start + batch_size, len(training_data))
                    batch_size_actual = batch_end - batch_start
                    batch_labels = training_labels[i:batch_end]

                    batch_images = training_data[i:batch_end].squeeze(-1)


                    image = batch_images

                    # out_image = Image.fromarray((image[0, :, :] * 255).astype(cp.uint8).get())
                    # out_image.show()

                    batch_conv = front.filter_image(image, conv_filter, "L")

                    # out_image = Image.fromarray((batch_conv[0, :, :] * 255).astype(cp.uint8).get())
                    # out_image.show()

                    batch_out_RELU_1 = front.leaky_RELU(batch_conv, 0.001)

                    batch_pooled = front.max_pooling(batch_out_RELU_1, "L")

                    batch_flat = batch_pooled.reshape(batch_size_actual, -1)

                    batch_RELU_2 = front.leaky_RELU(batch_flat, 0.01)

                    hidden_out_dl1 = front.forward_pass(batch_RELU_2, dense_layer_1, bias_hidden)

                    hidden_out_RELU_3 = front.leaky_RELU(hidden_out_dl1, 0.001)

                    logits = front.forward_pass(hidden_out_RELU_3, dense_layer_2, bias_out)


                    
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

                avg_loss = total_loss / len(training_data)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

                cp.save("Saves/trained_dense_2.npy", dense_layer_2)
                cp.save("Saves/trained_dense_1.npy", dense_layer_1)
                cp.save("Saves/trained_bias_2.npy", bias_out)
                cp.save("Saves/trained_bias_1.npy", bias_hidden)
                cp.save("Saves/trained_filter.npy", conv_filter)

            training = 1

        else:
            training = 0

    validation = ""
    validation_start = -1

    while validation_start == -1:
        validation = input("Run validaiton test for model? (Y/n) \n")

        if validation == "Y" or validation == "y" or validation == "Yes" or validation == "yes":

            settings = ""
            load_settings = -1
            while load_settings == -1:
                settings = input("Use default settings? (Y/n) \n")

                if settings == "y" or settings == "Y" or settings == "Yes" or settings == "yes":
                    batch_size = 512
                    load_settings = 1
                elif settings == "n" or settings == "N" or settings == "No" or settings == "no":
                    batch_size =        int(input("Batch size (int) =         "))
                    load_settings = 0

            print("Running...")

            total_loss = 0

            for i in range(0, len(validation_data), batch_size):
                batch_start = i
                batch_end = min(batch_start + batch_size, len(validation_data))
                batch_size_actual = batch_end - batch_start
                batch_labels = validation_labels[i:batch_end]

                batch_images = validation_data[i:batch_end].squeeze(-1)


                image = batch_images

                batch_conv = front.filter_image(image, conv_filter, "L")

                batch_out_RELU_1 = front.leaky_RELU(batch_conv, 0.001)

                batch_pooled = front.max_pooling(batch_out_RELU_1, "L")

                batch_flat = batch_pooled.reshape(batch_size_actual, -1)

                batch_RELU_2 = front.leaky_RELU(batch_flat, 0.01)

                hidden_out_dl1 = front.forward_pass(batch_RELU_2, dense_layer_1, bias_hidden)

                hidden_out_RELU_3 = front.leaky_RELU(hidden_out_dl1, 0.001)

                logits = front.forward_pass(hidden_out_RELU_3, dense_layer_2, bias_out)


                
                exp_logits = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))

                batch_preds = exp_logits / cp.sum(exp_logits, axis=1, keepdims = True)


                batch_loss_all = back.cross_entropy_loss(batch_preds, batch_labels)
                batch_loss = cp.mean(batch_loss_all)

                total_loss += batch_loss * batch_size_actual

            avg_loss = total_loss / len(validation_data)
            print(f"Total loss: {avg_loss:.4f}")

            validation_start = 1

        else:
            validation_start = 0


    




