from PIL import Image
import gzip
import os
import json
#import numpy as np
import cupy as cp
import warnings
import sys
#import cv2


from src.front_prop import front_propagation
from src.back_prop import back_propagation
from src.adam_optimizer import AdamOptimizer


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

    # load = ""
    # load_bool = -1

    filter_batch = 16

    # while load_bool == -1:
    #     load = input("Load from save? (Y/n) \n")


    load = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
    training_start = sys.argv[2] if len(sys.argv) > 2 else "Unknown"
    settings_list = json.loads(sys.argv[3])
    validation = sys.argv[4]



    if load == "y" or load == "Y" or load == "Yes" or load == "yes":
        dense_layer_2 = cp.load("Saves/trained_dense_2.npy")
        dense_layer_1 = cp.load("Saves/trained_dense_1.npy")
        bias_out = cp.load("Saves/trained_bias_2.npy")
        bias_hidden = cp.load("Saves/trained_bias_1.npy")
        conv_filter = cp.load("Saves/trained_filter.npy")
        load_bool = 1
    elif load == "n" or load == "N" or load == "No" or load == "no":
        dense_layer_2 = kaiming_uniform_init((128, 10), 128)
        dense_layer_1 = kaiming_uniform_init((169 * filter_batch, 128), 169)
        conv_filter = kaiming_uniform_init((filter_batch, 3, 3), 9)
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


    # training = -1

    # while training == -1:
    #     training_start = input("Run training for model? (Y/n) \n")

    if training_start == "Y" or training_start == "y" or training_start == "Yes" or training_start == "yes":

        # settings = ""
        # load_settings = -1

            # while load_settings == -1:
            #     settings = input("Use default settings? (Y/n) \n")

        for i in range(len(settings_list)):
            learning_rate = settings_list[0]
            epochs = settings_list[1]
            batch_size = settings_list[2]

        # if settings == "y" or settings == "Y" or settings == "Yes" or settings == "yes":
        #     learning_rate = 0.001
        #     epochs = 1
        #     batch_size = 64
        #     load_settings = 1
        # elif settings == "n" or settings == "N" or settings == "No" or settings == "no":
        #     learning_rate =   float(input("Learning rate (float) =    "))
        #     epochs =            int(input("Epochs (int) =             "))
        #     batch_size =        int(input("Batch size (int) =         "))
        #     load_settings = 0

        opt = AdamOptimizer(lr = learning_rate)
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

                RELU_2_error = (RELU_3_error @ dense_layer_1.T).reshape(batch_size_actual, filter_batch, 13, 13) * back.deriv_leaky_RELU(batch_pooled, 0.01)


                max_pooling_error = cp.repeat(cp.repeat(RELU_2_error, 2, axis=2), 2, axis=3) * back.deriv_max_pooling(batch_out_RELU_1)

                RELU_1_error = max_pooling_error * back.deriv_leaky_RELU(batch_conv, 0.001)

                _, grad_conv_filter, _ = back.conv_backward_im2col(
                    RELU_1_error,
                    (batch_images, conv_filter, None, {"stride": 1, "pad": 0}),
                )
                grad_conv_filter = grad_conv_filter / batch_size_actual

                grad_W_out = cp.clip(grad_W_out, -1.0, 1.0)
                grad_W_hidden = cp.clip(grad_W_hidden, -1.0, 1.0)
                grad_conv_filter = cp.clip(grad_conv_filter, -1.0, 1.0)

                dense_layer_2 = opt.update('W2', dense_layer_2, grad_W_out)
                bias_out = opt.update('b2', bias_out, grad_b_out)

                dense_layer_1 = opt.update('W1', dense_layer_1, grad_W_hidden)
                bias_hidden = opt.update('b1', bias_hidden, grad_b_hidden)
                conv_filter = opt.update('conv_filter', conv_filter, grad_conv_filter)

                # Update the weight matrix
                # dense_layer_2, bias_out = back.learning(
                #     grad_W_out, grad_b_out, dense_layer_2, bias_out, learning_rate
                # )
                # dense_layer_1, bias_hidden = back.learning(
                #     grad_W_hidden, grad_b_hidden, dense_layer_1, bias_hidden, learning_rate
                # )

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

    # validation = ""
    # validation_start = -1

    # while validation_start == -1:

        if validation == "Y" or validation == "y" or validation == "Yes" or validation == "yes":

            # settings = ""
            # load_settings = -1
            # while load_settings == -1:
            #     settings = input("Use default settings? (Y/n) \n")

            #     if settings == "y" or settings == "Y" or settings == "Yes" or settings == "yes":
            #         batch_size = 64
            #         load_settings = 1
            #     elif settings == "n" or settings == "N" or settings == "No" or settings == "no":
            #         batch_size =        int(input("Batch size (int) =         "))
            #         load_settings = 0

            batch_size = settings_list[2]

            print("Running validation...")

            total_loss = 0
            total_correct = 0

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

                predictions = cp.argmax(batch_preds, axis=1)
                correct_predictions = cp.sum(predictions == batch_labels)
                total_correct += int(correct_predictions.get())


                batch_loss_all = back.cross_entropy_loss(batch_preds, batch_labels)
                batch_loss = cp.mean(batch_loss_all)

                total_loss += batch_loss * batch_size_actual

            avg_loss = total_loss / len(validation_data)
            print(f"Total loss: {avg_loss:.4f}")

            final_accuracy = (total_correct / len(validation_data)) * 100
            print(f"True Validation Accuracy: {final_accuracy:.2f}%")

            validation_start = 1

        else:
            validation_start = 0
