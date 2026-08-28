from PIL import Image
import gzip

import numpy as np
import cupy as cp
import warnings
import multiprocessing

if __name__ == "__main__":
    """
    image_path = "images-of-a-crab-11.jpg"
    image_path = "number5.png"
    img = Image.open(image_path)
    img = img.resize((129, 129), Image.Resampling.BICUBIC)
    rgb_img = img.convert("RGB")

    # rgb_img.show()

    """
    # f = gzip.open("FCT/Python/train-images-idx3-ubyte.gz", "r")
    f = gzip.open("train-images-idx3-ubyte.gz", "r")

    image_size = 28
    num_images = 60000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = cp.frombuffer(buf, dtype=cp.uint8).astype(cp.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    data = data / 255.0  # Normalize to [0, 1]

    f = gzip.open("train-labels-idx1-ubyte.gz", "r")
    f.read(8)  # Skip the 8-byte header (magic number + num_items)
    buf = f.read(num_images)  # Read all labels (one byte per label)

    labels = cp.frombuffer(buf, dtype=cp.uint8)  # MNIST labels are 0-9, use uint8

    indices = cp.arange(data.shape[0])
    cp.random.shuffle(indices)

    data = data[indices]
    labels = labels[indices]

    learning_rate = 0.0001

    # weight_file = "conv_neural_net/trained_weights.txt"

    # image = np.asarray(data[1]).squeeze()

    # # image2 = Image.frombuffer("RGB", (image_size, image_size), data[2])
    # rgb_img = Image.fromarray(image)
    # rgb_img = rgb_img.resize((129, 129), Image.Resampling.BICUBIC)
    # rgb_img = rgb_img.convert("L")
    # rgb_img.show()

    def kaiming_uniform_init(shape, fan_in):
        limit = cp.sqrt(6.0 / fan_in)
        weights = cp.random.uniform(low=-limit, high=limit, size=shape)
        return weights

    def filter_img(image):
        if image.mode == "RGB":
            r_band, g_band, b_band = image.split()

            r_matrix = cp.array(r_band)
            g_matrix = cp.array(g_band)
            b_matrix = cp.array(b_band)

            # red_img = Image.fromarray(g_matrix)

            # red_img.show()

            # Add values for height and width

            width, height = image.size

            """
            for y in range(height):
                for x in range(width):
                    print(r_matrix[x, y])
            """

            # filters

            # Laplacian filter

            k_height_max = height / 3
            k_width_max = width / 3

            """
            for y in range(int(k_height_max)):
                for x in range(int(k_width_max)):
                    print(r_matrix[x * 3, y * 3])
            """

            # laplacian_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            sobel_h_filter = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_v_filter = cp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

            # filtered_matrix = np.zeros((int(k_height_max), int(k_width_max)))

            c_r_matrix = cp.empty((int(k_height_max), int(k_width_max)))
            c_g_matrix = cp.empty((int(k_height_max), int(k_width_max)))
            c_b_matrix = cp.empty((int(k_height_max), int(k_width_max)))

            # c_matrix = np.array([])

            # old filter program
            """
            for y in range(0, int(k_height_max), 1):
                c_r_row = np.array([])
                c_g_row = np.array([])
                c_b_row = np.array([])
            
                for x in range(0, int(k_width_max), 1):
                    r_region = r_matrix[x * 3 : x * 3 + 3, y * 3 : y * 3 + 3]
                    g_region = g_matrix[x * 3 : x * 3 + 3, y * 3 : y * 3 + 3]
                    b_region = b_matrix[x * 3 : x * 3 + 3, y * 3 : y * 3 + 3]
            
                    r_c_matrix = np.clip(np.multiply(r_region, sobel_h_filter), 0, 255).astype(
                        np.uint8
                    )
        
                    r_c_matrix = np.clip(np.multiply(r_matrix, sobel_v_filter), 0, 255).astype(
                        np.uint8
                    )
            
                    g_c_matrix = np.clip(np.multiply(g_region, sobel_h_filter), 0, 255).astype(
                        np.uint8
                    )
        
                    g_c_matrix = np.clip(np.multiply(g_matrix, sobel_v_filter), 0, 255).astype(
                        np.uint8
                    )
        
                    b_c_matrix = np.clip(np.multiply(b_region, sobel_h_filter), 0, 255).astype(
                        np.uint8
                    )
                
                    b_c_matrix = np.clip(np.multiply(b_matrix, sobel_v_filter), 0, 255).astype(
                        np.uint8
                    )
            
                    if c_r_row.size == 0 and c_g_row.size == 0 and c_b_row.size == 0:
                        c_r_row = r_c_matrix
                        c_g_row = g_c_matrix
                        c_b_row = b_c_matrix
        
                    else:
                        c_r_row = np.hstack((c_r_row, r_c_matrix))
                        c_g_row = np.hstack((c_g_row, g_c_matrix))
                        c_b_row = np.hstack((c_b_row, b_c_matrix))
                
                if c_r_matrix.size == 0:
                    c_r_matrix = c_r_row
                    c_g_matrix = c_g_row
                    c_b_matrix = c_b_row
        
                else:
                    c_r_matrix = np.vstack((c_r_row, c_r_matrix))
                    c_g_matrix = np.vstack((c_g_row, c_g_matrix))
                    c_b_matrix = np.vstack((c_b_row, c_b_matrix))
        
        
            c_img = Image.merge(
                "RGB",
                (
                    Image.fromarray(c_r_matrix),
                    Image.fromarray(c_g_matrix),
                    Image.fromarray(c_b_matrix),
                ),
            )
            c_img.show()
            """

            # filter the image

            for y in range(0, int(k_height_max)):
                for x in range(0, int(k_width_max)):
                    r_region = r_matrix[x * 3 : x * 3 + 3, y * 3 : y * 3 + 3]
                    g_region = g_matrix[x * 3 : x * 3 + 3, y * 3 : y * 3 + 3]
                    b_region = b_matrix[x * 3 : x * 3 + 3, y * 3 : y * 3 + 3]

                    r_h_matrix = cp.sum(r_region * sobel_h_filter)
                    r_v_matrix = cp.sum(r_region * sobel_v_filter)

                    g_h_matrix = cp.sum(g_region * sobel_h_filter)
                    g_v_matrix = cp.sum(g_region * sobel_v_filter)

                    b_h_matrix = cp.sum(b_region * sobel_h_filter)
                    b_v_matrix = cp.sum(b_region * sobel_v_filter)

                    r_magnitude = cp.sqrt(r_h_matrix**2 + r_v_matrix**2)
                    g_magnitude = cp.sqrt(g_h_matrix**2 + g_v_matrix**2)
                    b_magnitude = cp.sqrt(b_h_matrix**2 + b_v_matrix**2)

                    c_r_matrix[y, x] = r_magnitude
                    c_g_matrix[y, x] = g_magnitude
                    c_b_matrix[y, x] = b_magnitude



            c_r_matrix = cp.clip(c_r_matrix, 0, 255).astype(cp.uint8)
            c_g_matrix = cp.clip(c_g_matrix, 0, 255).astype(cp.uint8)
            c_b_matrix = cp.clip(c_b_matrix, 0, 255).astype(cp.uint8)

            hor_r_row = cp.empty(0)
            hor_g_row = cp.empty(0)
            hor_b_row = cp.empty(0)

            for i in range(k_height_max):
                if hor_r_row * hor_g_row * hor_b_row == cp.empty(0):
                    hor_r_row = r_matrix[:, 0 : 3]
                    hor_g_row = g_matrix[:,0 : 3]
                    hor_b_row = b_matrix[:,0 : 3]
                
                else: 
                    hor_r_row = cp.hstack(hor_r_row, r_matrix[:, i : i + 3])
                    hor_g_row = cp.hstack(hor_r_row, r_matrix[:, i : i + 3])
                    hor_b_row = cp.hstack(hor_r_row, r_matrix[:, i : i + 3])

            hor_c_r_row = hor_r_row * filter
            hor_c_g_row = hor_g_row * filter
            hor_c_b_row = hor_b_row * filter

            for i in range(hor_c_r_row.shape[1] / width);
                c_r_matrix = cp.vstack(hor)


            # c_img = Image.fromarray(c_r_matrix)
            # c_img.show()

            c_img = Image.merge(
                "RGB",
                (
                    Image.fromarray(c_r_matrix),
                    Image.fromarray(c_g_matrix),
                    Image.fromarray(c_b_matrix),
                ),
            )
            # c_img.show()

            return c_img

        elif image.mode == "L":
            img_matrix = cp.array(image)

            width, height = image.size

            k_height_max = height / 3
            k_width_max = height / 3

            sobel_h_filter = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_v_filter = cp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

            c_img_matrix = cp.empty((int(k_height_max), int(k_width_max)))

            for y in range(0, int(k_height_max)):
                for x in range(0, int(k_width_max)):
                    img_region = img_matrix[x * 3 : x * 3 + 3, y * 3 : y * 3 + 3]

                    img_h_matrix = cp.sum(img_region * sobel_h_filter)
                    img_v_matrix = cp.sum(img_region * sobel_v_filter)

                    img_magnitude = cp.sqrt(img_h_matrix**2 + img_v_matrix**2)

                    c_img_matrix[y, x] = img_magnitude

            c_img_matrix = cp.clip(c_img_matrix, 0, 255).astype(cp.uint8)
            c_img = Image.fromarray(c_img_matrix)
            c_img = c_img.convert("L")
            return c_img

    # convoluted_img = filter_img(rgb_img)
    # convoluted_img.show()

    # activation function

    def leaky_RELU_img(image, alpha):
        if image.mode == "RGB":
            r_band, g_band, b_band = image.split()

            r_matrix = cp.array(r_band)
            g_matrix = cp.array(g_band)
            b_matrix = cp.array(b_band)

            # width, height = image.size

            act_r_image = cp.where(r_matrix > 0, r_matrix, r_matrix * alpha)
            act_g_image = cp.where(g_matrix > 0, g_matrix, g_matrix * alpha)
            act_b_image = cp.where(b_matrix > 0, b_matrix, b_matrix * alpha)

            act_r_image = cp.clip(act_r_image, 0, 255).astype(cp.uint8)
            act_g_image = cp.clip(act_g_image, 0, 255).astype(cp.uint8)
            act_b_image = cp.clip(act_b_image, 0, 255).astype(cp.uint8)

            a_img = Image.merge(
                "RGB",
                (
                    Image.fromarray(act_r_image),
                    Image.fromarray(act_g_image),
                    Image.fromarray(act_b_image),
                ),
            )

            return a_img

        elif image.mode == "L":
            img_matrix = cp.array(image)

            # width, height = image.size

            act_image = cp.where(img_matrix > 0, img_matrix, img_matrix * alpha)

            act_image = cp.clip(act_image, 0, 255).astype(cp.uint8)

            act_img = Image.fromarray(act_image)

            return act_img

    def leaky_RELU(array, alpha):
        relud_array = cp.where(array > 0, array, array * alpha)

        return relud_array

    # activated_img = leaky_RELU_img(convoluted_img, 0.01)
    # activated_img.show()

    def max_pooling(image):
        if image.mode == "RGB":
            r_band, g_band, b_band = image.split()

            r_matrix = cp.array(r_band)
            g_matrix = cp.array(g_band)
            b_matrix = cp.array(b_band)

            width, height = image.size

            r_max_arr = cp.empty((int(width / 2), int(height / 2)))
            g_max_arr = cp.empty((int(width / 2), int(height / 2)))
            b_max_arr = cp.empty((int(width / 2), int(height / 2)))

            for y in range(0, int(height / 2)):
                for x in range(0, int(width / 2)):
                    r_region = r_matrix[x * 2 : x * 2 + 2, y * 2 : y * 2 + 2]
                    g_region = g_matrix[x * 2 : x * 2 + 2, y * 2 : y * 2 + 2]
                    b_region = b_matrix[x * 2 : x * 2 + 2, y * 2 : y * 2 + 2]

                    r_max = cp.max(r_region)
                    g_max = cp.max(g_region)
                    b_max = cp.max(b_region)

                    r_max_arr[y][x] = r_max
                    g_max_arr[y][x] = g_max
                    b_max_arr[y][x] = b_max

            r_max_arr = cp.clip(r_max_arr, 0, 255).astype(cp.uint8)
            g_max_arr = cp.clip(g_max_arr, 0, 255).astype(cp.uint8)
            b_max_arr = cp.clip(b_max_arr, 0, 255).astype(cp.uint8)

            p_img = Image.merge(
                "RGB",
                (
                    Image.fromarray(r_max_arr),
                    Image.fromarray(g_max_arr),
                    Image.fromarray(b_max_arr),
                ),
            )

            return p_img

        elif image.mode == "L":
            img_matrix = cp.array(image)

            width, height = image.size

            img_max_arr = cp.empty((int(width / 2), int(height / 2)))

            for y in range(0, int(height / 2)):
                for x in range(0, int(width / 2)):
                    img_region = img_matrix[y * 2 : y * 2 + 2, x * 2 : x * 2 + 2]

                    img_max = cp.max(img_region)

                    img_max_arr[y][x] = img_max

            img_max_arr = cp.clip(img_max_arr, 0, 255).astype(cp.uint8)

            img_max = Image.fromarray(img_max_arr)

            return img_max

    # max_pooled_img = max_pooling(activated_img)
    # max_pooled_img.show()

    # max_pooled_array = np.array(max_pooled_img)
    # flattened_array = max_pooled_array.flatten()
    # print(flattened_array.shape)

    # Fully connected layer
    #

    # softmax activation

    def forward_pass(array, weights):
        # weight_vec = kaiming_uniform_init((array.size, 10), array.size)
        bias = cp.zeros((1, weights.shape[1]))
        return cp.dot(array, weights) + bias

    """
    def softmax_function(array, epsilon=1e-15):
        array = array.astype(float)
        probabilities = np.empty((array.size), dtype=np.float64)
        for i in range(int(array.size)):
            max_element = np.max(array)
            numerator = np.e ** (array[0][i] - max_element)
            denominator = np.sum(np.exp(array - max_element))
            probability = numerator / denominator
            probabilities[i] = np.clip(probability, epsilon, 1.0 - epsilon)
        return probabilities
    """

    def deriv_Leaky_RELU(input, alpha):
        if input > 0:
            return 1

        if input < 0:
            return alpha

    def softmax_function(array, epsilon=1e-15):
        x = cp.asarray(array, dtype=cp.float64).reshape(-1)
        x = x - cp.max(x)
        exps = cp.exp(x)
        probs = exps / cp.sum(exps)
        return cp.clip(probs, epsilon, 1.0 - epsilon)

    def soft_to_label(array):
        max_index = cp.argmax(array)
        return max_index

    def get_one_hot(label_index, num_classes=10):
        one_hot = cp.zeros(num_classes)
        one_hot[label_index] = 1
        return one_hot

    def cross_entropy_loss(y_pred, y_true, epsilon=1e-15):
        y_pred = cp.clip(y_pred, epsilon, 1.0 - epsilon)
        return -cp.log(y_pred[cp.arange(len(y_true)), y_true.astype(cp.int32)])

    def deriv_CEL_and_softmax(ypredScores, ytrueScores, epsilon=1e-15):
        # gradient = np.subtract(ypredScores, ytrueScores)
        # mask = (ypredScores > epsilon) & (ypredScores < (1.0 - epsilon))
        # return np.array(gradient * mask)

        return cp.array(ypredScores - ytrueScores).reshape(-1)

    def deriv_leaky_RELU(input, alpha):
        if input >= 0:
            return 1
        if input < 0:
            return alpha

    def backpropagate(layer, prev_activtions, error_current):
        grad_W = prev_activtions.T @ error_current
        grad_b = cp.sum(error_current, axis=0)
        error_prev = error_current @ layer.T

        return grad_W, grad_b, error_prev

    def hidden_unit_errors(activ_derivs, W, next_errors):
        return cp.asarray(activ_derivs) * (W.dot(next_errors))

    def convolute_image(rgb_img):
        return max_pooling(leaky_RELU_img(filter_img(rgb_img), 0.01))

    def flatten_image(image):
        image = cp.array(image).flatten()
        image = cp.reshape(image, (1, image.shape[0]))  # Use actual flattened size
        return image

    def predict(array):
        return cp.array(
            softmax_function(
                forward_pass(
                    leaky_RELU(forward_pass(array, dense_layer), 0.01), weight_matrix
                )
            ).astype(cp.float32)
        )

    def learning(grad_W, grad_b, current_W, current_b, learning_rate):
        # Clip gradients to prevent explosion
        grad_W = cp.clip(grad_W, -1e5, 1e5)
        grad_b = cp.clip(grad_b, -1e5, 1e5)

        new_grad = current_W - learning_rate * grad_W
        new_bias = current_b - learning_rate * grad_b

        return new_grad, new_bias

    # weight_matrix = np.load("conv_neural_net/trained_weights.npy")

    backprop_gradients = cp.empty((0))
    hidden_unit_errors = cp.empty((0))

    # weight_matrix = kaiming_uniform_init((flattened_array.size, 10), flattened_array.size)
    # weight_matrix = np.empty((0))

    # forwarded = forward_pass(flattened_array, weight_matrix)

    label_arr = cp.empty((0))

    epochs = 50
    batch_size = 256

    load = ""
    load_bool = -1

    while load_bool == -1:
        load = input("Load from save? (y/n) \n")
        if load == "y" or load == "Y":
            weight_matrix = cp.load("Saves/trained_weights.npy")
            dense_layer = cp.load("Saves/trained_dense.npy")
            load_bool = 1
        elif load == "n" or load == "N":
            # weight_matrix = kaiming_uniform_init((128, 10), 128)
            # dense_layer = kaiming_uniform_init((16, 128), 16)
            weight_matrix = kaiming_uniform_init((128, 10), 128)
            dense_layer = kaiming_uniform_init((784, 128), 784)
            load_bool = 1

    bias_hidden = cp.zeros((1, 128))
    bias_out = cp.zeros((1, 10))
    batch_start = 0

    def run(weight_matrix, dense_layer, bias_out, bias_hidden):

        for epoch in range(epochs):
            total_loss = 0

            for i in range(0, len(data), batch_size):
                batch_end = min(i + batch_size, len(data))
                batch_size_actual = batch_end - i

                # Vectorize: process entire batch at once
                batch_images = cp.asarray(
                    [cp.asarray(data[j]).squeeze() for j in range(i, batch_end)]
                )
                batch_flat = cp.array(
                    [batch_images[k].flatten() for k in range(batch_size_actual)]
                )  # Shape: (batch_size, 784)
                batch_labels = labels[i:batch_end]

                # Forward pass on entire batch (NumPy parallelizes this)
                hidden_out = leaky_RELU(
                    forward_pass(batch_flat, dense_layer), 0.01
                )  # (batch_size, 128)
                logits = forward_pass(hidden_out, weight_matrix)  # (batch_size, 10)

                # Batch softmax
                batch_preds = cp.array(
                    [softmax_function(logits[k]) for k in range(batch_size_actual)]
                )  # (batch_size, 10)

                # Batch one-hot labels
                batch_one_hot = cp.array(
                    [
                        get_one_hot(int(batch_labels[k]), 10)
                        for k in range(batch_size_actual)
                    ]
                )  # (batch_size, 10)

                # Batch error
                batch_error = batch_preds - batch_one_hot  # (batch_size, 10)

                # Backward pass on batch
                grad_W_out, grad_b_out, _ = backpropagate(
                    weight_matrix, hidden_out, batch_error
                )
                grad_W_hidden, grad_b_hidden, _ = backpropagate(
                    dense_layer, batch_flat, batch_error @ weight_matrix.T
                )

                # Update weights
                weight_matrix, bias_out = learning(
                    grad_W_out, grad_b_out, weight_matrix, bias_out, 0.001
                )
                dense_layer, bias_hidden = learning(
                    grad_W_hidden, grad_b_hidden, dense_layer, bias_hidden, 0.001
                )

                # Batch loss
                batch_loss_all = cross_entropy_loss(batch_preds, batch_labels)
                batch_loss = cp.mean(batch_loss_all)

                total_loss += batch_loss * batch_size_actual

            avg_loss = total_loss / len(data)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            cp.save("Saves/trained_weights.npy", weight_matrix)
            cp.save("Saves/trained_dense.npy", dense_layer)

    run(weight_matrix, dense_layer, bias_out, bias_hidden)

    # result = np.array(softmax_function(forwarded))
    # result = result.astype(float)
    # label = soft_to_label(result)

    # print(result)
    # print(label)

    ### NOTES
    # Currently doing back prop
    # need to do derivative of softmax
