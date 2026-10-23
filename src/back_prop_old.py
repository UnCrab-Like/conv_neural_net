from os import stat

from PIL import Image
import cupy as cp


class back_propagation:
    @staticmethod
    def softmax_function(array, epsilon=1e-15):
        x = cp.asarray(array, dtype=cp.float64).reshape(-1)
        x = x - cp.max(x)
        exps = cp.exp(x)
        probs = exps / cp.sum(exps)
        return cp.clip(probs, epsilon, 1.0 - epsilon)

    @staticmethod
    def get_one_hot(label_index, num_classes=10):
        one_hot = cp.zeros(num_classes)
        one_hot[label_index] = 1
        return one_hot

    @staticmethod
    def cross_entropy_loss(y_pred, y_true, epsilon=1e-15):
        y_pred = cp.clip(y_pred, epsilon, 1.0 - epsilon)
        return -cp.log(y_pred[cp.arange(len(y_true)), y_true.astype(cp.int32)])

    @staticmethod
    def deriv_CEL_and_softmax(ypredScores, ytrueScores, epsilon=1e-15):
        # gradient = np.subtract(ypredScores, ytrueScores)
        # mask = (ypredScores > epsilon) & (ypredScores < (1.0 - epsilon))
        # return np.array(gradient * mask)

        return cp.array(ypredScores - ytrueScores).reshape(-1)

    @staticmethod
    def deriv_leaky_RELU(input, alpha):
        derived_RELU = cp.where(input >= 0, 1, alpha)
        return derived_RELU

    @staticmethod
    def deriv_max_pooling(input):
        input = cp.asarray(input)
        batch, height, width = input.shape
        input_reshaped = input.reshape((batch, height // 2, 2, width // 2, 2))
        #input_reshaped = input.reshape((batch, height // 2, 2, width))
        #derived_max = [[cp.where(input_reshaped[:, i, : j, :] == input_reshaped[:, i, :, j, :].max(), 1, 0) for j in range(height//2)]for i in range(width//2)]
        #return input_reshaped[:, 1, :, 0:4]
        derived_max = [cp.where(input_reshaped[:, i, :, j, :] == input_reshaped[:, i, :, j, :].max(), 1, 0) for i in range(height//2) for j in range(width//2)]
        derived_max = cp.stack(derived_max, axis=2).reshape(batch, height, width)
        return derived_max

    @staticmethod
    def deriv_conv_layer(input, filter, output, error):
        input = cp.asarray(input)
        filter = cp.asarray(filter)
        output = cp.asarray(output)
        error = cp.asarray(error)

        dW = cp.zeros(filter.shape)   # Partial derivative of the filter (weight)
        dX = cp.zeros(input.shape)    # Partial derivative of the input (x)

        for m in range(filter.shape[0]):
            for n in range(filter.shape[1]):
                dW[m, n] = cp.sum(error * input[:,m:m+output.shape[1], n:n+output.shape[2]]) / input.shape[0]
                dX[:, m:m+output.shape[1], n:n+output.shape[2]] += error * filter[m, n]
        
        return dW, dX

    @staticmethod
    def backpropagate(layer, prev_activtions, error_current, batch_size):
        grad_W = (prev_activtions.T @ error_current)/ batch_size
        grad_b = cp.sum(error_current, axis=0) / batch_size
        error_prev = error_current @ layer.T

        return grad_W, grad_b, error_prev

    @staticmethod
    def hidden_unit_errors(activ_derivs, W, next_errors):
        return cp.asarray(activ_derivs) * (W.dot(next_errors))

    @staticmethod
    def learning(grad_W, grad_b, current_W, current_b, learning_rate):
        # Clip gradients to prevent explosion
        grad_W = cp.clip(grad_W, -1e5, 1e5)
        grad_b = cp.clip(grad_b, -1e5, 1e5)

        new_grad = current_W - learning_rate * grad_W
        new_bias = current_b - learning_rate * grad_b
