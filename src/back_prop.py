from os import stat

from PIL import Image
import cupy as cp

from src.im_and_col import IMCOL_

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
        batch, filter_batch, height, width = input.shape
        return_height, return_width = height // 2, width // 2

        input_reshaped = input.reshape((batch, filter_batch, return_height, 2, return_width, 2))
        block_maxes = input_reshaped.max(axis=(3, 5), keepdims=True)

        derived_max = cp.where(input_reshaped == block_maxes, 1, 0)
        derived_max = derived_max.transpose(0, 1, 2, 3, 4, 5)
        return derived_max.reshape(batch, filter_batch, height, width)
    
    @staticmethod
    def conv_backward_im2col(dOut, cache):
        """
        Computes the backward pass for a convolutional layer using col2im.
        """
        x, w, _, conv_param = cache
        x = cp.asarray(x)
        w = cp.asarray(w)
        dOut = cp.asarray(dOut)
        stride, pad = conv_param["stride"], conv_param["pad"]

        grayscale = x.ndim == 3 and w.ndim == 3
        if grayscale:
            x = x[:, cp.newaxis, :, :]
            w = w[:, cp.newaxis, :, :]
        elif x.ndim != 4 or w.ndim != 4:
            raise ValueError("x and w must both be 3D grayscale or 4D channel tensors")

        N, C, H, W = x.shape
        F, weight_channels, HH, WW = w.shape
        if C != weight_channels:
            raise ValueError("The input and filter channel counts must match")

        out_h = (H + 2 * pad - HH) // stride + 1
        out_w = (W + 2 * pad - WW) // stride + 1
        if dOut.shape != (N, F, out_h, out_w):
            raise ValueError(
                f"dOut must have shape {(N, F, out_h, out_w)}, got {dOut.shape}"
            )

        X_col = IMCOL_.im2col(x, HH, WW, pad, stride)
        dOut_reshaped = dOut.transpose(1, 0, 2, 3).reshape(F, -1)

        db = cp.sum(dOut_reshaped, axis=1)
        dW = (dOut_reshaped @ X_col.T).reshape(F, C, HH, WW)
        dX_col = w.reshape(F, -1).T @ dOut_reshaped
        dX = IMCOL_.col2im(dX_col, N, C, H, W, HH, WW, pad, stride)

        if grayscale:
            dX = dX[:, 0, :, :]
            dW = dW[:, 0, :, :]
        return dX, dW, db

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


        return new_grad, new_bias
