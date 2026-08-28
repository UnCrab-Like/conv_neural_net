PIL import Image
import numpy as np

class process_data:
    
    def to_image(array):
        return Image.fromarray(array)
    
    def softmax_function(array, epsilon=1e-15):
        x = np.asarray(array, dtype=np.float64).reshape(-1)
        x = x - np.max(x)
        exps = np.exp(x)
        probs = exps / np.sum(exps)
        return np.clip(probs, epsilon, 1.0 - epsilon)
    
    def soft_to_label(array):
        max_index = np.argmax(array)
        return max_index

    def get_one_hot(label_index, num_classes=10):
        one_hot = np.zeros(num_classes)
        one_hot[label_index] = 1
        return one_hot
