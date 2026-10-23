import cupy as cp

class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # State dictionaries to track history per layer variable
        self.m = {}
        self.v = {}
        self.t = {}  # Time steps

    def update(self, key, weights, grads):
        """
        Performs an inline Adam parameter update.
        'key' must be a unique string identifier for the layer (e.g., 'conv', 'dense1').
        """
        if key not in self.m:
            self.m[key] = cp.zeros_like(weights)
            self.v[key] = cp.zeros_like(weights)
            self.t[key] = 0

        self.t[key] += 1
        t = self.t[key]

        # 1. Update biased first moment estimate (Momentum)
        self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grads

        # 2. Update biased second raw moment estimate (RMSprop)
        self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grads ** 2)

        # 3. Compute bias-corrected first moment estimate
        m_hat = self.m[key] / (1.0 - self.beta1 ** t)

        # 4. Compute bias-corrected second raw moment estimate
        v_hat = self.v[key] / (1.0 - self.beta2 ** t)

        # 5. Update weights inline
        weights -= self.lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)
        return weights
