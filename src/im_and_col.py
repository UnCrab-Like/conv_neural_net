import cupy as cp

class IMCOL_:
    @staticmethod
    def im2col(x, filter_h, filter_w, padding=0, stride=1):
        """
        Transforms an image tensor (N, C, H, W) into a column matrix.
        Used in the FORWARD pass.
        """
        N, C, H, W = x.shape
        out_h = (H + 2 * padding - filter_h) // stride + 1
        out_w = (W + 2 * padding - filter_w) // stride + 1
        
        # Apply padding to height and width dimensions
        x_pad = cp.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
        
        # Initialize the output matrix structure
        cols = cp.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=x.dtype)
        
        # Extract overlapping regions
        for kh in range(filter_h):
            h_end = kh + out_h * stride
            for kw in range(filter_w):
                w_end = kw + out_w * stride
                cols[:, :, kh, kw, :, :] = x_pad[:, :, kh:h_end:stride, kw:w_end:stride]
                
        # Reshape to a 2D matrix: (C * filter_h * filter_w, N * out_h * out_w)
        cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(C * filter_h * filter_w, -1)
        return cols

    @staticmethod
    def col2im(x_col, N, C, H, W, filter_h, filter_w, padding=0, stride=1):
        """
        Transforms a column matrix back into an image tensor (N, C, H, W).
        Used in the BACKWARD pass to accumulate overlapping gradients.
        """
        out_h = (H + 2 * padding - filter_h) // stride + 1
        out_w = (W + 2 * padding - filter_w) // stride + 1
        
        # Initialize zero-filled array for the padded image dimensions
        x_pad = cp.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=x_col.dtype)
        
        # Reshape and transpose column matrix back into 6D patch dimensions:
        # (N, C, filter_h, filter_w, out_h, out_w)
        x_col_reshaped = x_col.reshape(C, filter_h, filter_w, N, out_h, out_w).transpose(3, 0, 1, 2, 4, 5)
        
        # Accumulate overlapping receptive fields back into the padded image matrix
        for kh in range(filter_h):
            h_end = kh + out_h * stride
            for kw in range(filter_w):
                w_end = kw + out_w * stride
                x_pad[:, :, kh:h_end:stride, kw:w_end:stride] += x_col_reshaped[:, :, kh, kw, :, :]
                
        # Remove the padding boundaries to recover the original image shape
        if padding > 0:
            return x_pad[:, :, padding:-padding, padding:-padding]
        return x_pad
