from os import stat

from PIL import Image

# import numpy as np   Switching to cupy for quicker GPU calculation.
import cupy as cp


class front_propagation:
    @staticmethod
    def filter_image(
        img_matrix,
        filter,
        mode="L",
    ):

        if filter.size == (0):
            return -1

        if mode == "RGB":
            # ### For RGB images, image will be split in 3 bands, R, G, B, filtered then recombined
            # r_band, g_band, b_band = self.image.split()

            # r_matrix = np.array(r_band)
            # g_matrix = np.array(g_band)
            # b_matrix = np.array(b_band)

            ### Dimensions of the image.

            r_matrix = img_matrix[:, :, 0]
            g_matrix = img_matrix[:, :, 1]
            b_matrix = img_matrix[:, :, 2]

            if r_matrix.shape != g_matrix.shape or r_matrix.shape != b_matrix.shape:
                return -1

            width, height = r_matrix.shape()
            f_height_max = height / 3
            f_width_max = width / 3

            ### Filters.
            # The filter is non-static -> self.filter, contrary to my first iteration which used sobel. It is kaimed already

            ### Filtered image matrix initialisation
            # filtered_r_matrix = cp.empty((int(f_height_max), int(f_width_max)))
            # filtered_g_matrix = cp.empty((int(f_height_max), int(f_width_max)))
            # filtered_b_matrix = cp.empty((int(f_height_max), int(f_width_max)))
            #
            # ### Filtering.
            # # Passing the filter through the different color bands.
            # for y in range(int(f_height_max)):
            #     for x in range(int(f_width_max)):
            #         r_region = r_matrix[x * 3 : x * 3 + 3, y * 3 : y * 3 + 3]
            #         g_region = g_matrix[x * 3 : x * 3 + 3, y * 3 : y * 3 + 3]
            #         b_region = b_matrix[x * 3 : x * 3 + 3, y * 3 : y * 3 + 3]
            #
            #         f_r_matrix = cp.sum(r_region * filter)
            #         f_g_matrix = cp.sum(g_region * filter)
            #         f_b_matrix = cp.sum(b_region * filter)
            #
            #         filtered_r_matrix[y, x] = f_r_matrix
            #         filtered_g_matrix[y, x] = f_g_matrix
            #         filtered_b_matrix[y, x] = f_b_matrix
            #
            # ### Color clipping
            # # Reajusting the matrix values to be in between 0 and 255.
            # filtered_r_matrix = cp.clip(filtered_r_matrix, 0, 255).astype(cp.uint8)
            # filtered_g_matrix = cp.clip(filtered_g_matrix, 0, 255).astype(cp.uint8)
            # filtered_b_matrix = cp.clip(filtered_b_matrix, 0, 255).astype(cp.uint8)
            #
            # r_hor_matrix = cp.empty(0)
            # g_hor_matrix = cp.empty(0)
            # b_hor_matrix = cp.empty(0)
            #
            # for y in range(f_height_max):
            #     if r_hor_matrix * g_hor_matrix * b_hor_matrix == cp.empty((0)):
            #         r_hor_matrix = r_matrix[0:width, y * 3 : y * 3 + 3]
            #         r_hor_matrix = r_matrix[0:width, y * 3 : y * 3 + 3]
            #         b_hor_matrix = b_matrix[0:width, y * 3 : y * 3 + 3]
            #     else:
            #         r_hor_matrix = cp.hstack(
            #             (r_hor_matrix, r_matrix[0:width, y * 3 : y * 3 + 3])
            #         )
            #         g_hor_matrix = cp.hstack(
            #             (g_hor_matrix, g_matrix[0:width, y * 3 : y * 3 + 3])
            #         )
            #         b_hor_matrix = cp.hstack(
            #             (b_hor_matrix, b_matrix[0:width, y * 3 : y * 3 + 3])
            #         )
            #
            # f_r_matrix = cp.sum(r_hor_matrix * filter)
            # f_g_matrix = cp.sum(g_hor_matrix * filter)
            # f_b_matrix = cp.sum(b_hor_matrix * filter)

            ### Vectorize the image matrix to a 9x1024
            # Create the coordinate matrix

            fil_row = cp.repeat(cp.arange(3), 3)  # 9x
            fil_col = cp.tile(cp.arange(3), 3)  # 9x

            matrix_row = cp.repeat(cp.arange(width - 2), width - 2)  # nx
            matrix_column = cp.tile(cp.arange(height - 2), height - 2)  # mx

            cords_x = fil_row[:, cp.newaxis] + matrix_row  # 9xn
            cords_y = fil_col[:, cp.newaxis] + matrix_column  # 9xm

            r_cord_matrix = r_matrix[cords_x, cords_y]
            g_cord_matrix = g_matrix[cords_x, cords_y]
            b_cord_matrix = b_matrix[cords_x, cords_y]

            filtered_r_matrix = (
                cp.sum(filter.ravel()[:, cp.newaxis] * r_cord_matrix, axis=0)
            ).reshape(width - 2, height - 2)

            filtered_g_matrix = (
                cp.sum(filter.ravel()[:, cp.newaxis] * g_cord_matrix, axis=0)
            ).reshape(width - 2, height - 2)

            filtered_b_matrix = (
                cp.sum(filter.ravel()[:, cp.newaxis] * b_cord_matrix, axis=0)
            ).reshape(width - 2, height - 2)

            # filtered_r_matrix = cp.clip(filtered_r_matrix, 0, 255).astype(cp.uint8)
            # filtered_g_matrix = cp.clip(filtered_g_matrix, 0, 255).astype(cp.uint8)
            # filtered_b_matrix = cp.clip(filtered_b_matrix, 0, 255).astype(cp.uint8)

            ### Merging color bands
            # c_img = Image.merge(
            #     "RGB",
            #     (
            #         Image.fromarray(filtered_r_matrix),
            #         Image.fromarray(filtered_g_matrix),
            #         Image.fromarray(filtered_b_matrix)
            #     ),
            # )

            ###
            return filtered_r_matrix, filtered_g_matrix, filtered_b_matrix

        elif mode == "L":
            ### Image initialisation

            ### Dimensions of the image.
            batch, height, width = img_matrix.shape
            filter = cp.asarray(filter)
            # f_height_max = height / 3
            # f_width_max = width / 3

            ### Filters.
            # The filter is non-static -> self.filter, contrary to my first iteration which used sobel. It is kaimed already

            ### Filtered image matrix initialisation
            # filtered_img_matrix = cp.empty((int(f_height_max), int(f_width_max)))
            #
            # ### Filtering.
            # # Passing the filter through the image
            # for y in range(int(f_height_max)):
            #     for x in range(int(f_width_max)):
            #         img_region = img_matrix[x * 3 : x * 3 + 3, y * 3 : y * 3 + 3]
            #
            #         f_img_matrix = cp.sum(img_region * filter)
            #
            #         filtered_img_matrix[y, x] = f_img_matrix
            #
            # ### Color clipping
            # # Reajusting the matrix values to be in between 0 and 255.
            # filtered_img_matrix = cp.clip(filtered_img_matrix, 0, 255).astype(cp.uint8)
            # # Image creation.
            # # filtered_img = Image.fromarray(filtered_img_matrix)
            # # filtered_img = filtered_img.convert("L")

            ###
            
            output_height = height - 3 + 1 # 3 being filter height
            output_width = width - 3 + 1 # 3 being filter height

            fil_row = cp.repeat(cp.arange(3), 3)  # 9x
            fil_col = cp.tile(cp.arange(3), 3)  # 9x

            matrix_row = cp.repeat(cp.arange(output_height), output_width)  # nx
            matrix_column = cp.tile(cp.arange(output_width), output_height )  # mx

            # matrix_row, matrix_column = cp.meshgrid(cp.arange(width - 2), cp.arange(height - 2), indexing='ij')

            # # Flatten to establish a uniform 1D spatial mapping
            # matrix_row = matrix_row.ravel()
            # matrix_column = matrix_column.ravel()


            cords_x = fil_row[:, cp.newaxis] + matrix_row  # 9xn
            cords_y = fil_col[:, cp.newaxis] + matrix_column  # 9xm

            cords_x = cp.asarray(cords_x).astype(cp.int32)
            cords_y = cp.asarray(cords_y).astype(cp.int32)

            img_matrix_ = cp.asarray(img_matrix)

            cord_matrix = img_matrix_[:, cords_x, cords_y]

            filtered_img_matrix = (
                cp.sum(filter.ravel()[cp.newaxis, :, cp.newaxis] * cord_matrix, axis=1)
            ).reshape(batch, output_height, output_width)

            # filtered_img_matrix = cp.clip(filtered_img_matrix, 0, 255).astype(cp.uint8)

            return filtered_img_matrix

    # def leaky_RELU_img(array, ):
    #     if self.image.mode == "RGB":
    #         r_band, g_band, b_band = self.image.split()

    #         r_matrix = np.array(r_band)
    #         g_matrix = np.array(g_band)
    #         b_matrix = np.array(b_band)

    #         # width, height = image.size

    #         act_r_image = np.where(r_matrix > 0, r_matrix, r_matrix * self.alpha)
    #         act_g_image = np.where(g_matrix > 0, g_matrix, g_matrix * self.alpha)
    #         act_b_image = np.where(b_matrix > 0, b_matrix, b_matrix * self.alpha)

    #         act_r_image = np.clip(act_r_image, 0, 255).astype(np.uint8)
    #         act_g_image = np.clip(act_g_image, 0, 255).astype(np.uint8)
    #         act_b_image = np.clip(act_b_image, 0, 255).astype(np.uint8)

    #         activ_img = Image.merge(
    #             "RGB",
    #             (
    #                 Image.fromarray(act_r_image),
    #                 Image.fromarray(act_g_image),
    #                 Image.fromarray(act_b_image),
    #             ),
    #         )

    #         self.image = activ_img
    #         return activ_img

    #     elif self.image.mode == "L":
    #         img_matrix = np.array(self.image)

    #         # width, height = image.size

    #         activ_img = np.where(img_matrix > 0, img_matrix, img_matrix * self.alpha)

    #         activ_img = np.clip(activ_img, 0, 255).astype(np.uint8)

    #         activ_img = Image.fromarray(activ_img)

    #         self.image = activ_img
    #         return activ_img

    @staticmethod
    def leaky_RELU(array, alpha):
        arr = array
        relud_array = cp.where(arr > 0, arr, arr * alpha)
        # relud_array = cp.clip(relud_array, 0, 255).astype(cp.uint8)

        return relud_array

    @staticmethod
    def max_pooling(img_matrix, mode="L"):
        if mode == "RGB":
            # r_band, g_band, b_band = self.image.split()

            # r_matrix = np.array(r_band)
            # g_matrix = np.array(g_band)
            # b_matrix = np.array(b_band)

            r_matrix = img_matrix[:, :, 0]
            g_matrix = img_matrix[:, :, 1]
            b_matrix = img_matrix[:, :, 2]

            if r_matrix.size() != g_matrix or r_matrix != b_matrix:
                return -1

            width, height = r_matrix.size()

            r_max_arr = cp.empty((int(width / 2), int(height / 2)))
            g_max_arr = cp.empty((int(width / 2), int(height / 2)))
            b_max_arr = cp.empty((int(width / 2), int(height / 2)))

            for y in range(int(height / 2)):
                for x in range(int(width / 2)):
                    r_region = r_matrix[y * 2 : y * 2 + 2, x * 2 : x * 2 + 2]
                    g_region = g_matrix[y * 2 : y * 2 + 2, x * 2 : x * 2 + 2]
                    b_region = b_matrix[y * 2 : y * 2 + 2, x * 2 : x * 2 + 2]

                    r_max = cp.max(r_region)
                    g_max = cp.max(g_region)
                    b_max = cp.max(b_region)

                    r_max_arr[y][x] = r_max
                    g_max_arr[y][x] = g_max
                    b_max_arr[y][x] = b_max

            #r_max_arr = cp.clip(r_max_arr, 0, 255).astype(cp.uint8)
            #g_max_arr = cp.clip(g_max_arr, 0, 255).astype(cp.uint8)
            #b_max_arr = cp.clip(b_max_arr, 0, 255).astype(cp.uint8)

            # p_img = Image.merge(
            #     "RGB",
            #     (
            #         Image.fromarray(r_max_arr),
            #         Image.fromarray(g_max_arr),
            #         Image.fromarray(b_max_arr),
            #     ),
            # )

            return r_max_arr, g_max_arr, b_max_arr

        elif mode == "L":
            # img_matrix = np.array(self.image)

            # width, height = img_matrix.shape

            # img_max_arr = cp.empty((int(height / 2), int(width / 2)))

            # for y in range(int(height / 2)):
            #     for x in range(int(width / 2)):
            #         img_region = img_matrix[y * 2 : y * 2 + 2, x * 2 : x * 2 + 2]

            #         img_max = cp.max(img_region)

            #         img_max_arr[y][x] = img_max

            batch, height, width = img_matrix.shape

            pooled = img_matrix.reshape(
                batch, height // 2,
                2,
                width // 2,
                2,
            ).max(axis=(2, 4))

            #img_max_arr = cp.clip(img_max_arr, 0, 255).astype(cp.uint8)

            # img_max = Image.fromarray(img_max_arr)

            # self.image = img_max
            # return img_max_arr
            return pooled

    # max_pooled_img = max_pooling(activated_img)
    # max_pooled_img.show()

    # max_pooled_array = np.array(max_pooled_img)
    # flattened_array = max_pooled_array.flatten()
    # print(flattened_array.shape)

    # Fully connected layer
    #

    # softmax activation

    @staticmethod
    def forward_pass(array, matrix, bias):
        # weight_vec = kaiming_uniform_init((array.size, 10), array.size)
        return cp.dot(array, matrix) + bias