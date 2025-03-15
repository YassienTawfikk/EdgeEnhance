import numpy as np
import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage.filters import sobel
from skimage.util import img_as_float
from skimage._shared.utils import _supported_float_type


class Contour:
    def initialize_contour(self, image, center, radius, num_points):
        """ Initialize a discrete circle contour. """
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        return np.array([x, y]).T

    def active_contour(self, input_image, initial_snake, alpha=0.01, beta=0.1, weight_line=0, weight_edge=1, gamma=0.01,
                       max_pixel_move=1.0, max_iterations=2500, convergence_threshold=0.1):

        max_iterations = int(max_iterations)
        if max_iterations <= 0:
            raise ValueError("max_iterations should be > 0.")
        convergence_order = 10

        # Convert image to float
        img = img_as_float(input_image)
        float_dtype = _supported_float_type(input_image.dtype)
        img = img.astype(float_dtype, copy=False)

        is_rgb = img.ndim == 3

        # Find edges using Sobel filter
        if weight_edge != 0:
            if is_rgb:
                edges = [sobel(img[:, :, channel]) for channel in range(3)]
            else:
                edges = [sobel(img)]
        else:
            edges = [0]

        # Superimpose intensity and edge images
        if is_rgb:
            img = weight_line * np.sum(img, axis=2) + weight_edge * sum(edges)
        else:
            img = weight_line * img + weight_edge * edges[0]

        # Interpolate for smoothness
        interpolator = RectBivariateSpline(
            np.arange(img.shape[1]), np.arange(img.shape[0]), img.T, kx=2, ky=2, s=0
        )

        snake_coordinates = initial_snake[:, ::-1]
        x_coords = snake_coordinates[:, 0].astype(float_dtype)
        y_coords = snake_coordinates[:, 1].astype(float_dtype)
        n_points = len(x_coords)
        x_save = np.empty((convergence_order, n_points), dtype=float_dtype)
        y_save = np.empty((convergence_order, n_points), dtype=float_dtype)

        # Build snake shape matrix for Euler equation in double precision
        identity_matrix = np.eye(n_points, dtype=float)
        second_order_derivative = (
                np.roll(identity_matrix, -1, axis=0) + np.roll(identity_matrix, -1, axis=1) - 2 * identity_matrix
        )  # second order derivative, central difference
        fourth_order_derivative = (
                np.roll(identity_matrix, -2, axis=0)
                + np.roll(identity_matrix, -2, axis=1)
                - 4 * np.roll(identity_matrix, -1, axis=0)
                - 4 * np.roll(identity_matrix, -1, axis=1)
                + 6 * identity_matrix
        )  # fourth order derivative, central difference
        A_matrix = -alpha * second_order_derivative + beta * fourth_order_derivative

        # Only one inversion is needed for implicit spline energy minimization
        inverse_matrix = np.linalg.inv(A_matrix + gamma * identity_matrix)
        # Can use float_dtype once we have computed the inverse in double precision
        inverse_matrix = inverse_matrix.astype(float_dtype, copy=False)

        # Explicit time stepping for image energy minimization
        for iteration in range(max_iterations):
            # RectBivariateSpline always returns float64, so call astype here
            fx = interpolator(x_coords, y_coords, dx=1, grid=False).astype(float_dtype, copy=False)
            fy = interpolator(x_coords, y_coords, dy=1, grid=False).astype(float_dtype, copy=False)

            new_x_coords = inverse_matrix @ (gamma * x_coords + fx)
            new_y_coords = inverse_matrix @ (gamma * y_coords + fy)

            # Movements are capped to max_pixel_move per iteration
            delta_x = max_pixel_move * np.tanh(new_x_coords - x_coords)
            delta_y = max_pixel_move * np.tanh(new_y_coords - y_coords)

            x_coords += delta_x
            y_coords += delta_y

            # Convergence criteria needs to compare to a number of previous configurations since oscillations can occur
            j = iteration % (convergence_order + 1)
            if j < convergence_order:
                x_save[j, :] = x_coords
                y_save[j, :] = y_coords
            else:
                distance = np.min(
                    np.max(np.abs(x_save - x_coords[None, :]) + np.abs(y_save - y_coords[None, :]), axis=1)
                )
                if distance < convergence_threshold:
                    break

        return np.stack([y_coords, x_coords], axis=1)

    def active_contour_greedy(self, image, snake, alpha=0.01, beta=0.1, w_line=0, w_edge=1, gamma=0.01,
                              max_num_iter=15):
        max_num_iter = int(max_num_iter)
        if max_num_iter <= 0:
            raise ValueError("max_num_iter should be >0.")

        img = img_as_float(image)
        float_dtype = _supported_float_type(image.dtype)
        img = img.astype(float_dtype, copy=False)

        RGB = img.ndim == 3

        # Find edges using Sobel:
        if w_edge != 0:
            if RGB:
                edge = [sobel(img[:, :, 0]), sobel(img[:, :, 1]), sobel(img[:, :, 2])]
            else:
                edge = [sobel(img)]
        else:
            edge = [0]

        # Superimpose intensity and edge images:
        if RGB:
            img = w_line * np.sum(img, axis=2) + w_edge * sum(edge)
        else:
            img = w_line * img + w_edge * edge[0]

        # Interpolate for smoothness:
        intp = RectBivariateSpline(np.arange(img.shape[1]), np.arange(img.shape[0]), img.T, kx=2, ky=2, s=0)

        snake_xy = snake[:, ::-1]
        x = snake_xy[:, 0].astype(float_dtype)
        y = snake_xy[:, 1].astype(float_dtype)
        n = len(x)

        # Build snake shape matrix for Euler equation in double precision
        eye_n = np.eye(n, dtype=float)
        a = (np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1) - 2 * eye_n)  # second order derivative
        b = (np.roll(eye_n, -2, axis=0) + np.roll(eye_n, -2, axis=1) - 4 * np.roll(eye_n, -1, axis=0) - 4 * np.roll(
            eye_n, -1, axis=1) + 6 * eye_n)  # fourth order derivative
        A = -alpha * a + beta * b

        # Only one inversion is needed for implicit spline energy minimization:
        inv = np.linalg.inv(A + gamma * eye_n)
        inv = inv.astype(float_dtype, copy=False)

        # Explicit time stepping for image energy minimization:
        for i in range(max_num_iter):
            fx = intp(x, y, dx=1, grid=False).astype(float_dtype, copy=False)
            fy = intp(x, y, dy=1, grid=False).astype(float_dtype, copy=False)

            xn = inv @ (gamma * x + fx)
            yn = inv @ (gamma * y + fy)

            # Directly update positions:
            x = xn
            y = yn

        return np.stack([y, x], axis=1)


