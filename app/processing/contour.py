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

    def active_contour(self,
                       image,
                       snake,
                       alpha=0.01,
                       beta=0.1,
                       w_line=0,
                       w_edge=1,
                       gamma=0.01,
                       max_px_move=1.0,
                       max_num_iter=2500,
                       convergence=0.1,
                       *,
                       boundary_condition='periodic',
                       ):

        max_num_iter = int(max_num_iter)
        if max_num_iter <= 0:
            raise ValueError("max_num_iter should be >0.")
        convergence_order = 10
        valid_bcs = [
            'periodic',
            'free',
            'fixed',
            'free-fixed',
            'fixed-free',
            'fixed-fixed',
            'free-free',
        ]
        if boundary_condition not in valid_bcs:
            raise ValueError(
                "Invalid boundary condition.\n"
                + "Should be one of: "
                + ", ".join(valid_bcs)
                + '.'
            )

        img = img_as_float(image)
        float_dtype = _supported_float_type(image.dtype)
        img = img.astype(float_dtype, copy=False)

        RGB = img.ndim == 3

        # Find edges using sobel:
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
        intp = RectBivariateSpline(
            np.arange(img.shape[1]), np.arange(img.shape[0]), img.T, kx=2, ky=2, s=0
        )

        snake_xy = snake[:, ::-1]
        x = snake_xy[:, 0].astype(float_dtype)
        y = snake_xy[:, 1].astype(float_dtype)
        n = len(x)
        xsave = np.empty((convergence_order, n), dtype=float_dtype)
        ysave = np.empty((convergence_order, n), dtype=float_dtype)

        # Build snake shape matrix for Euler equation in double precision
        eye_n = np.eye(n, dtype=float)
        a = (
                np.roll(eye_n, -1, axis=0) + np.roll(eye_n, -1, axis=1) - 2 * eye_n
        )  # second order derivative, central difference
        b = (
                np.roll(eye_n, -2, axis=0)
                + np.roll(eye_n, -2, axis=1)
                - 4 * np.roll(eye_n, -1, axis=0)
                - 4 * np.roll(eye_n, -1, axis=1)
                + 6 * eye_n
        )  # fourth order derivative, central difference
        A = -alpha * a + beta * b

        # Impose boundary conditions different from periodic:
        sfixed = False
        if boundary_condition.startswith('fixed'):
            A[0, :] = 0
            A[1, :] = 0
            A[1, :3] = [1, -2, 1]
            sfixed = True
        efixed = False
        if boundary_condition.endswith('fixed'):
            A[-1, :] = 0
            A[-2, :] = 0
            A[-2, -3:] = [1, -2, 1]
            efixed = True
        sfree = False
        if boundary_condition.startswith('free'):
            A[0, :] = 0
            A[0, :3] = [1, -2, 1]
            A[1, :] = 0
            A[1, :4] = [-1, 3, -3, 1]
            sfree = True
        efree = False
        if boundary_condition.endswith('free'):
            A[-1, :] = 0
            A[-1, -3:] = [1, -2, 1]
            A[-2, :] = 0
            A[-2, -4:] = [-1, 3, -3, 1]
            efree = True

        # Only one inversion is needed for implicit spline energy minimization:
        inv = np.linalg.inv(A + gamma * eye_n)
        # can use float_dtype once we have computed the inverse in double precision
        inv = inv.astype(float_dtype, copy=False)

        # Explicit time stepping for image energy minimization:
        for i in range(max_num_iter):
            # RectBivariateSpline always returns float64, so call astype here
            fx = intp(x, y, dx=1, grid=False).astype(float_dtype, copy=False)
            fy = intp(x, y, dy=1, grid=False).astype(float_dtype, copy=False)

            if sfixed:
                fx[0] = 0
                fy[0] = 0
            if efixed:
                fx[-1] = 0
                fy[-1] = 0
            if sfree:
                fx[0] *= 2
                fy[0] *= 2
            if efree:
                fx[-1] *= 2
                fy[-1] *= 2
            xn = inv @ (gamma * x + fx)
            yn = inv @ (gamma * y + fy)

            # Movements are capped to max_px_move per iteration:
            dx = max_px_move * np.tanh(xn - x)
            dy = max_px_move * np.tanh(yn - y)
            if sfixed:
                dx[0] = 0
                dy[0] = 0
            if efixed:
                dx[-1] = 0
                dy[-1] = 0
            x += dx
            y += dy

            # Convergence criteria needs to compare to a number of previous
            # configurations since oscillations can occur.
            j = i % (convergence_order + 1)
            if j < convergence_order:
                xsave[j, :] = x
                ysave[j, :] = y
            else:
                dist = np.min(
                    np.max(np.abs(xsave - x[None, :]) + np.abs(ysave - y[None, :]), 1)
                )
                if dist < convergence:
                    break

        return np.stack([y, x], axis=1)


