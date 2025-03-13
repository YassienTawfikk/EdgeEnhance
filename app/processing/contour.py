import numpy as np
import cv2

class Contour:
    def gaussian_blur(self,image, kernel_size, sigma):
        """Apply Gaussian blur to an image."""
        # Create a Gaussian kernel
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)  # Normalize the kernel

        # Apply the kernel to the image
        blurred_image = np.zeros_like(image)
        pad_width = kernel_size // 2
        padded_image = np.pad(image, pad_width, mode='edge')

        for i in range(padded_image.shape[0] - kernel_size):
            for j in range(padded_image.shape[1] - kernel_size):
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                blurred_image[i, j] = np.sum(region * kernel)

        return blurred_image


    def initialize_contour(self,image, num_points, radius):
        """Initialize the contour around the center of the image."""
        s = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(s) + image.shape[1] // 2
        y = radius * np.sin(s) + image.shape[0] // 2
        return np.array([x, y]).T

    def evolve_contour(self,snake, image, num_iterations, alpha, beta, gamma, window_size):
        """Evolve the contour using the greedy algorithm."""
        if window_size <= 0 or window_size % 2 == 0:
            window_size=5
        for _ in range(num_iterations):
            # Compute the energy terms
            dx = np.roll(snake[:, 0], -1) - snake[:, 0]
            dy = np.roll(snake[:, 1], -1) - snake[:, 1]

            # Internal forces
            internal_force = alpha * (np.roll(snake, -1, axis=0) + np.roll(snake, 1, axis=0) - 2 * snake)

            # External forces from the image
            blurred_image = cv2.GaussianBlur(image, (window_size, window_size), 0)
            img_grad = np.gradient(blurred_image.astype(float))

            # Clamp the snake coordinates to be within valid range
            x_indices = np.clip(snake[:, 0].astype(int), 0, img_grad[0].shape[1] - 1)
            y_indices = np.clip(snake[:, 1].astype(int), 0, img_grad[0].shape[0] - 1)

            # Calculate the external force based on gradients
            # Use both x and y gradients to construct external force
            external_force_x = gamma * (img_grad[0][y_indices, x_indices] - img_grad[0].min())
            external_force_y = gamma * (img_grad[1][y_indices, x_indices] - img_grad[1].min())

            # Combine external forces into a single array with shape (N, 2)
            external_force = np.column_stack((external_force_x, external_force_y))

            # Update the snake
            snake += internal_force + external_force

        return snake

    def chain_code(self,snake):
        """Compute the chain code representation of the contour."""
        codes = []
        for i in range(len(snake)):
            dx = int(snake[(i + 1) % len(snake)][0] - snake[i][0])
            dy = int(snake[(i + 1) % len(snake)][1] - snake[i][1])
            if dx == 1 and dy == 0:
                codes.append(0)  # Right
            elif dx == 0 and dy == 1:
                codes.append(1)  # Down
            elif dx == -1 and dy == 0:
                codes.append(2)  # Left
            elif dx == 0 and dy == -1:
                codes.append(3)  # Up
        return codes

    def compute_area_perimeter(self,snake,image):
        """Compute the area and perimeter of the contour."""
        snake = np.round(snake).astype(int)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for point in snake:
            mask[point[1], point[0]] = 1
        area = np.sum(mask)

        # Compute perimeter using the chain code
        perimeter = 0
        for i in range(len(snake)):
            perimeter += np.linalg.norm(snake[i] - snake[(i + 1) % len(snake)])

        return area, perimeter
