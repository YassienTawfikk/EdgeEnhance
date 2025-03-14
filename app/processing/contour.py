import numpy as np
import cv2


class Contour:
    def initialize_contour(self, image, center, radius, num_points):
        """ Initialize a discrete circle contour. """
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        return np.array([x, y]).T

    def evolve_contour(self, image, initial_contour, max_iterations=100, alpha=1, beta=1, gamma=1):
        """ Evolve the active contour using a greedy algorithm. """
        # Check if the image is colored and convert it to grayscale if necessary
        if len(image.shape) == 3:  # Color image has 3 channels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradients of the image
        Gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        Gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

        external_energy = -np.sqrt(Gx ** 2 + Gy ** 2)

        # Initialize the contour
        contour = initial_contour.copy()
        n = contour.shape[0]

        for _ in range(max_iterations):
            # Internal energy calculation
            internal_energy = np.zeros((n, 2))
            for i in range(n):
                v_next = contour[(i + 1) % n]
                v_prev = contour[(i - 1) % n]
                v_current = contour[i]

                # Energy terms
                term1 = alpha * np.linalg.norm(v_next - v_current) ** 2
                term2 = beta * np.linalg.norm(v_next - 2 * v_current + v_prev) ** 2

                internal_energy[i] = term1 + term2

            # External energy calculation
            external_energy_values = np.zeros(n)
            for i in range(n):
                x, y = int(contour[i, 0]), int(contour[i, 1])
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    external_energy_values[i] = external_energy[y, x]

            # Total energy
            total_energy = internal_energy + gamma * external_energy_values[:, np.newaxis]

            # Update contour
            for i in range(n):
                contour[i] -= total_energy[i] * 0.1  # Adjust step size as needed

        return contour
