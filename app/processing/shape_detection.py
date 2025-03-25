import numpy as np
import cv2
from app.processing.canny_edge import CannyEdge
import random


class ShapeDetection:
    # HAYA5OD KOL ARGUMENTS EL CANNY PLUX EL ORIGINAL IMAGE
    @staticmethod
    def superimpose_line(original_image, threshold=150, theta_res=1, rho_res=1):
        binary_edge_map = CannyEdge.apply_canny(original_image, 5, 3, 25, 100, 3, True)

        # Get image dimensions
        height, width = binary_edge_map.shape

        # Define the maximum possible value for rho (image diagonal)
        diagonal = int(np.sqrt(height ** 2 + width ** 2))

        # Define rho and theta ranges
        rhos = np.arange(-diagonal, diagonal, rho_res)
        thetas = np.deg2rad(np.arange(-90, 90, theta_res))  # Convert degrees to radians

        # Create the accumulator array (votes)
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

        # Get edge points
        edge_points = np.argwhere(binary_edge_map > 0)

        # Precompute cos(theta) and sin(theta) values
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)

        # Voting process (optimized)
        for y, x in edge_points:  # For each edge pixel
            rhos_calc = (x * cos_thetas + y * sin_thetas).astype(int)  # Compute rho values for all thetas at once
            rho_indices = np.clip(rhos_calc + diagonal, 0, len(rhos) - 1)  # Map rho to index
            accumulator[rho_indices, np.arange(len(thetas))] += 1  # Increment votes in one operation

        # Extract lines based on threshold
        detected_lines = np.argwhere(accumulator > threshold)

        # Convert the grayscale image to BGR for visualization
        processed_image = original_image

        # Draw the detected lines
        for rho_idx, theta_idx in detected_lines:
            rho = rhos[rho_idx]
            theta = thetas[theta_idx]

            # Convert (rho, theta) to two points for line drawing
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return processed_image

    @staticmethod
    def superimpose_circle(original_image, canny_high_threshold=200, max_radius=190, min_radius=0, threshold_factor=0.8):

        canny_low_threshold = canny_high_threshold / 2

        image_edges = CannyEdge.apply_canny(original_image, 3, 0.1, canny_low_threshold, canny_high_threshold, 3, False)

        height, width = image_edges.shape
        # max_radius = min(height, width) // 2
        # print(max_radius)
        # min_radius = 15

        accumulator = np.zeros((max_radius, width, height), dtype=np.uint8)

        edge_points = np.argwhere(image_edges > 0)
        angle_step = 5

        angles = np.deg2rad(np.arange(0, 360, angle_step))
        # having x,y coords, and looping through r, and through angle, we want to find a,b
        for y, x in edge_points:
            for r in range(min_radius, max_radius):
                a_vals = (x - r * np.cos(angles)).astype(int)
                b_vals = (y - r * np.sin(angles)).astype(int)
                for a, b in zip(a_vals, b_vals):
                    if 0 <= a < width and 0 <= b < height:
                        accumulator[r, a, b] += 1

        threshold = np.max(accumulator) * threshold_factor  # dynamic threshold
        print(threshold)

        circles = np.argwhere(accumulator > threshold)  # Get (r, a, b) where votes are high
        print(accumulator)
        print(circles)

        for r, a, b in circles:
            print(f"Detected circle at ({a}, {b}) with radius {r}")
            cv2.circle(original_image, (a, b), r, (0, 0, 255), 1)

        return original_image

    @staticmethod
    def superimpose_ellipse(original_image, min_radius=10, max_radius=60, threshold=0.5):
        """
        1) Detect edges using Canny.
        2) Perform a brute-force Hough-like transform to find axis-aligned ellipses.
        3) Draw all ellipses whose vote count exceeds 'threshold * max_votes' in red.

        :param original_image: Input image in BGR format (HxWx3).
        :param min_radius: Minimum semi-axis length to consider.
        :param max_radius: Maximum semi-axis length to consider.
        :param threshold: Fraction (0 to 1). E.g., 0.5 => only ellipses with
                          at least 50% of the max votes are drawn.
        :return: The same image with detected ellipses drawn in red.
        """

        print(min_radius)
        print(max_radius)
        print(threshold)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # 1) Canny edge detection using built-in OpenCV function
        edge_image = CannyEdge.apply_canny(original_image, 5, 3, 25, 100, 3, True)

        # Get image dimensions
        height, width = edge_image.shape

        # 2) Gather edge points
        edge_points = np.argwhere(edge_image > 0)  # Each is [y, x]
        if len(edge_points) == 0:
            print("No edges found. Returning original image.")
            return original_image

        # 3) Accumulator for axis-aligned ellipses
        #    Key: (center_x, center_y, a, b) => votes
        votes = {}

        # For each edge pixel, for each (a, b) in range,
        # we consider the 4 possible center positions:
        for (y, x) in edge_points:
            for a in range(min_radius, max_radius + 1):
                for b in range(min_radius, max_radius + 1):
                    centers = [
                        (x - a, y - b),
                        (x - a, y + b),
                        (x + a, y - b),
                        (x + a, y + b)
                    ]
                    for (cx, cy) in centers:
                        if 0 <= cx < width and 0 <= cy < height:
                            key = (cx, cy, a, b)
                            votes[key] = votes.get(key, 0) + 1

        if not votes:
            print("No ellipse votes found. Returning original image.")
            return original_image

        # 4) Find the maximum vote
        max_votes = max(votes.values())

        # 5) Determine the required votes for an ellipse to be considered valid
        vote_cutoff = max_votes * threshold

        # 6) Convert to a list of (key, count) and filter
        ellipse_candidates = [(k, v) for k, v in votes.items() if v >= vote_cutoff]

        # 7) Draw ellipses in red
        for (cx, cy, a, b), count in ellipse_candidates:
            cv2.ellipse(original_image, (cx, cy), (a, b), 0, 0, 360, (0, 0, 255), 2)

        return original_image

    @staticmethod
    def detect_ellipses(original_image, low_threshold=50, high_threshold=150):
        # Apply the Canny edge detector
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Find contours from the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Superimpose ellipses on the original image
        processed_image = original_image.copy()
        for contour in contours:
            if len(contour) >= 5:  # Need at least 5 points to fit ellipse
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(processed_image, ellipse, (0, 0, 255), 2)

        return processed_image
