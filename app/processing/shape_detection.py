import numpy as np
import cv2
from app.processing.canny_edge import CannyEdge
import random
from collections import defaultdict
from typing import Optional, Tuple


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
    def superimpose_ellipse(
            original_image,
            scale_factor=0.5,  # downscale 50%
            canny_high_threshold=300,  # reduce # of edges
            min_major_axis=20,  # narrower range
            max_major_axis=80,
            min_minor_axis=20,
            max_minor_axis=80,
            threshold_factor=0.8,
            max_edge_points=500,  # fewer edge points
            angle_step=15  # fewer angles
    ):
        """
        Detect and draw axis-aligned ellipses using a naive Hough approach,
        but with quick fixes for speed and memory:

        1. Resizing the input image to reduce resolution.
        2. Increasing Canny thresholds to reduce edge points.
        3. Randomly sampling edge points if there are too many.
        4. Reducing the angle steps for fewer accumulator votes.

        :param original_image: Input image (grayscale or BGR).
        :param scale_factor: Factor to resize the image (e.g., 0.5 halves the width/height).
        :param canny_high_threshold: High threshold for Canny (low = half).
        :param min_major_axis: Minimum major-axis length (a).
        :param max_major_axis: Maximum major-axis length (a).
        :param min_minor_axis: Minimum minor-axis length (b).
        :param max_minor_axis: Maximum minor-axis length (b).
        :param threshold_factor: Fraction of max votes used as the threshold.
        :param max_edge_points: Max number of edges to keep (randomly sampled).
        :param angle_step: Step size in degrees for angles around the ellipse.
        :return: The original image (scaled back up) with detected ellipses drawn in red.
        """

        # 1) Optionally resize the image for faster processing
        #    We'll do the detection on a smaller image, then scale results back up.
        if scale_factor != 1.0:
            small_image = cv2.resize(
                original_image,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_AREA
            )
        else:
            small_image = original_image

        # 2) Convert to grayscale if needed
        if len(small_image.shape) == 3:
            gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = small_image

        # 3) Canny edge detection
        canny_low_threshold = canny_high_threshold // 2
        edges = cv2.Canny(gray, canny_low_threshold, canny_high_threshold)

        # 4) Prepare the 4D accumulator: [a, b, center_x, center_y]
        height, width = edges.shape
        accumulator = np.zeros(
            (
                max_major_axis + 1,
                max_minor_axis + 1,
                width,
                height
            ),
            dtype=np.uint32
        )

        # 5) Gather edge points
        edge_points = np.argwhere(edges > 0)

        # -- Quick Fix: Randomly sample edges if there are too many
        num_edges = len(edge_points)
        if num_edges > max_edge_points:
            indices = np.random.choice(num_edges, max_edge_points, replace=False)
            edge_points = edge_points[indices]

        # 6) Discrete angles (larger angle_step => fewer votes => faster)
        angles = np.deg2rad(np.arange(0, 360, angle_step))

        # 7) Vote in the accumulator
        #    For each edge point (y, x), for each a in [min_major_axis..max_major_axis]
        #    and b in [min_minor_axis..max_minor_axis],
        #    we find center (cx, cy) with: x = cx + a*cos(theta), y = cy + b*sin(theta)
        #    => cx = x - a*cos(theta), cy = y - b*sin(theta)
        for (y, x) in edge_points:
            for a in range(min_major_axis, max_major_axis + 1):
                for b in range(min_minor_axis, max_minor_axis + 1):
                    for theta in angles:
                        cx = int(round(x - a * np.cos(theta)))
                        cy = int(round(y - b * np.sin(theta)))

                        if 0 <= cx < width and 0 <= cy < height:
                            accumulator[a, b, cx, cy] += 1

        # 8) Determine threshold based on max votes
        max_votes = accumulator.max()
        vote_threshold = max_votes * threshold_factor
        print("[INFO] Max Votes:", max_votes)
        print("[INFO] Vote Threshold:", vote_threshold)

        # 9) Find all (a, b, cx, cy) above threshold
        candidates = np.argwhere(accumulator > vote_threshold)

        # 10) Draw the candidate ellipses
        #    Note: We have been working on the "small_image".
        #    We'll draw on that, then scale back if scale_factor != 1.
        for (a, b, cx, cy) in candidates:
            cv2.ellipse(
                small_image,
                (cx, cy),  # center
                (a, b),  # (major_axis, minor_axis)
                0,  # rotation angle
                0, 360,  # full ellipse
                (0, 0, 255),  # red
                1
            )

        # 11) If we resized at the start, scale back up for final display
        if scale_factor != 1.0:
            # Scale the annotated image back to original size
            annotated_fullsize = cv2.resize(
                small_image,
                (original_image.shape[1], original_image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            return annotated_fullsize

        return small_image

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

    @staticmethod
    def custom_fitEllipse(points: np.ndarray,
                          confidence_level: float = 0.95,
                          min_points: int = 5) -> Optional[Tuple]:
        """
        Enhanced ellipse fitting with more robust parameter estimation

        Args:
            points (np.ndarray): 2D array of (x,y) points
            confidence_level (float): Statistical confidence for ellipse estimation
            min_points (int): Minimum number of points required for fitting

        Returns:
            Optional tuple: (center, axes lengths, angle of rotation)
        """
        # Validate input
        points = np.array(points)
        if len(points) < min_points:
            return None

        # Remove potential outliers using mask method
        def remove_outliers(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Create a mask for points within the bounds
            mask = (data >= lower_bound) & (data <= upper_bound)
            return mask

        # Create masks for x and y
        x_mask = remove_outliers(points[:, 0])
        y_mask = remove_outliers(points[:, 1])

        # Combine masks to keep points that pass both x and y outlier checks
        combined_mask = x_mask & y_mask

        # Apply mask to points
        cleaned_points = points[combined_mask]

        # Check if we have enough points after cleaning
        if len(cleaned_points) < min_points:
            return None

        # Compute center and covariance
        center = np.mean(cleaned_points, axis=0)
        centered_points = cleaned_points - center

        # Compute covariance matrix
        cov_matrix = np.cov(centered_points.T)

        # Eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        except np.linalg.LinAlgError:
            return None

        # Sort eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Chi-square value mapping for different confidence levels
        chi_square_map = {
            0.90: 4.605,
            0.95: 5.991,
            0.99: 9.210
        }
        chi_square_val = chi_square_map.get(confidence_level, 5.991)

        # Compute axis lengths
        a = 2 * np.sqrt(chi_square_val * eigenvalues[0])
        b = 2 * np.sqrt(chi_square_val * eigenvalues[1])

        # Compute angle of rotation (in degrees)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        return (tuple(center), (a, b), angle)

    @staticmethod
    def detect_and_draw_hough_ellipses(
            original_image: np.ndarray,
            min_contour_length: int = 5,
            max_ellipses: int = 10,
            threshold_factor: float = 0.95
    ) -> np.ndarray:
        """
        Advanced ellipse detection and drawing with multiple configurations

        Args:
            original_image (np.ndarray): Input image
            low_threshold (int): Lower threshold for Canny edge detection
            high_threshold (int): Higher threshold for Canny edge detection
            min_contour_length (int): Minimum points in a contour to consider
            max_ellipses (int): Maximum number of ellipses to detect
            threshold_factor (float): Confidence level for ellipse fitting

        Returns:
            Processed image with detected ellipses
        """
        # Convert to grayscale
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = CannyEdge.apply_canny(original_image, 5, 3, 50, 150, 3, True)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        processed_image = original_image.copy()

        # Sort contours by length, descending
        contours = sorted(contours, key=len, reverse=True)

        for contour in contours[:max_ellipses]:
            if len(contour) >= min_contour_length:
                contour_points = contour.reshape(-1, 2)

                # Fit ellipse with enhanced method
                ellipse_params = ShapeDetection.custom_fitEllipse(
                    contour_points,
                    confidence_level=threshold_factor
                )

                if ellipse_params:
                    (center, axes, angle) = ellipse_params
                    center_int = (int(center[0]), int(center[1]))
                    axes_int = (int(axes[0] / 2.0), int(axes[1] / 2.0))

                    # Draw ellipse with thickness based on confidence
                    thickness = 2 if threshold_factor == 0.95 else 1
                    cv2.ellipse(
                        processed_image,
                        (center_int, axes_int, angle),
                        (0, 255, 0),
                        thickness
                    )

        return processed_image
