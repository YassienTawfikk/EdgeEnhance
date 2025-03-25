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
        processed_image = original_image.copy()

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

        # set low threshold to be half the high
        canny_low_threshold = canny_high_threshold / 2

        # apply canny to extract edges
        image_edges = CannyEdge.apply_canny(original_image, 3, 0.1, canny_low_threshold, canny_high_threshold, 3, False)

        height, width = image_edges.shape
   
        # initialize the accumulator array to hold the votes
        accumulator = np.zeros((max_radius, width, height), dtype=np.uint8)

        # extract the coordinates where there are edges 
        edge_points = np.argwhere(image_edges > 0)

        # set an angle step size and convert to radiends to avoid extra computation in the loop
        angle_step = 5
        angles = np.deg2rad(np.arange(0, 360, angle_step))

        # having x,y coords, and looping through r, and through angle, we want to find a,b
        for y, x in edge_points:
            for r in range(min_radius, max_radius):
                a_vals = (x - r * np.cos(angles)).astype(int)
                b_vals = (y - r * np.sin(angles)).astype(int)
                for a, b in zip(a_vals, b_vals):
                    # ensure a and b are withing image coordinates
                    if 0 <= a < width and 0 <= b < height:
                        # increment the votes for those values of a, b and r
                        accumulator[r, a, b] += 1

        # have a dynamic threshold to only include circles above a certain percentage of max votes
        threshold = np.max(accumulator) * threshold_factor  
        print(threshold)

        # Get (r, a, b) where votes are high
        circles = np.argwhere(accumulator > threshold)  
        print(accumulator)
        print(circles)

        # draw the circles
        for r, a, b in circles:
            print(f"Detected circle at ({a}, {b}) with radius {r}")
            cv2.circle(original_image, (a, b), r, (0, 0, 255), 1)

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

    @staticmethod
    def detect_and_draw_hough_ellipses(original_image, min_contour_length=5, max_ellipses=10, threshold_factor=0.95):
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
        edges = CannyEdge.apply_canny(original_image, 3, 3, 50, 150, 5, True)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        processed_image = original_image.copy()

        # Sort contours by length, descending
        contours = sorted(contours, key=len, reverse=True)

        for contour in contours[:max_ellipses]:
            if len(contour) >= min_contour_length:
                contour_points = contour.reshape(-1, 2)

                # Fit ellipse with enhanced method
                ellipse_params = ShapeDetection.__custom_fitEllipse(contour_points, confidence_level=threshold_factor)

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

    @staticmethod
    def __custom_fitEllipse(points, confidence_level=0.95, min_points=5):
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
