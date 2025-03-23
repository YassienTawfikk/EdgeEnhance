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
    def superimpose_ellipse(original_image, min_axis=20, max_axis=60, orientation_step=10, threshold=100):
        """
        Naive ellipse detection from scratch using a brute-force parameter search.

        :param original_image:    Input BGR image (will be drawn upon).
        :param min_axis:          Minimum semi-axis length to consider.
        :param max_axis:          Maximum semi-axis length to consider.
        :param orientation_step:  Step (in degrees) to sample the ellipse orientation.
        :param threshold:         Minimum number of edge points that must lie on
                                  the ellipse for it to be considered valid.
        :return:                  original_image with the best ellipse drawn (if found).
        """

        # 1) Convert to grayscale and run your custom Canny to get a binary edge map
        edges = CannyEdge.apply_canny(
            original_image,
            gaussianSize=3,  # =3
            sigma=1,  # or any appropriate sigma value
            low_threshold=50,
            high_threshold=120,
            apertureSize=3,  # =3
            L2gradient=False  # or True if you prefer
        )

        height, width = edges.shape

        # 2) Prepare a place to track the best ellipse
        best_count = 0
        best_params = None  # (center_x, center_y, major_axis, minor_axis, orientation)

        # 3) Brute-force search over ellipse parameters:
        #    center_x, center_y, major_axis, minor_axis, orientation
        #    For demonstration, we’ll just assume major_axis >= minor_axis
        #    and orientation in [0, 180) at increments of orientation_step
        for center_y in range(height):
            for center_x in range(width):

                # Skip searching if the center itself isn’t near edges
                # (optional micro-optimization)
                # if edges[center_y, center_x] == 0:
                #     continue

                for major_axis in range(min_axis, max_axis + 1):
                    for minor_axis in range(min_axis, major_axis + 1):
                        for orientation_deg in range(0, 180, orientation_step):
                            # 3A) Sample points on the candidate ellipse
                            #     We param by angle t in [0..360)
                            #     param eq in local ellipse coords:
                            #        X(t)=major_axis*cos(t), Y(t)=minor_axis*sin(t)
                            #     then rotate by orientation and translate by center

                            count_on_edges = 0
                            num_samples = 180  # how finely we sample around the ellipse
                            angle_rad = np.deg2rad(orientation_deg)
                            cos_angle = np.cos(angle_rad)
                            sin_angle = np.sin(angle_rad)

                            for step_t in range(num_samples):
                                t = 2.0 * np.pi * step_t / num_samples
                                cos_t = np.cos(t)
                                sin_t = np.sin(t)

                                # parametric in local coords
                                x_local = major_axis * cos_t
                                y_local = minor_axis * sin_t

                                # rotate by orientation
                                x_rot = x_local * cos_angle - y_local * sin_angle
                                y_rot = x_local * sin_angle + y_local * cos_angle

                                # translate by center
                                x_final = int(round(center_x + x_rot))
                                y_final = int(round(center_y + y_rot))

                                # Check if that point is within the image and is an edge
                                if 0 <= x_final < width and 0 <= y_final < height:
                                    if edges[y_final, x_final] > 0:
                                        count_on_edges += 1

                            # 3B) Check if this ellipse is better than what we’ve seen
                            if count_on_edges > best_count:
                                best_count = count_on_edges
                                best_params = (
                                    center_x,
                                    center_y,
                                    major_axis,
                                    minor_axis,
                                    orientation_deg
                                )

        # 4) If we found an ellipse with enough edge points, draw it
        if best_params is not None and best_count >= threshold:
            print(f"Detected ellipse with {best_count} votes: {best_params}")
            cx, cy, maj, min_, angle_deg = best_params

            # We'll draw it by sampling again and setting red pixels
            angle_rad = np.deg2rad(angle_deg)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            num_samples = 360

            for step_t in range(num_samples):
                t = 2.0 * np.pi * step_t / num_samples
                cos_t = np.cos(t)
                sin_t = np.sin(t)

                x_local = maj * cos_t
                y_local = min_ * sin_t

                # rotate
                x_rot = x_local * cos_angle - y_local * sin_angle
                y_rot = x_local * sin_angle + y_local * cos_angle

                # translate
                x_final = int(round(cx + x_rot))
                y_final = int(round(cy + y_rot))

                if 0 <= x_final < width and 0 <= y_final < height:
                    original_image[y_final, x_final] = (0, 0, 255)  # BGR = red

        else:
            print("No ellipse found that satisfies the threshold.")

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
    def detect_ellipses_manually(original_image, min_axis_length=10, max_axis_length=200, max_iterations=1000, sample_size=5, inlier_tolerance=0.2, inlier_ratio_threshold=0.21):
        """
        Detect ellipses in the image using a RANSAC-like approach with cv2.fitEllipse():

        1) Apply Canny to get edge pixels.
        2) Randomly sample small sets of edge points.
        3) Fit an ellipse to each set (if possible).
        4) Count how many total edge points lie on that ellipse (within a tolerance).
        5) Keep ellipses that exceed an inlier ratio threshold.

        :param original_image: Input BGR image (as read by cv2).
        :param low_threshold: Lower threshold for Canny edge detection.
        :param high_threshold: Upper threshold for Canny edge detection.
        :param min_axis_length: Minimum axis length to keep an ellipse (post-fit filter).
        :param max_axis_length: Maximum axis length to keep an ellipse (post-fit filter).
        :param max_iterations: How many random samples to try (RANSAC iterations).
        :param sample_size: How many points to sample for each fit (minimum for cv2.fitEllipse is 5).
        :param inlier_tolerance: How close a point’s “ellipse equation” value must be to 1.0 to be counted as an inlier.
        :param inlier_ratio_threshold: Fraction of total edge points that must fit the ellipse to consider it valid.
        :return: A copy of the input image with the detected ellipses drawn.
        """

        # -----------------------------
        # 1. Edge detection
        # -----------------------------
        edges = CannyEdge.apply_canny(original_image, 5, 3, 25, 100, 3, True)

        edge_points = np.argwhere(edges > 0)  # shape: (num_points, 2) in (y, x)
        total_points = len(edge_points)
        if total_points < sample_size:
            # Not enough points to fit an ellipse
            return original_image.copy()

        # For speed, you can also random-subset your edge points if there are too many:
        max_points_for_speed = 5000
        if total_points > max_points_for_speed:
            chosen_idx = np.random.choice(total_points, max_points_for_speed, replace=False)
            edge_points = edge_points[chosen_idx]
            total_points = len(edge_points)

        # -----------------------------
        # 2. RANSAC loop
        # -----------------------------
        best_ellipses = []
        best_scores = []

        for _ in range(max_iterations):
            # 2a) Randomly pick a small subset of points (at least 5 for fitEllipse)
            sample_indices = random.sample(range(total_points), sample_size)
            sample = edge_points[sample_indices]

            # OpenCV's fitEllipse requires the contour to be shaped [N,1,2],
            # and coordinates in (x, y) order.
            # Our sample is in (y, x) so we flip it.
            contour = np.array([[[pt[1], pt[0]]] for pt in sample], dtype=np.int32)

            # Attempt to fit ellipse
            try:
                ellipse = cv2.fitEllipse(contour)
            except:
                # fitEllipse may fail if the points are degenerate (collinear, etc.)
                continue

            # 2b) Filter out extremely small or large axes
            (center_x, center_y), (major_axis, minor_axis), angle_deg = ellipse
            # Ensure major_axis >= minor_axis (OpenCV doesn't guarantee which is which)
            big = max(major_axis, minor_axis)
            small = min(major_axis, minor_axis)
            if big < min_axis_length or small < min_axis_length:
                continue
            if big > max_axis_length or small > max_axis_length:
                continue

            # 2c) Score how many points from all edges fit this ellipse
            inliers = 0
            for (py, px) in edge_points:
                # Convert the ellipse from cv2's format into an equation check
                #  - center = (center_x, center_y)
                #  - axes = (major_axis, minor_axis)
                #  - rotation = angle_deg
                val = ShapeDetection.__ellipse_equation(px, py, center_x, center_y, big, small, angle_deg)
                # If val is ~1, point is near the ellipse
                if abs(val - 1.0) < inlier_tolerance:
                    inliers += 1

            inlier_ratio = inliers / total_points
            if inlier_ratio >= inlier_ratio_threshold:
                # This ellipse passes our threshold
                best_ellipses.append(ellipse)
                best_scores.append(inlier_ratio)

        # -----------------------------
        # 3. Draw the detected ellipses
        # -----------------------------
        processed_image = original_image.copy()

        # If you want to draw *all* ellipses that passed the threshold:
        for ellipse, score in zip(best_ellipses, best_scores):
            # ellipse is in the format: ((cx, cy), (major, minor), angle)
            cv2.ellipse(processed_image, ellipse, (0, 0, 255), 2)

        return processed_image

    @staticmethod
    def __ellipse_equation(x, y, cx, cy, major, minor, angle_deg):
        """
        Returns the "ellipse equation" value for point (x, y) relative to
        an ellipse described by:
            center (cx, cy),
            axes (major >= minor),
            rotation angle in degrees.

        The ellipse equation in its local frame is:
          (x'^2 / major^2) + (y'^2 / minor^2) = 1
        where x', y' is the point (x, y) rotated into the ellipse's frame.
        We return that computed value. If exactly 1, the point lies on the ellipse.
        """

        # Convert angle to radians
        theta = np.deg2rad(angle_deg)

        # Translate
        dx = x - cx
        dy = y - cy

        # Rotate to ellipse-aligned coordinates:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x_prime = dx * cos_t + dy * sin_t
        y_prime = -dx * sin_t + dy * cos_t

        # Normalize by axes
        val = (x_prime ** 2) / (major ** 2) + (y_prime ** 2) / (minor ** 2)
        return val
