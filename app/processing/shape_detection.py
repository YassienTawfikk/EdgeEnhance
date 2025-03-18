import numpy as np
import cv2
from app.processing.canny_edge import CannyEdge


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
        processed_image = cv2.cvtColor(binary_edge_map, cv2.COLOR_GRAY2BGR)

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

            cv2.line(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return original_image

    @staticmethod
    def superimpose_circle(original_image, canny_high_threshold=200, max_radius=190, min_radius=0, threshold_factor=0.8):

        canny_low_threshold = canny_high_threshold / 2
        # HANTALA3 DA FEL CONTROLLER TO AVOID COMPUTING EVERY TIME
        image_edges = CannyEdge.apply_canny(original_image, 3, 0.1, canny_low_threshold, canny_high_threshold, 3, True)

        height, width = image_edges.shape
        max_radius = min(height, width) // 2
        print(max_radius)
        min_radius = 15

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
