import numpy as np
import math
from skimage.filters import gaussian
from skimage.color import rgb2gray

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Snake:
    def __init__(self):
        pass

    def calculate_internal_energy(self, point, previous_point, next_point, alpha):
        dx1 = float(point.x) - float(previous_point.x)
        dy1 = float(point.y) - float(previous_point.y)
        dx2 = float(next_point.x) - float(point.x)
        dy2 = float(next_point.y) - float(point.y)
        if dx1 * dx1 + dy1 * dy1 == 0:
            return 0
        curvature = (dx1 * dy2 - dx2 * dy1) / math.pow(dx1 * dx1 + dy1 * dy1, 1.5)
        return float(alpha * curvature)

    def calculate_external_energy(self, image, point, beta):
        gray_image = rgb2gray(image)
        return float(-beta * gray_image[point.y, point.x])  # Assuming image is a 2D numpy array

    def calculate_gradients(self, point, prev_point, gamma):
        dx = float(point.x) - float(prev_point.x)
        dy = float(point.y) - float(prev_point.y)
        return float(gamma * (dx * dx + dy * dy))

    def calculate_point_energy(self, image, point, prev_point, next_point, alpha, beta, gamma):
        internal_energy = self.calculate_internal_energy(point, prev_point, next_point, alpha)
        external_energy = self.calculate_external_energy(image, point, beta)
        gradients = self.calculate_gradients(point, prev_point, gamma)
        return float(internal_energy + external_energy + gradients)

    def snake_operation(self, image, curve, window_size, alpha, beta, gamma):
        window_index = (window_size - 1) // 2
        num_points = len(curve)
        new_curve = []

        for i in range(num_points):
            pt = curve[i]
            prev_pt = curve[(i - 1 + num_points) % num_points]
            next_pt = curve[(i + 1) % num_points]
            min_energy = float('inf')  # Initialize to infinity
            new_pt = pt

            # Explore points in window_size x window_size square
            for dx in range(-window_index, window_index + 1):
                for dy in range(-window_index, window_index + 1):
                    move_pt = Point(pt.x + dx, pt.y + dy)
                    energy = self.calculate_point_energy(image, move_pt, prev_pt, next_pt, alpha, beta, gamma)
                    if energy < min_energy:
                        min_energy = energy
                        new_pt = move_pt

            new_curve.append(new_pt)

        return new_curve

    def initialize_contours(self, center, radius, number_of_points):
        curve = []
        resolution = 360 / number_of_points
        for i in range(number_of_points):
            angle = math.radians(i * resolution)
            x = int(radius * math.cos(angle) + center.x)
            y = int(radius * math.sin(angle) + center.y)
            curve.append(Point(x, y))
        return curve

    def points_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_contour_area(self, snake_points):
        area = 0.0
        j = len(snake_points) - 1
        for i in range(len(snake_points)):
            area += (snake_points[j].x + snake_points[i].x) * (snake_points[j].y - snake_points[i].y)
            j = i
        return abs(area / 2.0)

    def calculate_contour_perimeter(self, snake_points):
        distance_sum = 0
        for i in range(len(snake_points)):
            next_point = (i + 1) % len(snake_points)
            distance = self.points_distance(snake_points[i].x, snake_points[i].y, snake_points[next_point].x,
                                            snake_points[next_point].y)
            distance_sum += distance
        return distance_sum

    def draw_contours(self, image, snake_points):
        output_image = np.copy(image)  # Clone the original image to output_image
        chain_code = []  # Clear the chain code before generating new one

        for i in range(len(snake_points)):
            pt = snake_points[i]
            # Draw circle at the point (pt)
            output_image[pt.y, pt.x] = [255, 0, 0]  # Example of marking the point red

            if i > 0:
                prev_pt = snake_points[i - 1]
                # Draw line from previous point to current point
                self.draw_line(output_image, prev_pt, pt, color=[0, 255, 0])  # Marking line green

        # Draw line from the last point back to the first point to close the contour
        self.draw_line(output_image, snake_points[0], snake_points[-1], color=[0, 255, 0])  # Marking line green

        # Generate chain code while drawing the contour
        for i in range(len(snake_points)):
            dx = snake_points[(i + 1) % len(snake_points)].x - snake_points[i].x
            dy = snake_points[(i + 1) % len(snake_points)].y - snake_points[i].y

            # Convert the relative motion to a chain code
            code = self.get_chain_code(int(dx), int(dy))
            chain_code.append(code)

        return output_image, chain_code

    def get_chain_code(self, dx, dy):
        if dx == 0 and dy == 1:
            return 0
        elif dx == 1 and dy == 1:
            return 1
        elif dx == 1 and dy == 0:
            return 2
        elif dx == 1 and dy == -1:
            return 3
        elif dx == 0 and dy == -1:
            return 4
        elif dx == -1 and dy == -1:
            return 5
        elif dx == -1 and dy == 0:
            return 6
        elif dx == -1 and dy == 1:
            return 7

    def draw_line(self, image, start, end, color):
        # Bresenham's line algorithm for drawing a line on a 2D grid
        x1, y1 = start.x, start.y
        x2, y2 = end.x, end.y
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            image[y1, x1] = color  # Draw pixel in the specified color
            if x1 == x2 and y1 == y2:
                break
            err2 = err * 2
            if err2 > -dy:
                err -= dy
                x1 += sx
            if err2 < dx:
                err += dx
                y1 += sy

    def active_contour(self, input_image, center, radius, num_of_iterations, num_of_points, window_size, alpha, beta,
                       gamma):
        # Create initial contour points
        curve = self.initialize_contours(center, radius, num_of_points)

        # Convert image to grayscale manually here as needed (not shown)
        gray_image = input_image  # Assume input_image is in grayscale for this function
        gray_image = gaussian(gray_image, sigma=1)

        # Iterate to update contour points
        for _ in range(num_of_iterations):
            curve = self.snake_operation(gray_image, curve, window_size, alpha, beta, gamma)

        output_image, chain_code = self.draw_contours(input_image, curve)
        print("Chain Code:", chain_code)

        return curve

