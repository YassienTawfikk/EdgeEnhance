import itertools
from typing import Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import scipy.stats as st





class ActiveContour:


    def iterate_contour(self,source: np.ndarray, contour_x: np.ndarray, contour_y: np.ndarray,
                        external_energy: np.ndarray, window_coordinates: list,
                        alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:

        """
        :param source: image source
        :param contour_x: list of x coordinates of the contour
        :param contour_y: list of y coordinates of the contour
        :param alpha: factor multiplied to E_cont term in internal energy
        :param beta: factor multiplied to E_curv term in internal energy
        :param external_energy: Image Energy (E_line + E_edge)
        :param window_coordinates: array of window coordinates for each pixel
        :return:
        """

        src = np.copy(source)
        cont_x = np.copy(contour_x)
        cont_y = np.copy(contour_y)

        contour_points = len(cont_x)

        for Point in range(contour_points):
            MinEnergy = np.inf
            TotalEnergy = 0
            NewX = None
            NewY = None
            for Window in window_coordinates:
                # Create Temporary Contours With Point Shifted To A Coordinate
                CurrentX, CurrentY = np.copy(cont_x), np.copy(cont_y)
                CurrentX[Point] = CurrentX[Point] + Window[0] if CurrentX[Point] < src.shape[1] else src.shape[1] - 1
                CurrentY[Point] = CurrentY[Point] + Window[1] if CurrentY[Point] < src.shape[0] else src.shape[0] - 1

                # Calculate Energy At The New Point
                try:
                    TotalEnergy = - external_energy[CurrentY[Point], CurrentX[Point]] + self.calculate_internal_energy(
                        CurrentX,
                        CurrentY,
                        alpha,
                        beta)
                except:
                    pass

                # Save The Point If It Has The Lowest Energy In The Window
                if TotalEnergy < MinEnergy:
                    MinEnergy = TotalEnergy
                    NewX = CurrentX[Point] if CurrentX[Point] < src.shape[1] else src.shape[1] - 1
                    NewY = CurrentY[Point] if CurrentY[Point] < src.shape[0] else src.shape[0] - 1

            # Shift The Point In The Contour To It's New Location With The Lowest Energy
            cont_x[Point] = NewX
            cont_y[Point] = NewY

        return cont_x, cont_y

    def create_square_contour(self,source, num_xpoints, num_ypoints):
        """
        Create a square shape to be the initial contour
        :param source: image source
        :param num_points: number of points in the contour
        :return: list of x points coordinates, list of y points coordinates and list of window coordinates
        """
        step = 5

        # Create x points lists
        top_edge_x = np.arange(0, num_xpoints, step)
        right_edge_x = np.repeat((num_xpoints) - step, num_xpoints // step)
        bottom_edge_x = np.flip(top_edge_x)
        left_edge_x = np.repeat(0, num_xpoints // step)

        # Create y points list
        top_edge_y = np.repeat(0, num_ypoints // step)
        right_edge_y = np.arange(0, num_ypoints, step)
        bottom_edge_y = np.repeat(num_ypoints - step, num_ypoints // step)
        left_edge_y = np.flip(right_edge_y)

        # Concatenate all the lists in one array
        contour_x = np.array([top_edge_x, right_edge_x, bottom_edge_x, left_edge_x]).ravel()
        contour_y = np.array([top_edge_y, right_edge_y, bottom_edge_y, left_edge_y]).ravel()

        # Shift the shape to a specific location in the image
        # contour_x = contour_x + (source.shape[1] // 2) - 85
        contour_x = contour_x + (source.shape[1] // 2) - 95
        contour_y = contour_y + (source.shape[0] // 2) - 40

        # Create neighborhood window
        WindowCoordinates = self.GenerateWindowCoordinates(5)

        return contour_x, contour_y, WindowCoordinates

    def create_ellipse_contour(self,source, num_points,radius=117,type="ellipse"):
        """
            Represent the snake with a set of n points
            Vi = (Xi, Yi) , where i = 0, 1, ... n-1
        :param source: Image Source
        :param num_points: number of points to create the contour with
        :return: list of x coordinates, list of y coordinates and list of window coordinates
        """

        # Create x and y lists coordinates to initialize the contour
        t = np.arange(0, num_points/10 , 0.1)


        if type == "ellipse":
            # Coordinates for image05.png image
            contour_x = (source.shape[1] // 2) + 215 * np.cos(t)
            contour_y = (source.shape[0] // 2) + 115 * np.sin(t) - 10
        else:
            # Coordinates for Circles
            contour_x = (source.shape[1] // 2) + radius * np.cos(t)
            contour_y = (source.shape[0] // 2) + radius * np.sin(t)

        contour_x = contour_x.astype(int)
        contour_y = contour_y.astype(int)

        # Create neighborhood window
        WindowCoordinates = self.GenerateWindowCoordinates(5)

        return contour_x, contour_y, WindowCoordinates

    def GenerateWindowCoordinates(self,Size: int):
        """
        Generates A List of All Possible Coordinates Inside A Window of Size "Size"
        if size == 3 then the output is like this:
        WindowCoordinates = [[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, 1], [2, 2]]

        :param Size: Size of The Window
        :return Coordinates: List of All Possible Coordinates
        """

        # Generate List of All Possible Point Values Based on Size
        Points = list(range(-Size // 2 + 1, Size // 2 + 1))
        PointsList = [Points, Points]

        # Generates All Possible Coordinates Inside The Window
        Coordinates = list(itertools.product(*PointsList))
        return Coordinates

    def calculate_internal_energy(self,CurrentX, CurrentY, alpha: float, beta: float):
        """
        The internal energy is responsible for:
            1. Forcing the contour to be continuous (E_cont)
            2. Forcing the contour to be smooth     (E_curv)
            3. Deciding if the snake wants to shrink/expand

        Internal Energy Equation:
            E_internal = E_cont + E_curv

        E_cont
            alpha * ||dc/ds||^2

            - Minimizing the first derivative.
            - The contour is approximated by N points P1, P2, ..., Pn.
            - The first derivative is approximated by a finite difference:

            E_cont = | (Vi+1 - Vi) | ^ 2
            E_cont = (Xi+1 - Xi)^2 + (Yi+1 - Yi)^2

        E_curv
            beta * ||d^2c / d^2s||^2

            - Minimizing the second derivative
            - We want to penalize if the curvature is too high
            - The curvature can be approximated by the following finite difference:

            E_curv = (Xi-1 - 2Xi + Xi+1)^2 + (Yi-1 - 2Yi + Yi+1)^2

        ==============================

        Alpha and Beta
            - Small alpha make the energy function insensitive to the amount of stretch
            - Big alpha increases the internal energy of the snake as it stretches more and more

            - Small beta causes snake to allow large curvature so that snake will curve into bends in the contour
            - Big beta leads to high price for curvature so snake prefers to be smooth and not curving

        :return:
        """
        JoinedXY = np.array((CurrentX, CurrentY))
        Points = JoinedXY.T

        # Continuous  Energy
        PrevPoints = np.roll(Points, 1, axis=0)
        NextPoints = np.roll(Points, -1, axis=0)
        Displacements = Points - PrevPoints
        PointDistances = np.sqrt(Displacements[:, 0] ** 2 + Displacements[:, 1] ** 2)
        MeanDistance = np.mean(PointDistances)
        ContinuousEnergy = np.sum((PointDistances - MeanDistance) ** 2)

        # Curvature Energy
        CurvatureSeparated = PrevPoints - 2 * Points + NextPoints
        Curvature = (CurvatureSeparated[:, 0] ** 2 + CurvatureSeparated[:, 1] ** 2)
        CurvatureEnergy = np.sum(Curvature)

        return alpha * ContinuousEnergy + beta * CurvatureEnergy

    def calculate_external_energy(self,source, WLine, WEdge):
        """
        The External Energy is responsible for:
            1. Attracts the contour towards the closest image edge with dependence on the energy map.
            2. Determines whether the snake feels attracted to object boundaries

        An energy map is a function f (x, y) that we extract from the image – I(x, y):

            By given an image – I(x, y), we can build an energy map – f(x, y),
            that will attract our snake to edges on our image.

        External Energy Equation:
            E_external = w_line * E_line + w_edge * E_edge


        E_line
            I(x, y)
            Smoothing filter could be applied to I(x, y) to remove noise
            Depending on the sign of w_line the snake will be attracted either to bright lines or dark lines


        E_curv
            -|| Gradiant(I(x,y)) ||^2

        ==============================

        :param source: Image source
        :param WLine: weight of E_line term
        :param WEdge: weight of E_edge term
        :return:
        """

        src = np.copy(source)

        # convert to gray scale if not already
        if len(src.shape) > 2:
            gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        else:
            gray = src

        # Apply Gaussian Filter to smooth the image
        ELine = self.gaussian_filter(gray, 7, 7 * 7)

        # Get Gradient Magnitude & Direction
        EEdge, gradient_direction = self.sobel_edge(ELine, True)
        # EEdge *= 255 / EEdge.max()
        # EEdge = EEdge.astype("int16")

        #Depending on the sign of WLine, the contour may be attracted to bright or dark lines.
        return WLine * ELine + WEdge * EEdge[1:-1, 1:-1]

    def sobel_edge(self,source: np.ndarray, GetDirection: bool = False):
        """
            Apply Sobel Operator to detect edges
            :param source: Image to detect edges in
            :param GetDirection: Get Gradient direction in Pi Terms
            :return: edges image
        """
        # define filters
        # vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        # horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        vertical = np.flip(horizontal.T)

        def apply_kernel(source: np.ndarray, horizontal_kernel: np.ndarray, vertical_kernel: np.ndarray,
                         ReturnEdge: bool = False):
            """
                Convert image to gray scale and convolve with kernels
                :param source: Image to apply kernel to
                :param horizontal_kernel: The horizontal array of the kernel
                :param vertical_kernel: The vertical array of the kernel
                :param ReturnEdge: Return Horizontal & Vertical Edges
                :return: The result of convolution
            """
            # convert to gray scale if not already
            if len(source.shape) > 2:
                gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
            else:
                gray = source

            # convolution
            horizontal_edge = convolve2d(gray, horizontal_kernel)
            vertical_edge = convolve2d(gray, vertical_kernel)

            mag = np.sqrt(pow(horizontal_edge, 2.0) + pow(vertical_edge, 2.0))
            if ReturnEdge:
                return mag, horizontal_edge, vertical_edge
            return mag
        mag, HorizontalEdge, VerticalEdge = apply_kernel(source, horizontal, vertical, True)

        if GetDirection:
            Direction = np.arctan2(VerticalEdge, HorizontalEdge)
            return mag, Direction
        return mag



    def square_pad(self,source: np.ndarray, size_x: int, size_y: int, pad_value: int) -> np.ndarray:
        """
            Pad Image/Array to Desired Output shape
        :param source: Input Array/Image
        :param size_x: Desired width size
        :param size_y: Desired height
        :param pad_value: value to be added as a padding
        :return: Padded Square Array
        """
        src = np.copy(source)
        x, y = src.shape

        out_x = (size_x - x) // 2
        out_xx = size_x - out_x - x

        out_y = (size_y - y) // 2
        out_yy = size_y - out_y - y

        return np.pad(src, ((out_x, out_xx), (out_y, out_yy)), constant_values=pad_value)

    def gaussian_filter(self,source: np.ndarray, shape: int = 3, sigma: [int, float] = 64) -> np.ndarray:
        """
            Gaussian Low Pass Filter Implementation
        :param source: Image to Apply Filter to
        :param shape: An Integer that denotes th Kernel size if 3
                      then the kernel is (3, 3)
        :param sigma: Standard Deviation
        :return: Filtered Image
        """
        src = np.copy(source)

        def create_square_kernel(size: int, mode: str, sigma: [int, float] = None) -> np.ndarray:
            """
                Create/Calculate a square kernel for different low pass filter modes

            :param size: Kernel Size
            :param mode: Low Pass Filter Mode ['ones' -> Average Filter Mode, 'gaussian', 'median' ]
            :param sigma: Variance amount in case of 'Gaussian' mode
            :return: Square Array Kernel
            """
            if mode == 'ones':
                return np.ones((size, size))
            elif mode == 'gaussian':
                space = np.linspace(np.sqrt(sigma), -np.sqrt(sigma), size * size)
                kernel1d = np.diff(st.norm.cdf(space))
                kernel2d = np.outer(kernel1d, kernel1d)
                return kernel2d / kernel2d.sum()
        # Create a Gaussian Kernel
        kernel = create_square_kernel(shape, 'gaussian', sigma)

        def apply_kernel(source: np.ndarray, kernel: np.ndarray, mode: str) -> np.ndarray:
            """
                Calculate/Apply Convolution of two arrays, one being the kernel
                and the other is the image

            :param source: First Array
            :param kernel: Calculated Kernel
            :param mode: Convolution mode ['valid', 'same']
            :return: Convoluted Result
            """
            src = np.copy(source)

            # Check for Grayscale Image
            if len(src.shape) == 2 or src.shape[-1] == 1:
                conv = convolve2d(src, kernel, mode)
                return conv.astype('uint8')

            out = []
            # Apply Kernel using Convolution
            for channel in range(src.shape[-1]):
                conv = convolve2d(src[:, :, channel], kernel, mode)
                out.append(conv)
            return np.stack(out, -1)

        # Apply the Kernel
        out = apply_kernel(src, kernel, 'same')
        return out.astype('uint8')

    def calculate_area(self, contour_x, contour_y):
        """Calculate the area of the contour using the Shoelace formula."""
        area = 0.0
        n = len(contour_x)
        for i in range(n):
            j = (i + 1) % n  # Wrap around to the first point
            area += contour_x[i] * contour_y[j]
            area -= contour_y[i] * contour_x[j]
        area = abs(area) / 2.0
        return area

    def calculate_perimeter(self, contour_x, contour_y):
        """Calculate the perimeter of the contour."""
        perimeter = 0.0
        n = len(contour_x)
        for i in range(n):
            j = (i + 1) % n  # Wrap around to the first point
            perimeter += np.sqrt((contour_x[j] - contour_x[i]) ** 2 + (contour_y[j] - contour_y[i]) ** 2)
        return perimeter

    def calculate_chain_code(self, contour_x, contour_y):
        """Calculate the chain code for the contour."""
        chain_code = []
        directions = {
            (1, 0): 0,  # right
            (1, 1): 1,  # down-right
            (0, 1): 2,  # down
            (-1, 1): 3,  # down-left
            (-1, 0): 4,  # left
            (-1, -1): 5,  # up-left
            (0, -1): 6,  # up
            (1, -1): 7  # up-right
        }

        n = len(contour_x)
        for i in range(n):
            j = (i + 1) % n  # Wrap around to the first point
            dx = contour_x[j] - contour_x[i]
            dy = contour_y[j] - contour_y[i]
            if dx != 0 or dy != 0:
                direction = (int(np.sign(dx)), int(np.sign(dy)))
                if direction in directions:
                    chain_code.append(directions[direction])

        return chain_code
