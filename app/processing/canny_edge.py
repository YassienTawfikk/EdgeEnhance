import numpy as np
import cv2


class CannyEdge:

    @staticmethod
    def apply_canny(image, gaussianSize=3, sigma=0.1, low_threshold=100, high_threshold=200, apertureSize=3, L2gradient=False):
        """Full Canny Edge Detector pipeline."""

        # grayscaling the image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 1. Apply Gaussian filter
        filtered_image = CannyEdge.__apply_gaussian_filter(image, gaussianSize, sigma)

        # 2. Compute gradients
        gradient_magnitude, gradient_direction = CannyEdge.__apply_sobel(filtered_image, apertureSize, L2gradient)

        # 3. Apply Non-Maximum Suppression
        thinner_edges = CannyEdge.__non_maximum_suppression(gradient_magnitude, gradient_direction)

        # 4. Apply Double Thresholding
        thresholded_image = CannyEdge.__double_threshold(thinner_edges, low_threshold, high_threshold)

        # 5. Edge Tracking by Hysteresis
        final_edges = CannyEdge.__edge_tracking_hysteresis(thresholded_image)

        return final_edges
    
    def apply_canny_built_in(image, low_threshold=100, high_threshold=200, apertureSize=3, L2gradient=False):
        return cv2.Canny(image, low_threshold, high_threshold, apertureSize=apertureSize, L2gradient=L2gradient)

    def __non_maximum_suppression(gradient_magnitude, gradient_direction):
        """Thin edges by suppressing non-maximum pixels."""

        # define height and width dimensions and initialize output image array with zeros
        h, w = gradient_magnitude.shape
        output = np.zeros((h, w), dtype=np.uint8)

        # approximate gradient direction to 4 possible angles (0, 45, 90, 135)
        angle = np.round(gradient_direction / 45) * 45 % 180

        # loop over the image 
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                q, r = 255, 255  # Default values to store neighboring pixels

                # Determine neighboring pixels based on gradient direction
                if angle[i, j] == 0:
                    q, r = gradient_magnitude[i, j + 1], gradient_magnitude[i, j - 1]
                elif angle[i, j] == 45:
                    q, r = gradient_magnitude[i + 1, j - 1], gradient_magnitude[i - 1, j + 1]
                elif angle[i, j] == 90:
                    q, r = gradient_magnitude[i + 1, j], gradient_magnitude[i - 1, j]
                elif angle[i, j] == 135:
                    q, r = gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]

                # Keep only local maxima
                if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                    output[i, j] = gradient_magnitude[i, j]

        return output

    def __double_threshold(image, low_threshold, high_threshold):
        """Classifies edges into strong, weak, and non-edges."""
        # define the values for the high and low thresholds 
        strong_edges = 255
        weak_edges = 75

        # create arrays of booleans for the strong and weak edges that will act as a mask
        strong = image >= high_threshold
        weak = (image >= low_threshold) & (image < high_threshold)

        # initialize a zero array for the output 
        output = np.zeros_like(image, dtype=np.uint8)

        # set the strong and weak edges based on the mask
        output[strong] = strong_edges
        output[weak] = weak_edges

        return output

    def __edge_tracking_hysteresis(image):
        """Traces strong edges and retains weak edges if connected to a strong edge."""
        h, w = image.shape
        strong_edges = 255
        weak_edges = 75

        # loop over the rows and cols
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if image[i, j] == weak_edges:
                    # If weak edge has a strong edge neighbor, keep it and raise to 255
                    if (strong_edges in image[i - 1:i + 2, j - 1:j + 2]):
                        image[i, j] = strong_edges
                    else:
                        image[i, j] = 0  # Remove isolated weak edges
        return image

    def __convolve_gaussian(image, kernel, kernel_size, median=False):
        # calculate the padding size
        pad_size = kernel_size // 2

        # initialize the filtered image array and pad the image
        filtered_image = np.zeros_like(image, dtype=np.float32)
        padded_image = np.pad(image, pad_size, mode='reflect')

        # loop over the rows and columns to convolve the image with the kernel
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # region extraction
                region = padded_image[i:i + kernel_size, j:j + kernel_size]

                # for average and gaussian filtering, return a pixel value of the summation of the kernel multiplied by the region
                if median == False:
                    filtered_image[i, j] = np.sum(region * kernel)
                # for median filtering return a pixel value that the median of the region
                if median == True:
                    filtered_image[i, j] = np.median(region)

        # clip the filtered image values to [0-255] and cast to uint8
        return np.clip(filtered_image, 0, 255).astype(np.uint8)

    def __gaussian_kernel(kernel_size, sigma):
        """Generates a 2D Gaussian kernel."""
        # Create a 1D array of equally spaced points centered around 0
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)

        # Create two 2D grids (xx and yy) from the 1D array ax,
        # so that xx is the horizontal kernel and yy is the vertical kernel
        xx, yy = np.meshgrid(ax, ax)

        # Compute the Gaussian function for each point in the grid
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))

        # normalize the kernal to that its sum=1 and return it
        return kernel / np.sum(kernel)

    def __apply_gaussian_filter(image, kernel_size=3, sigma=1):
        # generate a gaussian kernel 
        kernel = CannyEdge.__gaussian_kernel(kernel_size, sigma)
        # convolve the image with the kernel
        output = CannyEdge.__convolve_gaussian(image, kernel, kernel_size)
        return output

    def __convolve_sobel(image, kernel):
        flipped_kernel = np.flipud(np.fliplr(kernel))  # Flip both vertically and horizontally
        # determine kernel height and width based on the kernel passed to the function
        kernel_height, kernel_width = flipped_kernel.shape

        # determine size of image padding. if type is Roberts padding size is always 1
        pad_h, pad_w = kernel_height // 2, kernel_width // 2

        # add constant zero padding around the image
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        # initialize the output image of an array of zeroes with the same shape as image
        output = np.zeros_like(image, dtype=np.float32)

        # loop over the rows and columns of the image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # extract a region of the size of the kernel
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                # return a pixel value that the summation of the mernel values multiplied by the region values
                output[i, j] = np.sum(region * flipped_kernel)

        # return the convolved output
        return output

    def __apply_sobel(image, size=3, L2gradient=False):
        # create sobel type kernels for horizontal and vertical edge detection
        sobel_x = CannyEdge.__generate_sobel_kernel(size, 'x')
        sobel_y = CannyEdge.__generate_sobel_kernel(size, 'y')

        # get the gradient (convolution) of each kernel over the image
        grad_x = CannyEdge.__convolve_sobel(image, sobel_x)
        grad_y = CannyEdge.__convolve_sobel(image, sobel_y)

        # Choose gradient magnitude calculation method
        if L2gradient:
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)  # L2 norm (Euclidean)
        else:
            magnitude = np.abs(grad_x) + np.abs(grad_y)  # L1 norm (Manhattan)

        # Normalize to uint8
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        # Compute gradient direction (in degrees)
        direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)  # Convert to degrees

        # Normalize angles to [0, 180]
        direction = (direction + 180) % 180

        return magnitude, direction

    def __generate_sobel_kernel(size, axis):
        # Define the center coordinates and initialize the kernel
        k = size // 2  
        kernel = np.zeros((size, size), dtype=np.int32)

        # loop over the rows and cols
        for i in range(size):
            for j in range(size):
                # this equation makes sure that as i get farther in columns the value decreases, and 
                # as i get farther in colums my value decreases, and (j-k) ensures my center column is zeroed
                kernel[i, j] = (j - k) * (k + 1 - abs(i - k))

        # transpose the kernel for vertical edge detection
        if axis == 'y':
            return kernel.T  
        return kernel
