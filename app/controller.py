from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, center
from app.design.design2 import Ui_MainWindow
from app.utils.clean_cache import remove_directories
from app.services.image_service import ImageServices
from app.processing.canny_edge import CannyEdge
from app.processing.circle_detection import DetectCircle
import cv2
import numpy as np
from app.processing.activeContour import ActiveContour


class MainWindowController:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        # Track the previous sidebar
        self.previous_sidebar = None

        # Show sidebar_1 initially
        self.show_sidebar_1()

        # Set Ranges
        self.set_ranges_and_values()

        # Kernel sizes for regular kernel button
        self.kernel_sizes_array = [3, 5, 7]
        self.current_kernal_size = self.kernel_sizes_array[0]  # Initialize with the first kernel size

        # Kernel sizes for regular kernel button
        self.kernel_sizes_array = [3, 5, 7]
        self.ui.comboBox.addItem("Manhattan Distance")
        self.ui.comboBox.addItem("Euclidean Distance")

        # Kernel sizes for Gaussian filter kernel button
        self.gaussian_kernel_sizes_array = [3, 5, 7, 9]  # Example sizes for Gaussian filter

        # Center alignment for shapes_sidebar_layout
        self.ui.verticalLayout_5.setAlignment(Qt.AlignCenter)

        # Connect button signals
        self.ui.quit_app_button.clicked.connect(self.closeApp)
        self.ui.back_button.clicked.connect(self.go_back)  # Connect back button
        self.ui.shape_detection_button.clicked.connect(self.show_sidebar_2)
        self.ui.object_contour_button.clicked.connect(self.show_sidebar_3)
        self.ui.canny_edge_detection_button.clicked.connect(self.show_filter_sidebar)
        self.ui.line_detection_button.clicked.connect(lambda: self.show_groupbox(self.ui.line_groupBox))
        self.ui.circle_detection_button.clicked.connect(lambda: self.show_groupbox(self.ui.circle_groupBox))
        self.ui.apply_button_2.clicked.connect(self.apply_circle_detection)
        self.ui.ellipse_detection_button.clicked.connect(lambda: self.show_groupbox(self.ui.ellipse_groupBox))
        self.ui.apply_button_4.clicked.connect(self.apply_canny)
        self.srv = ImageServices()
        self.ui.upload_button.clicked.connect(self.drawImage)
        self.ui.save_button.clicked.connect(lambda: self.srv.save_image(self.processed_image))
        self.ui.reset_button.clicked.connect(self.reset_images)
        self.ui.apply_contour_button.clicked.connect(self.apply_contour)
        self.activeContour=ActiveContour()

        # Initialize kernel size button
        self.ui.filter_kernel_size_button.setText(f"{self.kernel_sizes_array[0]}×{self.kernel_sizes_array[0]}")
        self.ui.filter_kernel_size_button.clicked.connect(
            lambda: self.toggle_kernel_size(self.ui.filter_kernel_size_button, self.kernel_sizes_array)
        )

        # Initialize Gaussian filter kernel size button
        self.ui.gaussian_filter_kernel_size_button.setText(
            f"{self.gaussian_kernel_sizes_array[0]}×{self.gaussian_kernel_sizes_array[0]}")
        self.ui.gaussian_filter_kernel_size_button.clicked.connect(
            lambda: self.toggle_kernel_size(self.ui.gaussian_filter_kernel_size_button,
                                            self.gaussian_kernel_sizes_array)
        )

    def toggle_kernel_size(self, button, kernel_sizes_array):
        """
        Cycles through predefined kernel sizes and updates the button text to reflect the current selection.

        Args:
            button (QPushButton): The button whose text needs to be updated.
            kernel_sizes_array (list): The array of kernel sizes to cycle through.
        """
        # Get the current size from the button text
        current_size = int(button.text().split('×')[0])

        # Find the index of the current size in the kernel sizes array
        current_index = kernel_sizes_array.index(current_size)

        # Compute the next index; wrap around to the beginning if necessary
        next_index = (current_index + 1) % len(kernel_sizes_array)
        next_size = kernel_sizes_array[next_index]

        # Update the button text to the next kernel size
        button.setText(f"{next_size}×{next_size}")

    def set_ranges_and_values(self):
        self.ui.alpha_spinBox.setRange(0.0, 100.0)  # Set the minimum and maximum values
        self.ui.beta_spinBox.setRange(0.0, 100.0)
        self.ui.gamma_spinBox.setRange(0.0, 100.0)

    def drawImage(self):
        self.path = self.srv.upload_image_file()
        self.original_image = cv2.imread(self.path)
        self.processed_image = self.original_image

        # If user cancels file selection, path could be None
        if self.path:
            self.srv.clear_image(self.ui.original_image_groupbox)
            self.srv.clear_image(self.ui.processed_image_groupbox)
            self.srv.set_image_in_groupbox(self.ui.original_image_groupbox, self.original_image)
            self.srv.set_image_in_groupbox(self.ui.processed_image_groupbox, self.processed_image)

    def reset_images(self):
        if self.original_image is None:
            return

        self.srv.clear_image(self.ui.processed_image_groupbox)
        self.srv.set_image_in_groupbox(self.ui.processed_image_groupbox, self.original_image)
        self.srv.clear_image(self.ui.original_image_groupbox)
        self.srv.set_image_in_groupbox(self.ui.original_image_groupbox, self.original_image)

    def run(self):
        """Run the application."""
        self.MainWindow.showFullScreen()
        self.app.exec_()

    def quit_app(self):
        """Quit the application."""
        self.app.quit()
        remove_directories()

    def closeApp(self):
        """Close the application."""
        remove_directories()
        self.app.quit()

    def show_sidebar_1(self):
        """Show sidebar_1 and hide other sidebars."""
        self.ui.sidebar_2_layout.hide()  # Hide sidebar_2
        self.ui.sidebar_3_layout.hide()  # Hide sidebar_3
        self.ui.page_filter_layout.hide()
        self.ui.shapes_sidebar_layout.hide()  # Hide shapes_sidebar_layout
        self.ui.sidebar_1_layout.show()  # Show sidebar_1
        self.previous_sidebar = None  # Reset previous sidebar
        self.ui.back_button.hide()

    def show_sidebar_2(self):
        """Show sidebar_2 and hide other sidebars."""
        self.ui.sidebar_1_layout.hide()  # Hide sidebar_1
        self.ui.sidebar_3_layout.hide()  # Hide sidebar_3
        self.ui.page_filter_layout.hide()
        self.ui.shapes_sidebar_layout.hide()  # Hide shapes_sidebar_layout
        self.ui.sidebar_2_layout.show()  # Show sidebar_2
        self.previous_sidebar = "sidebar_1"  # Set previous sidebar
        self.ui.back_button.show()

    def show_sidebar_3(self):
        """Show sidebar_3 and hide other sidebars."""
        self.ui.sidebar_1_layout.hide()  # Hide sidebar_1
        self.ui.sidebar_2_layout.hide()  # Hide sidebar_2
        self.ui.page_filter_layout.hide()
        self.ui.shapes_sidebar_layout.hide()  # Hide shapes_sidebar_layout
        self.ui.sidebar_3_layout.show()  # Show sidebar_3
        self.previous_sidebar = "sidebar_1"  # Set previous sidebar
        self.ui.back_button.show()

    def show_filter_sidebar(self):
        """Show filter sidebar and hide other sidebars."""
        self.ui.sidebar_1_layout.hide()  # Hide sidebar_1
        self.ui.sidebar_2_layout.hide()  # Hide sidebar_2
        self.ui.sidebar_3_layout.hide()  # Hide sidebar_3
        self.ui.shapes_sidebar_layout.hide()  # Hide shapes_sidebar_layout
        self.ui.page_filter_layout.show()
        self.previous_sidebar = "sidebar_1"  # Set previous sidebar
        self.ui.back_button.show()

    def show_groupbox(self, groupbox_to_show):
        """
        Show shapes_sidebar_layout and the specified group box, hide other group boxes.

        Args:
            groupbox_to_show (QGroupBox): The group box to show (e.g., line_groupBox, circle_groupBox, ellipse_groupBox).
        """
        # Hide all sidebars
        self.ui.sidebar_1_layout.hide()  # Hide sidebar_1
        self.ui.sidebar_2_layout.hide()  # Hide sidebar_2
        self.ui.sidebar_3_layout.hide()  # Hide sidebar_3
        self.ui.page_filter_layout.hide()  # Hide filter sidebar

        # Show the shapes_sidebar_layout
        self.ui.shapes_sidebar_layout.show()

        # Hide all group boxes
        self.ui.line_groupBox.hide()
        self.ui.circle_groupBox.hide()
        self.ui.ellipse_groupBox.hide()

        # Show the specified group box
        groupbox_to_show.show()

        # Set previous sidebar to sidebar_2 (Shape Detection)
        self.previous_sidebar = "sidebar_2"

    def go_back(self):
        """Handle the back button click."""
        if self.previous_sidebar == "sidebar_1":
            self.show_sidebar_1()
        elif self.previous_sidebar == "sidebar_2":
            self.show_sidebar_2()
        elif self.previous_sidebar == "sidebar_3":
            self.show_sidebar_3()
        else:
            self.show_sidebar_1()  # Default to sidebar_1 if no previous sidebar is set


    def apply_contour(self):
        if self.original_image is None:
            print("No image available")
            return
        if self.processed_image is None:
            print("No image available")
            return

        num_points = self.ui.num_of_points_spinBox.value()
        num_iterations = self.ui.num_of_itr_spinBox.value()
        alpha = self.ui.alpha_spinBox.value()
        beta = self.ui.beta_spinBox.value()
        gamma = self.ui.gamma_spinBox.value()
        radius = self.ui.circle_radius_spinBox.value()
        type = self.ui.type_spinBox.value()
        center= self.original_image.shape[0]//2, self.original_image.shape[1]//2
        w_line=self.ui.w_line_spinBox.value()
        w_edge=self.ui.w_edge_spinBox.value()

        image_src = np.copy(self.original_image)

        # Create Initial Contour and display it on the GUI
        if type == 1:
           contour_x, contour_y, WindowCoordinates = self.activeContour.create_square_contour(source=image_src,
                                                                        num_xpoints=num_points,
                                                                        num_ypoints=num_points)

        elif type == 2:
            contour_x, contour_y, WindowCoordinates = self.activeContour.create_ellipse_contour(source=image_src,num_points=num_points)

        else:
            contour_x, contour_y, WindowCoordinates = self.activeContour.create_ellipse_contour(source=image_src,num_points=num_points,type="circle",radius=radius)

        # Draw contour on the processed image
        original_image_with_snake = self.draw_contour_on_image2(self.original_image, contour_x,
                                                                contour_y)
        self.showImage(original_image_with_snake, self.ui.original_image_groupbox)

        # Calculate External Energy which will be used in each iteration of greedy algorithm
        ExternalEnergy = gamma * self.activeContour.calculate_external_energy(image_src, w_line,
                                                                              w_edge)

        cont_x, cont_y = np.copy(contour_x), np.copy(contour_y)

        for iteration in range(num_iterations):
            # Start Applying Active Contour Algorithm
            cont_x, cont_y = self.activeContour.iterate_contour(source=image_src, contour_x=cont_x, contour_y=cont_y,
                                             external_energy=ExternalEnergy, window_coordinates=WindowCoordinates,
                                             alpha=alpha, beta=beta)

        # Draw contour on the processed image
        processed_image_with_snake = self.draw_contour_on_image2(self.original_image, cont_x,cont_y)
        self.showImage(processed_image_with_snake, self.ui.processed_image_groupbox)

        # After final contour points are calculated
        area = self.activeContour.calculate_area(cont_x, cont_y)
        perimeter = self.activeContour.calculate_perimeter(cont_x, cont_y)
        chain_code = self.activeContour.calculate_chain_code(cont_x, cont_y)

        self.ui.area_spinBox.clear()
        self.ui.perimeter_spinBox.clear()
        self.ui.perimeter_spinBox.setValue(perimeter)
        self.ui.area_spinBox.setValue(area)

        print(f"Area: {area}")
        print(f"Perimeter: {perimeter}")
        print(f"Chain Code: {chain_code}")

    def apply_canny(self):
        gaussianKsize=3
        sigma=self.ui.gaussian_filter_sigma_spinbox.value()
        print(f"sigma: {sigma}")  
        low_threshold=self.ui.edge_detection_low_threshold_spinbox.value()
        print(f"low threshold: {low_threshold}")  
        high_threshold=self.ui.edge_detection_high_threshold_spinbox.value() 
        print(f"high threshold: {high_threshold}")  
        # sobelKsize=3   #extract value
        gradient_method=self.ui.gradient_method_label.text() #extract value
        # print(f"l2gradient: {L2gradient}")
        if gradient_method=="Manhattan Distance":
            L2gradient=False
        else:
            L2gradient=True    
        # processed_image=CannyEdge.apply_canny(gaussianKsize,sigma,low_threshold,high_threshold,sobelKsize,L2gradient)
        processed_image=CannyEdge.apply_canny(self.original_image,3,5,100,200,3,L2gradient)
        self.showImage(processed_image,self.ui.processed_image_groupbox)

    def apply_circle_detection(self):
        max_radius=self.ui.max_radius_slider.value()
        min_radius=self.ui.min_radius_slider.value()
        threshold_factor=self.ui.circle_threshold_slider()
        print(f"max radius: {max_radius}")
        print(f"min radius: {min_radius}")
        print(f"threshold precentage: {threshold_factor}")
        processed_image=DetectCircle.superimpose(self.original_image)
        self.showImage(processed_image,self.ui.processed_image_groupbox)

    def showImage(self, image, groupbox):
        if image is None:
            print("Error: Processed image is None.")
            return  # Prevents crashing

        self.srv.clear_image(groupbox)
        self.srv.set_image_in_groupbox(groupbox, image)


    def draw_contour_on_image2(self, image, contour_x, contour_y):
        """
        Draws the contour (snake) on the image.

        Args:
            image (ndarray): The original image.
            contour_x (ndarray): The x coordinates of the contour.
            contour_y (ndarray): The y coordinates of the contour.

        Returns:
            ndarray: The image with the contour drawn on it.
        """
        # Create a copy of the original image to draw on
        image_with_contour = image.copy()

        # Draw the contour on the image
        for x, y in zip(contour_x.astype(int), contour_y.astype(int)):
            cv2.circle(image_with_contour, (x, y), radius=2, color=(0, 255, 0), thickness=-1)  # Green color for contour

        return image_with_contour
