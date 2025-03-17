from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
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
        # Initialize application and main window
        self.app = QtWidgets.QApplication([])
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        # Initialize attributes
        self.previous_sidebar = None
        self.original_image = None
        self.processed_image = None
        self.path = None
        self.kernel_sizes_array = [3, 5, 7]
        self.current_kernal_size = self.kernel_sizes_array[0]
        self.gaussian_kernel_sizes_array = [3, 5, 7, 9]

        # Setup UI
        self.setup_ui()

    def setup_ui(self):
        """Initialize UI components and connect signals."""
        # Show initial sidebar
        self.show_sidebar("sidebar_1")

        # Set ranges for spin boxes
        self.set_ranges()

        # Add items to combo box
        Gradient_methods_list = ["Manhattan Distance" ,"Euclidean Distance"]
        self.ui.comboBox.addItems(Gradient_methods_list)

        # Center alignment for shapes_sidebar_layout
        self.ui.verticalLayout_5.setAlignment(Qt.AlignCenter)

        # Connect button signals
        self.connect_signals()

        # Initialize kernel size buttons
        self.ui.filter_kernel_size_button.setText(f"{self.kernel_sizes_array[0]}×{self.kernel_sizes_array[0]}")
        self.ui.gaussian_filter_kernel_size_button.setText(
            f"{self.gaussian_kernel_sizes_array[0]}×{self.gaussian_kernel_sizes_array[0]}")

        # Initialize services
        self.srv = ImageServices()
        self.activeContour = ActiveContour()

    def connect_signals(self):
        """Connect UI signals to their respective slots."""
        # Sidebar navigation
        self.ui.quit_app_button.clicked.connect(self.closeApp)
        self.ui.back_button.clicked.connect(self.go_back)
        self.ui.shape_detection_button.clicked.connect(lambda: self.show_sidebar("sidebar_2"))
        self.ui.object_contour_button.clicked.connect(lambda: self.show_sidebar("sidebar_3"))
        self.ui.canny_edge_detection_button.clicked.connect(lambda: self.show_sidebar("filter_sidebar"))
        self.ui.line_detection_button.clicked.connect(lambda: self.show_groupbox(self.ui.line_groupBox))
        self.ui.circle_detection_button.clicked.connect(lambda: self.show_groupbox(self.ui.circle_groupBox))
        self.ui.ellipse_detection_button.clicked.connect(lambda: self.show_groupbox(self.ui.ellipse_groupBox))

        # Image operations
        self.ui.upload_button.clicked.connect(self.drawImage)
        self.ui.save_button.clicked.connect(lambda: self.srv.save_image(self.processed_image))
        self.ui.reset_button.clicked.connect(self.reset_images)

        # Processing buttons
        self.ui.apply_button_2.clicked.connect(self.apply_circle_detection)
        self.ui.apply_button_4.clicked.connect(self.apply_canny)
        self.ui.apply_contour_button.clicked.connect(self.apply_contour)

        # Kernel size toggling
        self.ui.filter_kernel_size_button.clicked.connect(
            lambda: self.toggle_kernel_size(self.ui.filter_kernel_size_button, self.kernel_sizes_array)
        )
        self.ui.gaussian_filter_kernel_size_button.clicked.connect(
            lambda: self.toggle_kernel_size(self.ui.gaussian_filter_kernel_size_button, self.gaussian_kernel_sizes_array)
        )

        # Threshold updates
        self.ui.edge_detection_low_threshold_spinbox.valueChanged.connect(self.update_high_threshold)
        self.ui.edge_detection_high_threshold_spinbox.valueChanged.connect(self.update_low_threshold)

    def set_ranges(self):
        """Set ranges for spin boxes."""
        self.ui.alpha_spinBox.setRange(0.0, 100.0)
        self.ui.beta_spinBox.setRange(0.0, 100.0)
        self.ui.gamma_spinBox.setRange(0.0, 100.0)

    def toggle_kernel_size(self, button, kernel_sizes_array):
        """Toggle kernel size for the given button."""
        current_size = int(button.text().split('×')[0])
        current_index = kernel_sizes_array.index(current_size)
        next_index = (current_index + 1) % len(kernel_sizes_array)
        next_size = kernel_sizes_array[next_index]
        button.setText(f"{next_size}×{next_size}")

    def update_high_threshold(self, low_threshold_value):
        """Update high threshold based on low threshold."""
        high_threshold_value = self.ui.edge_detection_high_threshold_spinbox.value()
        if low_threshold_value >= high_threshold_value:
            new_high_value = low_threshold_value + 1
            self.ui.edge_detection_high_threshold_spinbox.setValue(min(new_high_value, self.ui.edge_detection_high_threshold_spinbox.maximum()))

    def update_low_threshold(self, high_threshold_value):
        """Update low threshold based on high threshold."""
        low_threshold_value = self.ui.edge_detection_low_threshold_spinbox.value()
        if high_threshold_value <= low_threshold_value:
            new_low_value = high_threshold_value - 1
            self.ui.edge_detection_low_threshold_spinbox.setValue(max(new_low_value, self.ui.edge_detection_low_threshold_spinbox.minimum()))

    def drawImage(self):
        """Load and display the selected image."""
        self.path = self.srv.upload_image_file()
        self.original_image = cv2.imread(self.path)
        self.processed_image = self.original_image

        if self.path:
            self.srv.clear_image(self.ui.original_image_groupbox)
            self.srv.clear_image(self.ui.processed_image_groupbox)
            self.srv.set_image_in_groupbox(self.ui.original_image_groupbox, self.original_image)
            self.srv.set_image_in_groupbox(self.ui.processed_image_groupbox, self.processed_image)

    def reset_images(self):
        """Reset images to the original state."""
        if self.original_image is None:
            return

        self.srv.clear_image(self.ui.processed_image_groupbox)
        self.srv.set_image_in_groupbox(self.ui.processed_image_groupbox, self.original_image)
        self.srv.clear_image(self.ui.original_image_groupbox)
        self.srv.set_image_in_groupbox(self.ui.original_image_groupbox, self.original_image)

    def show_sidebar(self, sidebar_name):
        """Show the specified sidebar and hide others."""
        # Hide all sidebars
        self.ui.sidebar_1_layout.hide()
        self.ui.sidebar_2_layout.hide()
        self.ui.sidebar_3_layout.hide()
        self.ui.page_filter_layout.hide()
        self.ui.shapes_sidebar_layout.hide()

        # Show the specified sidebar
        if sidebar_name == "sidebar_1":
            self.ui.sidebar_1_layout.show()
            self.previous_sidebar = None
            self.ui.back_button.hide()
        elif sidebar_name == "sidebar_2":
            self.ui.sidebar_2_layout.show()
            self.previous_sidebar = "sidebar_1"
            self.ui.back_button.show()
        elif sidebar_name == "sidebar_3":
            self.ui.sidebar_3_layout.show()
            self.previous_sidebar = "sidebar_1"
            self.ui.back_button.show()
        elif sidebar_name == "filter_sidebar":
            self.ui.page_filter_layout.show()
            self.previous_sidebar = "sidebar_1"
            self.ui.back_button.show()
        elif sidebar_name == "shapes_sidebar":
            self.ui.shapes_sidebar_layout.show()
            self.previous_sidebar = "sidebar_2"
            self.ui.back_button.show()

    def show_groupbox(self, groupbox_to_show):
        """Show the specified group box and hide others."""
        # Hide all sidebars
        self.ui.sidebar_1_layout.hide()
        self.ui.sidebar_2_layout.hide()
        self.ui.sidebar_3_layout.hide()
        self.ui.page_filter_layout.hide()

        # Show shapes_sidebar_layout and the specified group box
        self.ui.shapes_sidebar_layout.show()
        self.ui.line_groupBox.hide()
        self.ui.circle_groupBox.hide()
        self.ui.ellipse_groupBox.hide()
        groupbox_to_show.show()
        self.previous_sidebar = "sidebar_2"

    def go_back(self):
        """Handle back button click."""
        if self.previous_sidebar == "sidebar_1":
            self.show_sidebar("sidebar_1")
        elif self.previous_sidebar == "sidebar_2":
            self.show_sidebar("sidebar_2")
        elif self.previous_sidebar == "sidebar_3":
            self.show_sidebar("sidebar_3")
        else:
            self.show_sidebar("sidebar_1")

    def apply_contour(self):
        """Apply active contour algorithm to the image."""
        if self.original_image is None or self.processed_image is None:
            print("No image available")
            return

        num_points = self.ui.num_of_points_spinBox.value()
        num_iterations = self.ui.num_of_itr_spinBox.value()
        alpha = self.ui.alpha_spinBox.value()
        beta = self.ui.beta_spinBox.value()
        gamma = self.ui.gamma_spinBox.value()
        radius = self.ui.circle_radius_spinBox.value()
        type = self.ui.type_spinBox.value()
        center = self.original_image.shape[0] // 2, self.original_image.shape[1] // 2
        w_line = self.ui.w_line_spinBox.value()
        w_edge = self.ui.w_edge_spinBox.value()

        image_src = np.copy(self.original_image)

        # Create initial contour
        if type == 1:
            contour_x, contour_y, WindowCoordinates = self.activeContour.create_square_contour(
                source=image_src, num_xpoints=num_points, num_ypoints=num_points)
        elif type == 2:
            contour_x, contour_y, WindowCoordinates = self.activeContour.create_ellipse_contour(
                source=image_src, num_points=num_points)
        else:
            contour_x, contour_y, WindowCoordinates = self.activeContour.create_ellipse_contour(
                source=image_src, num_points=num_points, type="circle", radius=radius)

        # Draw initial contour
        original_image_with_snake = self.draw_contour_on_image2(self.original_image, contour_x, contour_y)
        self.showImage(original_image_with_snake, self.ui.original_image_groupbox)

        # Calculate external energy
        ExternalEnergy = gamma * self.activeContour.calculate_external_energy(image_src, w_line, w_edge)

        # Iterate contour
        cont_x, cont_y = np.copy(contour_x), np.copy(contour_y)
        for iteration in range(num_iterations):
            cont_x, cont_y = self.activeContour.iterate_contour(
                source=image_src, contour_x=cont_x, contour_y=cont_y,
                external_energy=ExternalEnergy, window_coordinates=WindowCoordinates,
                alpha=alpha, beta=beta)

        # Draw final contour
        processed_image_with_snake = self.draw_contour_on_image2(self.original_image, cont_x, cont_y)
        self.showImage(processed_image_with_snake, self.ui.processed_image_groupbox)

        # Calculate and display area, perimeter, and chain code
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
        """Apply Canny edge detection to the image."""
        sigma = self.ui.gaussian_filter_sigma_horizontalSlider.value()
        low_threshold = self.ui.edge_detection_low_threshold_spinbox.value()
        high_threshold = self.ui.edge_detection_high_threshold_spinbox.value()
        gradient_method = self.ui.gradient_method_label.text()

        L2gradient = False if gradient_method == "Manhattan Distance" else True
        processed_image = CannyEdge.apply_canny(self.original_image, 3, 5, 100, 200, 3, L2gradient)
        self.showImage(processed_image, self.ui.processed_image_groupbox)

    def apply_circle_detection(self):
        """Apply circle detection to the image."""
        max_radius = self.ui.max_radius_slider.value()
        min_radius = self.ui.min_radius_slider.value()
        threshold_factor = self.ui.circle_threshold_slider()
        processed_image = DetectCircle.superimpose(self.original_image)
        self.showImage(processed_image, self.ui.processed_image_groupbox)

    def showImage(self, image, groupbox):
        """Display the image in the specified group box."""
        if image is None:
            print("Error: Processed image is None.")
            return

        self.srv.clear_image(groupbox)
        self.srv.set_image_in_groupbox(groupbox, image)

    def draw_contour_on_image2(self, image, contour_x, contour_y):
        """Draw the contour on the image."""
        image_with_contour = image.copy()
        for x, y in zip(contour_x.astype(int), contour_y.astype(int)):
            cv2.circle(image_with_contour, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
        return image_with_contour

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