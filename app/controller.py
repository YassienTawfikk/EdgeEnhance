from PyQt5 import QtWidgets

from app.design.design import Ui_MainWindow
from app.utils.clean_cache import remove_directories
from app.services.image_service import ImageServices
from app.processing.contour import Contour
import cv2


class MainWindowController:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        # Hide sidebar_2 and sidebar_3 initially
        self.ui.sidebar_2_layout.hide()
        self.ui.sidebar_3_layout.hide()
        self.ui.parametric_shape_combobox.addItems(["Line","Circle", "Ellipse" ])

        # Connect button signals
        self.ui.quit_app_button.clicked.connect(self.closeApp)
        self.ui.back_button.clicked.connect(self.show_sidebar_1)  # Connect back button
        self.ui.edge_detection_button.clicked.connect(self.show_sidebar_2)
        self.ui.object_contour_button.clicked.connect(self.show_sidebar_3)
        self.srv = ImageServices()
        self.ui.upload_button.clicked.connect(self.drawImage)
        self.ui.save_button.clicked.connect(lambda: self.srv.save_image(self.processed_image))
        self.ui.reset_button.clicked.connect(self.reset_images)
        self.ui.apply_contour_button.clicked.connect(self.apply_contour)
        self.contour=Contour()

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
        self.ui.sidebar_1_layout.show()  # Show sidebar_1

    def show_sidebar_2(self):
        """Show sidebar_2 and hide other sidebars."""
        self.ui.sidebar_1_layout.hide()  # Hide sidebar_1
        self.ui.sidebar_3_layout.hide()  # Hide sidebar_3
        self.ui.sidebar_2_layout.show()  # Show sidebar_2

    def show_sidebar_3(self):
        """Show sidebar_3 and hide other sidebars."""
        self.ui.sidebar_1_layout.hide()  # Hide sidebar_1
        self.ui.sidebar_2_layout.hide()  # Hide sidebar_2
        self.ui.sidebar_3_layout.show()  # Show sidebar_3

    def apply_contour(self):
        if self.original_image is None:
            print("No image available")
            return
        if self.processed_image is None:
            print("No image available")
            return

        num_points = self.ui.num_of_points_spin_box.value()
        num_iterations = self.ui.num_of_itr_spin_box.value()
        alpha = self.ui.alpha_spin_box.value()
        beta = self.ui.beta_spin_box.value()
        gamma = self.ui.gamma_spin_box.value()
        radius = self.ui.circle_radius_spinBox.value()
        window_size = self.ui.window_size_spin_box.value()

        # Initialize the contour
        original_snake = self.contour.initialize_contour(self.original_image, num_points, radius)
        processed_snake = self.contour.initialize_contour(self.processed_image, num_points, radius)
        # Evolve the contour
        processed_snake = self.contour.evolve_contour(processed_snake, self.processed_image, num_iterations, alpha, beta, gamma, window_size)

        # Update processed image with the equalized image
        #self.processed_image = equalized_image

        # Show the processed image
        self.showImage(self.original_image, self.ui.original_image_groupbox)
        self.showImage(self.processed_image, self.ui.processed_image_groupbox)

        # Compute chain code, area, and perimeter
        #codes = self.contour.chain_code(processed_snake)
        area, perimeter = self.contour.compute_area_perimeter(processed_snake,self.processed_image)
        self.ui.perimeter_label.setText(str(perimeter))
        self.ui.area_label.setText(str(area))
        #print("Chain Code:", codes)
        print("Area:", area)
        print("Perimeter:", perimeter)

    def showImage(self,image,groupbox):
        if image is None:
            print("Error: Processed image is None.")
            return  # Prevents crashing

        self.srv.clear_image(groupbox)
        self.srv.set_image_in_groupbox(groupbox, image)