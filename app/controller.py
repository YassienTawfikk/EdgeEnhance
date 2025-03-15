from PyQt5 import QtWidgets

from app.design.design import Ui_MainWindow
from app.utils.clean_cache import remove_directories
from app.services.image_service import ImageServices
from app.processing.contour import Contour
import cv2
from skimage.filters import gaussian





class MainWindowController:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        # Hide sidebar_2 and sidebar_3 initially
        self.ui.sidebar_2_layout.hide()
        self.ui.sidebar_3_layout.hide()
        self.ui.parametric_shape_combobox.addItems(["Line", "Circle", "Ellipse"])

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
        self.contour = Contour()

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
        center=(self.original_image.shape[0]//2,self.original_image.shape[1]//2)

        # Initialize the contour
        initial_snake = self.contour.initialize_contour(self.original_image,center, radius, num_points)
        # Evolve the contour
        image = gaussian(self.processed_image, sigma=1)
        final_snake = self.contour.active_contour(image, initial_snake, alpha=0.015, beta=10, gamma=0.01)

        # Update processed image with the equalized image
        # self.processed_image = equalized_image

        # Show the processed image
        processed_image_with_contour =self.draw_contour_on_image(self.processed_image,final_snake)
        original_image_with_contour =self.draw_contour_on_image(self.original_image,initial_snake)

        self.showImage(original_image_with_contour, self.ui.original_image_groupbox)
        self.showImage(processed_image_with_contour, self.ui.processed_image_groupbox)


    def showImage(self, image, groupbox):
        if image is None:
            print("Error: Processed image is None.")
            return  # Prevents crashing

        self.srv.clear_image(groupbox)
        self.srv.set_image_in_groupbox(groupbox, image)

    def draw_contour_on_image(self,image, contour):
        """ Draws the contour onto the image. """
        # Create a copy of the image to draw on
        image_with_contour = image.copy()

        # Draw the contour in red
        for i in range(len(contour)):
            next_index = (i + 1) % len(contour)
            # Draw a line segment between current point and next point
            cv2.line(image_with_contour, tuple(contour[i].astype(int)), tuple(contour[next_index].astype(int)),
                     (255, 0, 0), thickness=2)

        return image_with_contour


