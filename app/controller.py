from PyQt5 import QtWidgets
from app.ui.design import Ui_MainWindow
from app.utils.clean_cache import remove_directories
from app.services.image_service import ImageServices
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