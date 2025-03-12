from PyQt5 import QtWidgets
from app.ui.design import Ui_MainWindow
# from design import Ui_MainWindow
from app.utils.clean_cache import remove_directories
class MainWindowController:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)

        self.ui.quit_app_button.clicked.connect(self.closeApp)

        # Connect Back Button to show side_bar_1
        self.ui.back_button.clicked.connect(self.show_sidebar_1)

        # Connect the Edge Detection button to toggle_sidebar
        self.ui.edge_detection_button.clicked.connect(self.toggle_sidebar)

    def run(self):
        """Run the application."""
        self.MainWindow.showFullScreen()
        self.app.exec_()

    def quit_app(self):
        self.app.quit()
        remove_directories()

    def toggle_sidebar(self):
        """Show or hide Sidebar 2 when the Edge Detection button is clicked."""
        if self.ui.sidebar_2.isVisible():
            self.ui.sidebar_2.hide()
        else:
            self.ui.sidebar_2.show()

    def show_sidebar_1(self):
        """Show side_bar_1 when the back button is clicked."""
        self.ui.sidebar_1_widget.show()  # Show the parent widget
        self.ui.sidebar_2.hide()

    def closeApp(self):
        """Close the application."""
        remove_directories()
        self.app.quit()

