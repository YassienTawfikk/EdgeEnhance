import os
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog


class ImageServices:
    def __init__(self):
        self.last_upload_folder = "static/images"
        self.last_save_folder = "/"

    def upload_image_file(self):
        """
        Opens a file dialog for the user to select an image file, and returns the path to the selected file.
        """
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                None,
                "Select Image File",
                self.last_upload_folder,
                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)",
                options=options
            )

            if file_path:
                self.last_upload_folder = os.path.dirname(file_path)
                return file_path
            else:
                print("No file was selected.")
                return None
        except Exception as e:
            raise Exception(f"An error occurred while uploading the file: {str(e)}")

    def save_image(self, image, file_format='PNG'):
        """
        Opens a file dialog for the user to choose a save location and file name, then saves the image.

        Args:
            image (numpy.ndarray): The image to be saved.
            file_format (str): The format to save the image in (default is 'PNG').
        """
        if image is None:
            print("No image to save.")
            return False

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Image",
            os.path.join(self.last_save_folder, "Untitled." + file_format.lower()),
            f"Image Files (*.{file_format.lower()});;All Files (*)",
            options=options
        )

        if file_path:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(file_path, image)
            self.last_save_folder = os.path.dirname(file_path)
            print(f"Image successfully saved to {file_path}.")
            return True
        else:
            return False
        #

    def set_image_in_groupbox(self, groupbox, image):
        if image is None:
            return

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = image.shape
            bytes_per_line = channels * width
            qimage = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        else:
            height, width = image.shape
            bytes_per_line = width
            qimage = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_Grayscale8)

        pixmap = QtGui.QPixmap.fromImage(qimage)
        label = QtWidgets.QLabel(groupbox)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        label.setMinimumSize(500, 500)
        label.setMaximumSize(groupbox.size())

        layout = groupbox.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(groupbox)
            groupbox.setLayout(layout)

        layout.addWidget(label)
        layout.setContentsMargins(0, 25, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def clear_image(self, groupbox):
        layout = groupbox.layout()
        if layout:
            self.__clear_layout(layout)

    def __clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                sub_layout = item.layout()
                if sub_layout:
                    self.__clear_layout(sub_layout)
            del item
