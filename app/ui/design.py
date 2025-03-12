# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
# Created by: PyQt5 UI code generator 5.15.4
# WARNING: Any manual changes made to this file will be lost when pyuic5 is run again.

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 850)
        MainWindow.setStyleSheet("")

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color:#2c3a41;")
        self.centralwidget.setObjectName("centralwidget")

        # Layout for images
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(300, 190, 961, 511))
        self.layoutWidget.setObjectName("layoutWidget")
        self.images_layout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.images_layout.setContentsMargins(0, 0, 0, 0)
        self.images_layout.setObjectName("images_layout")

        self.original_image_groupbox = QtWidgets.QGroupBox(self.layoutWidget)
        self.original_image_groupbox.setStyleSheet("QGroupBox { color: rgb(255, 255, 255); }")
        self.original_image_groupbox.setObjectName("original_image_groupbox")
        self.images_layout.addWidget(self.original_image_groupbox)

        self.processed_image_groupbox = QtWidgets.QGroupBox(self.layoutWidget)
        self.processed_image_groupbox.setStyleSheet("QGroupBox { color: rgb(255, 255, 255); }")
        self.processed_image_groupbox.setObjectName("processed_image_groupbox")
        self.images_layout.addWidget(self.processed_image_groupbox)

        # Sidebar layout
        self.sidebar_1_widget = QtWidgets.QWidget(self.centralwidget)  # Create a parent widget for side_bar_1
        self.sidebar_1_widget.setGeometry(QtCore.QRect(20, 90, 261, 701))
        self.sidebar_1_widget.setObjectName("sidebar_1_widget")
        self.side_bar_1 = QtWidgets.QVBoxLayout(self.sidebar_1_widget)  # Assign layout to the widget
        self.side_bar_1.setContentsMargins(0, 0, 0, 0)
        self.side_bar_1.setObjectName("side_bar_1")

        # Edge Detection Button
        self.edge_detection_button = QtWidgets.QPushButton(self.sidebar_1_widget)
        self.setup_button(self.edge_detection_button, "edge_detection_button", "Edge Detection")
        self.side_bar_1.addWidget(self.edge_detection_button)

        # Object Contour Button
        self.object_contour_button = QtWidgets.QPushButton(self.sidebar_1_widget)
        self.setup_button(self.object_contour_button, "object_contour_button", "Object Contour")
        self.side_bar_1.addWidget(self.object_contour_button)

        # Back Button
        self.back_button = QtWidgets.QPushButton(self.centralwidget)
        self.setup_navigation_button(self.back_button, "back_button", "⇦", QtCore.QRect(20, 20, 50, 50))

        # Back Button
        self.quit_app_button = QtWidgets.QPushButton(self.centralwidget)
        self.setup_navigation_button(self.quit_app_button, "quit_app_button", "X", QtCore.QRect(1220, 20, 50, 50))


        # Sidebar with controls
        self.sidebar_2 = QtWidgets.QWidget(self.centralwidget)
        self.sidebar_2.setGeometry(QtCore.QRect(10, 70, 271, 731))
        self.sidebar_2.setObjectName("sidebar_2")
        self.sidebar_4 = QtWidgets.QVBoxLayout(self.sidebar_2)
        self.sidebar_4.setContentsMargins(0, 0, 0, 0)
        self.sidebar_4.setObjectName("sidebar_4")
        self.sidebar_2.hide()

        # Parametric Shape Label and ComboBox
        self.parametric_shape_label = self.setup_label("parametric_shape_label", "Parametric Shape", 16, True)
        self.sidebar_4.addWidget(self.parametric_shape_label)
        self.parametric_shape_combobox = QtWidgets.QComboBox(self.sidebar_2)
        self.setup_combobox(self.parametric_shape_combobox)
        # Add options to the ComboBox
        self.parametric_shape_combobox.addItems(["Line", "Circle", "Ellipse"])
        self.sidebar_4.addWidget(self.parametric_shape_combobox)

        # Line Threshold Slider
        self.line_threshold_label = self.setup_label("line_threshold_label", "Line Threshold", 16, True)
        self.sidebar_4.addWidget(self.line_threshold_label)
        self.line_threshold_slider = QtWidgets.QSlider(self.sidebar_2)
        self.line_threshold_slider.setOrientation(QtCore.Qt.Horizontal)
        self.sidebar_4.addWidget(self.line_threshold_slider)

        # Circle Sliders
        self.setup_circle_controls(self.sidebar_2, self.sidebar_4, "Circle", 1)
        self.setup_circle_controls(self.sidebar_2, self.sidebar_4, "Ellipse", 2)

        # Apply Button
        self.apply_button = QtWidgets.QPushButton(self.sidebar_2)
        self.setup_button(self.apply_button, "apply_button", "Apply")
        self.sidebar_4.addWidget(self.apply_button)

        # Save, Reset, Upload, and Quit Buttons
        self.setup_action_button("save_button", "Save", QtCore.QRect(410, 20, 200, 45))
        self.setup_action_button("reset", "Reset", QtCore.QRect(690, 20, 200, 45))
        self.setup_action_button("upload_button", "Upload", QtCore.QRect(970, 20, 200, 45))

        MainWindow.setCentralWidget(self.centralwidget)

    def setup_button(self, button, object_name, text):
        button.setMinimumSize(QtCore.QSize(0, 30))
        button.setMaximumSize(QtCore.QSize(300, 45))
        font = QtGui.QFont()
        font.setPointSize(12)
        button.setFont(font)
        button.setStyleSheet(
            "QPushButton { color: rgb(255, 255, 255); border: 3px solid rgb(255, 255, 255); }"
            "QPushButton:hover { border-color:rgb(5, 255, 142); color: rgb(5, 255, 142); }"
        )
        button.setObjectName(object_name)
        button.setText(text)

    def setup_navigation_button(self, button, object_name, text, geometry):
        button.setGeometry(geometry)
        button.setMaximumSize(QtCore.QSize(50, 50))
        font = QtGui.QFont()
        font.setPointSize(40)
        font.setBold(True)
        button.setFont(font)
        button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        button.setStyleSheet(
            "QPushButton { color: rgb(255, 255, 255); border: 3px solid rgb(255, 255, 255); }"
            "QPushButton:hover { border-color:rgb(253, 94, 80); color:rgb(253, 94, 80); }"
        )
        button.setObjectName(object_name)
        button.setText(text)

    def setup_action_button(self, object_name, text, geometry):
        button = QtWidgets.QPushButton(self.centralwidget)
        button.setGeometry(geometry)
        button.setMaximumSize(QtCore.QSize(230, 50))
        font = QtGui.QFont()
        font.setPointSize(12)
        button.setFont(font)
        button.setStyleSheet(
            "QPushButton { color: rgb(255, 255, 255); border: 3px solid rgb(255, 255, 255); }"
            "QPushButton:hover { border-color:rgb(5, 255, 142); color: rgb(5, 255, 142); }"
        )
        button.setObjectName(object_name)
        button.setText(text)

    def setup_label(self, object_name, text, size, bold):
        label = QtWidgets.QLabel(self.sidebar_2)
        font = QtGui.QFont()
        font.setPointSize(size)
        font.setBold(bold)
        label.setFont(font)
        label.setStyleSheet("QLabel { color: rgb(255, 255, 255); }")
        label.setObjectName(object_name)
        label.setText(text)
        return label

    def setup_combobox(self, combobox):
        combobox.setMinimumSize(QtCore.QSize(0, 25))
        combobox.setStyleSheet("QComboBox { background-color: rgb(255, 255, 255); border: 1px solid white; }")
        combobox.setObjectName("parametric_shape_combobox")

    def setup_circle_controls(self, parent, layout, name, index):
        label = self.setup_label(f"{name.lower()}_label", name, 16, True)
        layout.addWidget(label)
        for attr in ["min_radius", "max_radius", "threshold"]:
            slider_label = self.setup_label(f"{attr}_{index}", attr.replace("_", " ").title(), 12, False)
            layout.addWidget(slider_label)
            slider = QtWidgets.QSlider(parent)
            slider.setOrientation(QtCore.Qt.Horizontal)
            layout.addWidget(slider)


