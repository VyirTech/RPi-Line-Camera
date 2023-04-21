
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import datetime
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import math
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

import warnings
warnings.filterwarnings("ignore")

#main GUI window definitions
class Ui_MainWindow(object):

    W, H = 4056, 3040
    #set camera resolution which will be passed through the whole program
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Beam GUI")

        MainWindow.setFixedSize(1335, 1066)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)

        self.tabWidget.setGeometry(QtCore.QRect(14, 8, 1307, 1228))
        self.tabWidget.setObjectName("tabWidget")

        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.lineEdit = QtWidgets.QLineEdit(MainWindow)
        self.lineEdit.setGeometry(QtCore.QRect(156, 44, 539, 25))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText("Root directory: "+os.getcwd())
        self.label = QtWidgets.QLabel(MainWindow)
        self.label.setGeometry(QtCore.QRect(122, 45, 65, 21))
        self.label.setObjectName("label")

        self.pushButton = QtWidgets.QPushButton(MainWindow)
        self.pushButton.setGeometry(QtCore.QRect(703, 45, 55, 23))
        self.pushButton.setObjectName("pushButton")
        self.textEdit_2 = QtWidgets.QTextEdit(self.tab)
        self.textEdit_2.setGeometry(QtCore.QRect(52, 660, 801, 100))
        self.textEdit_2.setObjectName("textEdit_2")
        self.tabWidget.addTab(self.tab, "")

        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1343, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        #widgets for displaying centroid and estimated beam width
        #labels say d4 sigma. lcd's show computed widths
        self.label_centroid = QtWidgets.QLabel(self.tab_2)
        self.label_centroid.setFont(QtGui.QFont('Any', 12))
        self.label_centroid.setText("Centroid (x,y) = 0, 0")
        self.label_centroid.setGeometry(QtCore.QRect(550, 540, 250, 30))
        self.label_dx = QtWidgets.QLabel(self.tab_2)
        self.label_dx.setGeometry(QtCore.QRect(20, 230, 101, 41))
        self.lcdNumber_dx = QtWidgets.QLCDNumber(self.tab_2)
        self.lcdNumber_dx.setGeometry(QtCore.QRect(20, 260, 81, 41))
        self.lcdNumber_dx.display(0)
        self.label_dy = QtWidgets.QLabel(self.tab_2)
        self.label_dy.setGeometry(QtCore.QRect(20, 300, 101, 41))
        self.lcdNumber_dy = QtWidgets.QLCDNumber(self.tab_2)
        self.lcdNumber_dy.setGeometry(QtCore.QRect(20, 330, 81, 41))
        self.lcdNumber_dy.display(0)

        self.label_apx = QtWidgets.QLabel(self.tab_2)
        self.label_apx.setGeometry(QtCore.QRect(20, 380, 101, 41))
        self.lineEdit_apx = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_apx.setGeometry(QtCore.QRect(20, 410, 80, 40))
        self.lineEdit_apx.setText(str(int(self.W / 2)))
        self.label_apy = QtWidgets.QLabel(self.tab_2)
        self.label_apy.setGeometry(QtCore.QRect(20, 440, 101, 41))
        self.lineEdit_apy = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_apy.setGeometry(QtCore.QRect(20, 470, 80, 40))
        self.lineEdit_apy.setText(str(int(self.H / 2)))
        self.label_apr = QtWidgets.QLabel(self.tab_2)
        self.label_apr.setGeometry(QtCore.QRect(20, 500, 111, 40))
        self.lineEdit_apr = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_apr.setGeometry(QtCore.QRect(20, 530, 80, 40))
        self.lineEdit_apr.setText(str(int(self.H / 2 - 100)))
        self.live_chart_x = self.create_live_chart_x(self.tab_2)
        self.live_chart_x.setGeometry(165, 600, 535, 300)
        self.live_chart_y = self.create_live_chart_y(self.tab_2)
        self.live_chart_y.setGeometry(900, 20, 300, 535)
        self.label_shutter.setText(_translate("MainWindow", "Shutter Speed"))
        self.label_frame.setText(_translate("MainWindow", "Framerate"))

        self.pushButton_S = QtWidgets.QPushButton(MainWindow)
        self.pushButton_S.setGeometry(QtCore.QRect(765, 45, 55, 23))

        self.pushButton_L = QtWidgets.QPushButton(MainWindow)
        self.pushButton_L.setGeometry(QtCore.QRect(827, 45, 55, 23))

        # set and update the shutter speed and frame rate
        self.label_shutter = QtWidgets.QLabel(self.tab)
        self.label_shutter.setGeometry(QtCore.QRect(20, 60, 101, 41))
        self.lineEdit_shutter = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_shutter.setGeometry(QtCore.QRect(20, 90, 80, 40))
        #self.lineEdit_shutter.textChanged.connect(self.update_shutter_speed)
        self.lineEdit_shutter.setText(str(int(0)))

        self.label_frame = QtWidgets.QLabel(self.tab)
        self.label_frame.setGeometry(QtCore.QRect(20, 120, 101, 41))
        self.lineEdit_frame = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_frame.setGeometry(QtCore.QRect(20, 150, 80, 40))
        #self.lineEdit_frame.textChanged.connect(self.update_framerate)
        self.lineEdit_frame.setText(str(int(1)))

        #create an image frame for raw image (downsampled)
        #the image frame hosts the "Camera" tab image
        #the beam frame hosts the "Beam" tab image + processing
        #the cb frame hosts the colorbar (manual containment to insert colorbar). Requires cb.png in root directory
        self.image_frame = QtWidgets.QLabel(self.tab)
        self.beam_frame = QtWidgets.QLabel(self.tab_2)
        self.cb_frame = QtWidgets.QLabel(self.tab_2)

        colorbar = cv2.imread("cb.png")
        colorbar = cv2.cvtColor(colorbar, cv2.COLOR_RGB2BGR)
        self.cb_frame.move(790, 49)
        imGUI = QtGui.QImage(colorbar.data, colorbar.shape[1], colorbar.shape[0],
                             colorbar.shape[1]*3, QtGui.QImage.Format_RGB888)
        self.cb_frame.setPixmap(QtGui.QPixmap.fromImage(imGUI))

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.pushButton.clicked.connect(self.run)
        self.pushButton_S.clicked.connect(self.save)
        self.pushButton_L.clicked.connect(self.log)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def create_live_chart_x(self, parent):
        """
        Create an empty live chart in the given parent widget.
        """
        fig = Figure(figsize=(7, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.set_title('Beam profile along x-axis at y-centroid')
        canvas = FigureCanvas(fig)
        canvas.setParent(parent)
        return canvas
    
    def create_live_chart_y(self, parent):
        """
        Create an empty live chart in the given parent widget.
        """
        fig = Figure(figsize=(7, 3), dpi=80)
        ax = fig.add_subplot(111)
        ax.set_title('Beam profile along y-axis at x-centroid')
        canvas = FigureCanvas(fig)
        canvas.setParent(parent)
        return canvas
    
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Beam GUI"))
        self.label.setText(_translate("MainWindow", "Info:"))
        self.pushButton.setText(_translate("MainWindow", "Run"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(
            self.tab), _translate("MainWindow", "Camera"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(
            self.tab_2), _translate("MainWindow", "Beam"))
        self.label_dx.setText(_translate("MainWindow", "D4σx (μm)"))
        self.label_dy.setText(_translate("MainWindow", "D4σy (μm)"))
        self.label_apx.setText(_translate("MainWindow", "Aperture x"))
        self.label_apy.setText(_translate("MainWindow", "Aperture y"))
        self.label_apr.setText(_translate("MainWindow", "Ap. Radius"))
        self.pushButton_S.setText(_translate("MainWindow", "Save"))
        self.pushButton_L.setText(_translate("MainWindow", "Log"))

    RUNNING = False

    def run(self):
        if not self.RUNNING:
            self.threadA = captureThread(self, self.W, self.H)
            self.threadA.start()
            self.RUNNING = True
        else:
            self.lineEdit.setText("System already running")

    def log(self):
        if self.RUNNING:
            if not self.threadA.LOGGING:
                self.threadA.SAVE_NOW = True
                self.threadA.LOGGING = True
                self.pushButton_L.setText("Stop")
            else:
                self.threadA.SAVE_NOW = False
                self.threadA.LOGGING = False
                self.pushButton_L.setText("Log")
                self.lineEdit.setText("Data logging stopped")
        else:
            self.lineEdit.setText("Run the system before logging data")

    def save(self):
        if self.RUNNING:
            self.threadA.SAVE_NOW = True
        else:
            self.lineEdit.setText("Run the system before saving data")
def fit_gaussian(data):
    x = np.arange(len(data))
    mean = np.sum(x * data) / np.sum(data)
    standard_deviation = np.sqrt(np.sum((x - mean) ** 2 * data) / np.sum(data))
    amplitude = np.max(data)
    return curve_fit(gaussian, x, data, p0=[amplitude, mean, standard_deviation],maxfev=80000)[0]
def gaussian(x, amplitude, mean, standard_deviation):
    return amplitude * np.exp(-0.5 * ((x - mean) / standard_deviation) ** 2)
def full_width_half_maximum(standard_deviation):
    return 2 * np.sqrt(2 * np.log(2)) * standard_deviation
    


class captureThread(QThread):
    image_live = np.empty(1)
    camera = None
    rawCapture = None
    MainWindow = None
    SAVE_NOW = False
    LOGGING = False
    FRAMES_INIT = False
    sat_num_allowed = 20
    count_x, count_y, count_r = 0, 0, 0
    mask_x, mask_y, mask_r = 1296, 972, 880
    W = 0
    H = 0
    pixel_um = 1.55

    def __init__(self, MainWindow, W, H):
        QThread.__init__(self)
        self.W, self.H = W, H
        self.MainWindow = MainWindow
        self.init_camera()

    def run(self):
        while (1):
            self.live_image()
            self.beam()
            self.update_live_chart_x()
            self.update_live_chart_y()

    def update_live_chart_x(self):
        mask = np.zeros([self.H, self.W])
        image = cv2.cvtColor(self.image_live, cv2.COLOR_BGR2GRAY)

        image_m = np.copy(self.image_live)
        image_m[mask == 0] = 0
        MOM = cv2.moments(image)
        centroid_y = int(MOM['m01'] / MOM['m00'])

        x_prof = image[round(centroid_y), :]

        # Fit Gaussian
        popt_x = fit_gaussian(x_prof)
        fitted_x = gaussian(np.arange(len(x_prof)), *popt_x)

        # Calculate FWHM
        fwhm_x = full_width_half_maximum(popt_x[2])

        # Clear previous plot
        ax = self.MainWindow.live_chart_x.figure.get_axes()[0]
        ax.clear()

        # Update the live chart with new data
        ax.plot(range(len(x_prof)), x_prof, label='Data')
        ax.plot(range(len(fitted_x)), fitted_x, label='Fitted Gaussian', linestyle='--')
        ax.set_title(f'FWHM: {fwhm_x:.2f}')
        ax.set_xlim(0, len(x_prof) - 1)
        ax.set_ylim(0, 255)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Intensity')
        ax.legend()

        # Redraw the canvas
        self.MainWindow.live_chart_x.draw()

    def update_live_chart_y(self):
        mask = np.zeros([self.H, self.W])
        image = cv2.cvtColor(self.image_live, cv2.COLOR_BGR2GRAY)

        image_m = np.copy(self.image_live)
        image_m[mask == 0] = 0
        MOM = cv2.moments(image)
        centroid_x = int(MOM['m10'] / MOM['m00'])

        y_prof = image[:, round(centroid_x)]

        # Fit Gaussian
        popt_y = fit_gaussian(y_prof)
        fitted_y = gaussian(np.arange(len(y_prof)), *popt_y)

        # Calculate FWHM
        fwhm_y = full_width_half_maximum(popt_y[2])

        # Clear previous plot
        ax = self.MainWindow.live_chart_y.figure.get_axes()[0]
        ax.clear()

        # Update the live chart with new data
        ax.plot(range(len(y_prof)), y_prof, label='Data')
        ax.plot(range(len(fitted_y)), fitted_y, label='Fitted Gaussian', linestyle='--')
        ax.set_title(f'FWHM: {fwhm_y:.2f}')
        ax.set_ylim(0, len(y_prof) - 1)
        ax.set_xlim(0, 255)
        ax.set_ylabel('Pixel')
        ax.set_xlabel('Intensity')
        ax.legend()

        # Redraw the canvas
        self.MainWindow.live_chart_y.draw()

    
    

    def init_camera(self):
        camera = PiCamera()
        camera.resolution = (self.W, self.H)
        rawCapture = PiRGBArray(camera, size=(self.W, self.H))
        time.sleep(0.1)
        camera.awb_mode = 'off'
        camera.awb_gains = (3.1, 3.1)
        camera.brightness = 50
        camera.meter_mode = 'backlit'
        camera.exposure_mode = 'off'
        camera.exposure_compensation = 0
        camera.shutter_speed = 10
        camera.vflip = True
        camera.hflip = False
        camera.iso = 1
        camera.saturation = 0

        ZOOM_BOOL = False
        if ZOOM_BOOL:
            crop_factor = 0.4
            roi_start_x = (1-crop_factor)/2
            roi_start_y = (1-crop_factor)/2
            camera.zoom = (roi_start_x, roi_start_y, crop_factor, crop_factor)
        CAMERA_SETTINGS = True
        if CAMERA_SETTINGS:
            print("AWB is "+str(camera.awb_mode))
            print("AWB gain is "+str(camera.awb_gains))
            print("Brightness is "+str(camera.brightness))
            print("Aperture is "+str(camera.exposure_compensation))
            print("Shutter speed is "+str(camera.shutter_speed))
            print("Camera exposure speed is "+str(camera.exposure_speed))
            print("Iso is "+str(camera.iso))
            print("Camera digital gain is "+str(camera.digital_gain))
            print("Camera analog gain is "+str(camera.analog_gain))
            print("Camera v/h flip is "+str(camera.vflip)+", "+str(camera.hflip))
            print("Camera contrast is "+str(camera.contrast))
            print("Camera color saturation is "+str(camera.saturation))
            print("Camera meter mode is "+str(camera.meter_mode))
            if ZOOM_BOOL:
                print("Camera crop factor is "+str(crop_factor))
            else:
                print("Camera crop is disabled")
        self.camera = camera
        self.rawCapture = rawCapture
        self.MainWindow.lineEdit.setText(
            "Camera initialized! Image processing system running")

    def img_capture(self):
        self.camera.capture(self.rawCapture, format="bgr")
        self.image_live = self.rawCapture.array
        self.rawCapture.truncate(0)

    def update_shutter_speed(self, text):
        self.camera.shutter_speed = int(text)

  # def update_framerate(self, text):
  #     self.camera.framerate = int(text)

    def img_capture(self):
        self.camera.capture(self.rawCapture, format="bgr")
        self.image_live = self.rawCapture.array
        self.rawCapture.truncate(0)

    def live_image(self):

        self.img_capture()

        scale = 6
        imR = cv2.resize(self.image_live, (int(
            self.W/scale), int(self.H/scale)))
        if not self.FRAMES_INIT:
            self.MainWindow.image_frame.move(125, 60)
            self.MainWindow.image_frame.resize(
                int(self.W/scale), int(self.H/scale))
        imGUI = QtGui.QImage(
            imR.data, imR.shape[1], imR.shape[0], imR.shape[1]*3, QtGui.QImage.Format_RGB888)
        self.MainWindow.image_frame.setPixmap(QtGui.QPixmap.fromImage(imGUI))

    def beam(self):

        try:
            self.mask_x = int(self.MainWindow.lineEdit_apx.text())
            self.count_x = 0
        except:
            if self.count_x == 3:
                self.mask_x = int(self.W / 2)
                self.MainWindow.lineEdit_apx.setText(str(self.mask_x))
            self.count_x += 1
        try:
            self.mask_y = int(self.MainWindow.lineEdit_apy.text())
            self.count_y = 0
        except:
            if self.count_y == 3:
                self.mask_y = int(self.H / 2)
                self.MainWindow.lineEdit_apy.setText(str(self.mask_y))
            self.count_y += 1
        try:
            self.mask_r = int(self.MainWindow.lineEdit_apr.text())
            self.count_r = 0
        except:
            if self.count_r == 3:
                self.mask_r = int(self.H/2 - 100)
                self.MainWindow.lineEdit_apr.setText(str(self.mask_r))
            self.count_r += 1

        mask = np.zeros([self.H, self.W])
        mask = cv2.circle(mask, (self.mask_x, self.mask_y),
                          self.mask_r, 255, -1)

        image = cv2.cvtColor(self.image_live, cv2.COLOR_BGR2GRAY)

        image_m = np.copy(image)
        image_m[mask == 0] = 0

        MOM = cv2.moments(image_m)
        if MOM['m00'] != 0:
            centroid_x = MOM['m10']/MOM['m00']
            centroid_y = MOM['m01']/MOM['m00']

            d4x = self.pixel_um*4 * \
                math.sqrt(abs(MOM['m20']/MOM['m00'] - centroid_x**2))
            d4y = self.pixel_um*4 * \
                math.sqrt(abs(MOM['m02']/MOM['m00'] - centroid_y**2))
        else:
            centroid_x = self.mask_x
            centroid_y = self.mask_y
            d4x = 0
            d4y = 0
        
  

        self.MainWindow.label_centroid.setText(
            "Centroid x,y: "+str(round(centroid_x))+", "+str(round(centroid_y)))
        self.MainWindow.lcdNumber_dx.display(round(d4x))
        self.MainWindow.lcdNumber_dy.display(round(d4y))

        image_n = 255 - image
        beam = cv2.applyColorMap(image_n, cv2.COLORMAP_RAINBOW)

        if self.SAVE_NOW:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            savepath = os.path.join(os.getcwd(), "saves"+timestamp)
            if not os.path.exists(savepath):
                os.mkdir(savepath)
            
            filename1 = "camera_"+timestamp+".png"
            filename2 = "beam_"+timestamp+".png"
            filename3 = "stats_"+timestamp+".csv"
            filename4 = "x_profile_"+timestamp+".png"
            filename5 = "y_profile_"+timestamp+".png"
            cv2.imwrite(os.path.join(savepath, filename1), self.image_live)
            cv2.imwrite(os.path.join(savepath, filename2), beam)
            statsfile = open(os.path.join(savepath, filename3), 'w')
            statsfile.write("Image width (px), height (px)\n")
            statsfile.write(str(self.W)+","+str(self.H)+"\n")
            statsfile.write("Centroid x (px), y (px)\n")
            statsfile.write(str(centroid_x)+","+str(centroid_y)+"\n")
            statsfile.write("D4σ x, y\n")
            statsfile.write(str(d4x)+","+str(d4y)+"\n")
            statsfile.write("Aperture x (px), y (px), radius (px)\n")
            statsfile.write(str(self.mask_x)+"," +
                            str(self.mask_y)+","+str(self.mask_r)+"\n")
            statsfile.close()
            x_prof = image[round(centroid_y), :]
            plt.plot(range(len(x_prof)), x_prof)
            plt.title('Beam profile along x-axis at y-centroid')
            plt.xlim(0, len(x_prof)-1)
            plt.ylim(0, 255)
            plt.xlabel('Pixel')
            plt.ylabel('Intensity')
            plt.savefig(os.path.join(savepath, filename4))
            plt.close('all')
            y_prof = image[:, round(centroid_x)]
            plt.plot(range(len(y_prof)), y_prof)
            plt.title('Beam profile along y-axis at x-centroid')
            plt.xlim(0, len(y_prof)-1)
            plt.ylim(0, 255)
            plt.xlabel('Pixel')
            plt.ylabel('Intensity')
            plt.savefig(os.path.join(savepath, filename5))
            plt.close('all')
            output_file="pixel_data.txt"
            with open(os.path.join(savepath,output_file), "w") as f:
                # Write the header
                f.write("{:<10}{:<25}{:<25}\n".format("Pixel #", "X-axis Intensity Value", "Y-axis Intensity Value"))

                # Iterate through the pixel intensities and write them
                num_pixels = max(len(x_prof), len(y_prof))
                for i in range(num_pixels):
                    x_intensity = x_prof[i] if i < len(x_prof) else ''
                    y_intensity = y_prof[i] if i < len(y_prof) else ''
                    f.write("{:<10}{:<25}{:<25}\n".format(i, x_intensity, y_intensity))

            if not self.LOGGING:
                self.MainWindow.lineEdit.setText("Data saved to: "+savepath)
                self.SAVE_NOW = False
            else:
                self.MainWindow.lineEdit.setText("Data logging to: "+savepath)

        d4x, d4y, centroid_x, centroid_y = round(d4x), round(
            d4y), round(centroid_x), round(centroid_y)

        cv2.line(beam, (centroid_x, 0), (centroid_x, self.H),
                 (0, 0, 0), thickness=5)
        cv2.line(beam, (0, centroid_y), (self.W, centroid_y),
                 (0, 0, 0), thickness=5)
        scale = 6
        beam_R = cv2.resize(beam, (int(self.W/scale), int(self.H/scale)))
        beam_R = cv2.cvtColor(beam_R, cv2.COLOR_BGR2RGB)
        beam_R = cv2.circle(beam_R, (round(self.mask_x/scale), round(self.mask_y/scale)),
                            int(self.mask_r/scale), (0, 0, 0), 2)
        if not self.FRAMES_INIT:
            self.MainWindow.beam_frame.move(125, 60)
            self.MainWindow.beam_frame.resize(
                int(self.W/scale), int(self.H/scale))
            self.FRAMES_INIT = True
        imGUI = QtGui.QImage(beam_R.data, beam_R.shape[1], beam_R.shape[0],
                             beam_R.shape[1]*3, QtGui.QImage.Format_RGB888)
        self.MainWindow.beam_frame.setPixmap(QtGui.QPixmap.fromImage(imGUI))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
