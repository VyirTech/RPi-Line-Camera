#Vyir
#Vyirtech.com

# required imports
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

# ignore command line warnings
import warnings
warnings.filterwarnings("ignore")

# Define a Gaussian function that takes x values, amplitude (a),
# center position (x0), and standard deviation (sigma) as input


def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# Fit a Gaussian curve to the given `data` and return the optimized
# parameters: amplitude (a), center position (x0), and standard deviation (sigma)


def fit_gaussian(data):
    x = np.arange(len(data))
    mean = np.sum(x * data) / np.sum(data)
    sigma = np.sqrt(np.sum(data * (x - mean) ** 2) / np.sum(data))
    popt, _ = curve_fit(gaussian, x, data, p0=[np.max(data), mean, sigma],maxfev=100000)
    return popt

# Calculate and return the full width at half maximum (FWHM) for
# a Gaussian curve given its standard deviation (sigma)


def full_width_half_maximum(sigma):
    return sigma * np.sqrt(8 * np.log(2))


# main GUI window definition
class Ui_MainWindow(object):
    # set camera resolution which will be passed through the whole program

    W, H = 4056, 3040

    # setup UI elements

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Beam GUI")
        # resize the window to fit conveniently on a Raspberry Pi desktop

        MainWindow.setFixedSize(1655, 1066)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        # note: Qrect input params are: x, y, wx, wy

        self.tabWidget.setGeometry(QtCore.QRect(14, 8, 1600, 1066))
        self.tabWidget.setObjectName("tabWidget")
        # first tab is for "Camera" view

        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.lineEdit = QtWidgets.QLineEdit(MainWindow)
        self.lineEdit.setGeometry(QtCore.QRect(156, 44, 539, 25))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setText("Root directory: "+os.getcwd())
        self.label = QtWidgets.QLabel(MainWindow)
        self.label.setGeometry(QtCore.QRect(122, 45, 65, 21))
        self.label.setObjectName("label")
        # first pushbutton is connected to Run, which starts the image acquisition

        self.pushButton = QtWidgets.QPushButton(MainWindow)
        self.pushButton.setGeometry(QtCore.QRect(703, 45, 55, 23))
        self.pushButton.setObjectName("pushButton")
        self.textEdit_2 = QtWidgets.QTextEdit(self.tab)
        self.textEdit_2.setGeometry(QtCore.QRect(52, 660, 801, 100))
        self.textEdit_2.setObjectName("textEdit_2")
        self.tabWidget.addTab(self.tab, "")
        # second tab is for "Beam"

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

        # widgets for displaying centroid and estimated beam width
        # labels say d4 sigma. lcd's show computed widths
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

        # widgets for adjustable aperture
        # labels say aperture x, y, radius. line edits allow for adjustable (digital) aperture
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
        self.live_chart_y.setGeometry(900, 200, 535, 300)
        self.lineEdit_savePrefix = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_savePrefix.setGeometry(QtCore.QRect(900, 50, 100, 20))
        self.lineEdit_savePrefix.setPlaceholderText("Enter prefix")
        self.plainTextEdit_smallText = QtWidgets.QPlainTextEdit(
            self.centralwidget)
        self.plainTextEdit_smallText.setGeometry(
            QtCore.QRect(1000, 700, 535, 300))
        self.plainTextEdit_smallText.setObjectName("plainTextEdit_smallText")

        # widgets for saving data

        self.pushButton_S = QtWidgets.QPushButton(MainWindow)
        self.pushButton_S.setGeometry(QtCore.QRect(765, 45, 55, 23))
        # widgets for logging data continuously

        self.pushButton_L = QtWidgets.QPushButton(MainWindow)
        self.pushButton_L.setGeometry(QtCore.QRect(827, 45, 55, 23))
        # create an image frame for raw image (downsampled)
        # the image frame hosts the "Camera" tab image
        # the beam frame hosts the "Beam" tab image + processing
        # the cb frame hosts the colorbar (manual containment to insert colorbar). Requires cb.png in root directory
        self.image_frame = QtWidgets.QLabel(self.tab)
        self.beam_frame = QtWidgets.QLabel(self.tab_2)
        self.cb_frame = QtWidgets.QLabel(self.tab_2)
        # cb.png is required to manually place a colorbar (containment for cv2 heat map and colorbar)

        colorbar = cv2.imread("cb.png")
        colorbar = cv2.cvtColor(colorbar, cv2.COLOR_RGB2BGR)
        self.cb_frame.move(790, 49)
        imGUI = QtGui.QImage(colorbar.data, colorbar.shape[1], colorbar.shape[0],
                             colorbar.shape[1]*3, QtGui.QImage.Format_RGB888)
        self.cb_frame.setPixmap(QtGui.QPixmap.fromImage(imGUI))
        # connect the pushbuttons to start image capture, calibrate power meter, save data

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

    # set text for GUI elements

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
    # run image acquisition and processing thread

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
    # save images as png (can support other filetypes if needed)
    # save statistics as csv

    def save(self):
        if self.RUNNING:
            self.threadA.SAVE_NOW = True
        else:
            self.lineEdit.setText("Run the system before saving data")


# thread which handles live image acquisition and beam image processing
# runs separately from main GUI thread to prevent hang ups
class captureThread(QThread):
    # variables which can be accessed across functions and threads
    image_live = np.empty(1)  # live camera image
    camera = None  # camera variable for PiCamera
    rawCapture = None  # rawCapture variable for PiCamera
    MainWindow = None  # MainWindow passed to thread so thread can modify UI elements
    SAVE_NOW = False  # flag to save all data once
    LOGGING = False  # flag to continuously log data
    # used to set camera and beam frame sizes and locations to draw images on
    FRAMES_INIT = False
    # number of allowed saturated pixels before considering the beam profile 'saturated'
    sat_num_allowed = 20
    # used to reset aperture values if input is left blank
    count_x, count_y, count_r = 0, 0, 0
    # mask values for digital aperture. Changes based on text input
    mask_x, mask_y, mask_r = 1296, 972, 880
    W = 0  # camera/image width to be set
    H = 0  # camera/image height to be set
    # multiply a pixel width by 1.55 micron to get physical width #SENSOR DEPENDENT
    pixel_um = 1.55

    # initialize camera and set main window for interaction between thread and MainWindow
    def __init__(self, MainWindow, W, H):
        QThread.__init__(self)
        # set the camera resolution

        self.W, self.H = W, H
        self.MainWindow = MainWindow
        self.init_camera()

    # capture live images and convert to beam profile

    def run(self):
        while (1):
            self.live_image()
            self.beam()
            self.update_live_chart()

    def update_live_chart(self):
        mask = np.zeros([self.H, self.W])
        image = cv2.cvtColor(self.image_live, cv2.COLOR_BGR2GRAY)

        image_m = np.copy(self.image_live)
        image_m[mask == 0] = 0
        MOM = cv2.moments(image)
        centroid_x = int(MOM['m10'] / MOM['m00'])
        centroid_y = int(MOM['m01'] / MOM['m00'])

        x_prof = image[round(centroid_y), :]
        y_prof = image[:, round(centroid_x)]

        # Fit Gaussian
        popt_x = fit_gaussian(x_prof)
        popt_y = fit_gaussian(y_prof)
        fitted_x = gaussian(np.arange(len(x_prof)), *popt_x)
        fitted_y = gaussian(np.arange(len(y_prof)), *popt_y)

        # Calculate FWHM
        fwhm_x = full_width_half_maximum(popt_x[2])
        fwhm_y = full_width_half_maximum(popt_y[2])

        # Update the live charts with new data
        self.update_chart(self.MainWindow.live_chart_x,
                          x_prof, fitted_x, fwhm_x)
        self.update_chart(self.MainWindow.live_chart_y,
                          y_prof, fitted_y, fwhm_y)

    def update_chart(self, chart, data, fitted_data, fwhm):
        # Clear previous plot
        ax = chart.figure.get_axes()[0]
        ax.clear()

        # Update the chart with new data
        ax.plot(range(len(data)), data, label='Data')
        ax.plot(range(len(fitted_data)), fitted_data,
                label='Fitted Gaussian', linestyle='--')
        ax.set_title(f'FWHM: {fwhm:.2f}')
        ax.set_xlim(0, len(data) - 1)
        ax.set_ylim(0, 255)
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Intensity')
        ax.legend()

        # Redraw the canvas
        chart.draw()

    # initialize camera settings

    def init_camera(self):
        # initialize the PiCamera and PiRGBArray

        camera = PiCamera()
        camera.resolution = (self.W, self.H)
        rawCapture = PiRGBArray(camera, size=(self.W, self.H))
        time.sleep(0.1)
        # configure settings for the camera
        # these settings were chosen to minimize exposure and gain for laser profiling
        camera.awb_mode = 'off'  # AWB mode
        camera.awb_gains = (3.1, 3.1)  # manual AWB gain settings
        camera.brightness = 50  # Image brightness
        camera.meter_mode = 'backlit'  # Metering mode
        camera.exposure_mode = 'off'  # Prevent auto exposure so results are consistent
        camera.exposure_compensation = 0  # Exposure compensation
        camera.shutter_speed = 10  # Minimum shutter speed
        # vflip should be set to True since the camera sensor is lensless and will not have the image inverted
        # hflip can be set to True or False depending on how you want the image to move as you physically move
        # the beam spot location on the sensor. It is ultimately an alignment convention
        camera.vflip = True
        camera.hflip = False
        camera.iso = 1
        camera.saturation = 0
        # If ZOOM_BOOL is set to True, the camera capture will be zoomed in to a Region of Interest
        # The ROI will be the full Field of View cropped by a ratio of crop_factor
        # the roi_start_x, roi_start_y are the origin (top left corner) of the ROI
        ZOOM_BOOL = False
        if ZOOM_BOOL:
            crop_factor = 0.4
            roi_start_x = (1-crop_factor)/2
            roi_start_y = (1-crop_factor)/2
            camera.zoom = (roi_start_x, roi_start_y, crop_factor, crop_factor)
        # If CAMERA_SETTINGS is set to True, the camera settings will be printed on camera init
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
        # store the camera and capture for use by other functions
        self.camera = camera
        self.rawCapture = rawCapture
        self.MainWindow.lineEdit.setText(
            "Camera initialized! Image processing system running")

    # capture an image from the camera and store to self.image_live
    def img_capture(self):
        self.camera.capture(self.rawCapture, format="bgr")
        self.image_live = self.rawCapture.array
        self.rawCapture.truncate(0)
     # take camera capture and display live on "Camera" tab

    def live_image(self):
        # time printouts can be used for runtime optimization which directly translates to framerate of images
        # A = datetime.datetime.now()

        # take a raw capture
        self.img_capture()
        # downsample the image by scale factor 6 to fit on the GUI screen
        scale = 6
        imR = cv2.resize(self.image_live, (int(
            self.W/scale), int(self.H/scale)))
        # set the image to the proper position on the window if not already done

        if not self.FRAMES_INIT:
            self.MainWindow.image_frame.move(125, 60)
            self.MainWindow.image_frame.resize(
                int(self.W/scale), int(self.H/scale))

        imBGR2RGB = cv2.cvtColor(imR, cv2.COLOR_BGR2RGB)
        imGUI = QtGui.QImage(
            imBGR2RGB.data, imBGR2RGB.shape[1], imBGR2RGB.shape[0], imBGR2RGB.shape[1]*3, QtGui.QImage.Format_RGB888)
        self.MainWindow.image_frame.setPixmap(QtGui.QPixmap.fromImage(imGUI))

        # B = datetime.datetime.now()
        # print("Live image runtime: "+str(B-A))

    # convert camera image to beam profile (rainbow map) and display on GUI
    # compute metrics of beam (centroid, D4σ)
    def beam(self):
        # time printouts can be used for runtime optimization which directly translates to framerate of images
        # A = datetime.datetime.now()

        # set the aperture mask values to those input by the user in the text boxes
        # if the text boxes are left blank for some time, they will default

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

         # define a mask depending on aperture
        mask = np.zeros([self.H, self.W])
        mask = cv2.circle(mask, (self.mask_x, self.mask_y),
                          self.mask_r, 255, -1)
        # take grayscale version of the image for intensity profiling

        image = cv2.cvtColor(self.image_live, cv2.COLOR_BGR2GRAY)
        # copy and mask the image based on the circular aperture mask

        image_m = np.copy(image)
        image_m[mask == 0] = 0
        # compute the centroid and D4σ in pixel values if image is not empty

        MOM = cv2.moments(image_m)
        if MOM['m00'] != 0:
            centroid_x = MOM['m10']/MOM['m00']
            centroid_y = MOM['m01']/MOM['m00']
            # note 1 pixel has physical dimension: pixel_um * pixel_um (= 1.55 um (micron) * 1.55 um for Raspi HQ Camera module)
            # With no scaling (lens) the physical beam widths are then d4x (px) * 1.55 um, d4y (px) * 1.55 um

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
        # take the negative of the image in grayscale space to apply cv2 rainbow map

        image_n = 255 - image
        beam = cv2.applyColorMap(image_n, cv2.COLORMAP_RAINBOW)
        # save all data if SAVE_NOW is flagged by save button, then reset the flag

        if self.SAVE_NOW:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

            save_prefix = self.MainWindow.lineEdit_savePrefix.text()
            filename6 = save_prefix + "_entered_info_" + timestamp + ".txt"

            savepath = os.path.join(os.getcwd(), "saves"+timestamp)
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            filename1 = save_prefix+"camera_"+timestamp+".png"
            filename2 = save_prefix+"beam_"+timestamp+".png"
            filename3 = save_prefix+"stats_"+timestamp+".csv"
            filename4 = save_prefix+"x_profile_"+timestamp+".png"
            filename5 = save_prefix+"y_profile_"+timestamp+".png"
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
            max_pixel = np.amax(image_m)
            min_pixel = np.amin(image_m)
            total_pixel_counts = np.sum(image_m)
            average_pixel_count = np.mean(image_m)

            statsfile.write("Max pixel value\n")
            statsfile.write(str(max_pixel) + "\n")
            statsfile.write("Min pixel value\n")
            statsfile.write(str(min_pixel) + "\n")
            statsfile.write("Total pixel counts\n")
            statsfile.write(str(total_pixel_counts) + "\n")
            statsfile.write("Average pixel count\n")
            statsfile.write(str(average_pixel_count) + "\n")

            statsfile.close()
            with open(os.path.join(savepath, filename6), 'w') as small_text_file:
                small_text_file.write(
                    self.MainWindow.plainTextEdit_smallText.toPlainText())
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
            # only stop the saving if LOGGING is not enabled
            # update info bar depending on whether logging or single save
            output_file = "pixel_data.txt"
            with open(os.path.join(savepath, output_file), "w") as f:
                # Write the header
                f.write("{:<10}{:<25}{:<25}\n".format(
                    "Pixel #", "X-axis Intensity Value", "Y-axis Intensity Value"))

                # Iterate through the pixel intensities and write them
                num_pixels = max(len(x_prof), len(y_prof))
                for i in range(num_pixels):
                    x_intensity = x_prof[i] if i < len(x_prof) else ''
                    y_intensity = y_prof[i] if i < len(y_prof) else ''
                    f.write("{:<10}{:<25}{:<25}\n".format(
                        i, x_intensity, y_intensity))

            if not self.LOGGING:
                self.MainWindow.lineEdit.setText("Data saved to: "+savepath)
                self.SAVE_NOW = False
            else:
                self.MainWindow.lineEdit.setText("Data logging to: "+savepath)
        # convert to int by rounding
        d4x, d4y, centroid_x, centroid_y = round(d4x), round(
            d4y), round(centroid_x), round(centroid_y)
        # apply centroid line tracking

        cv2.line(beam, (centroid_x, 0), (centroid_x, self.H),
                 (0, 0, 0), thickness=5)
        cv2.line(beam, (0, centroid_y), (self.W, centroid_y),
                 (0, 0, 0), thickness=5)
        # downsample the image by scale factor 6 to fit on the GUI screen

        scale = 6
        beam_R = cv2.resize(beam, (int(self.W/scale), int(self.H/scale)))
        beam_R = cv2.cvtColor(beam_R, cv2.COLOR_BGR2RGB)
        # line below is to add aperture mask circle
        # image is reduced by scale times, so center coordinate is mask_x/scale, mask_y/scale; radius is mask_r/scale
        beam_R = cv2.circle(beam_R, (round(self.mask_x/scale), round(self.mask_y/scale)),
                            int(self.mask_r/scale), (0, 0, 0), 2)
        # set the image to the proper position on the window if not already done
        if not self.FRAMES_INIT:
            self.MainWindow.beam_frame.move(125, 60)
            self.MainWindow.beam_frame.resize(
                int(self.W/scale), int(self.H/scale))
            self.FRAMES_INIT = True
        imGUI = QtGui.QImage(beam_R.data, beam_R.shape[1], beam_R.shape[0],
                             beam_R.shape[1]*3, QtGui.QImage.Format_RGB888)
        self.MainWindow.beam_frame.setPixmap(QtGui.QPixmap.fromImage(imGUI))
        # B = datetime.datetime.now()
        # print("Beam runtime: "+str(B-A))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
