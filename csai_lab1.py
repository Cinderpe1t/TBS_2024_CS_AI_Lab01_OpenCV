"""
    Name: Angelina Kim
    Date: Fall 2024
    Purpose: Applying different computer vision techniques from the textbook, 
            Practical Python and OpenCV, 4th ed. by Adrian Rosebrouck to see if 
            there is possibility of rip current analysis and surfer detection 
            through a series of ocean images taken 10 fps.
"""

#CS AI and Computer Vision Lab #1
#by Angelina Kim
#
#Coverage
#Ch3: Loading, diplaying, saving
#Ch4: Image basics
#Ch5: Drawing
#Ch6: Image processing
#Ch7: Histograms
#Ch8: Smoothing and blurring
#Ch9: Thresholding
#Ch10: Gradients and edge detection
#Ch11: Contours
#Extra1: SIFT feature detection
#Extra2: Farnebeck dense optical flow

from __future__ import print_function
import numpy as np
import argparse
import cv2
import time
from matplotlib import pyplot as plt

#Class to store window and image setup
class SetUp:
    def __init__(self, image_path: str, image_prefix: str, image_suffix: str, 
                 frame_start_index: int, frame_count: int):
        #Window set up
        self.frame_size_x = 1920
        self.frame_size_y = 1080
        self.frame_size_half_x = int(self.frame_size_x / 2)
        self.frame_size_half_y = int(self.frame_size_y / 2)
        self.image_size_quart_y = int(self.frame_size_y / 4)
        self.frame_delay = 1
        self.rgb_color = ("b", "g", "r")

        #Title set up
        self.xy_cord_title = (100, 75)
        self.xy_cord_index = (1300, 75)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 2
        self.font_color = (255, 255, 255)
        self.font_thickness = 2
        self.window_title = "AI lab 01"

        #Work on image set #1
        self.image_path = image_path
        self.image_prefix = image_prefix
        self.image_suffix = image_suffix
        self.frame_start_index = frame_start_index
        self.frame_count = frame_count
    
        #Histogram set up
        self.hist_height_inch = 3
        self.hist_width_inch = 6
        self.hist_dpi = 100
        self.hist_anker_x = 1300
        self.hist_anker_y = 100
        self.hist_height = self.hist_height_inch * self.hist_dpi
        self.hist_width = self.hist_width_inch * self.hist_dpi

    def image_file_name(self, frame_index: int) -> str:
        return self.image_path + self.image_prefix + \
            str(self.frame_start_index + frame_index) + self.image_suffix

#Display first screen
def show00_display_title(setup: SetUp) -> None:
    """Display title page and pause"""
    frame_name = setup.image_file_name(0)
    frame = cv2.imread(frame_name)
    frame = cv2.putText(frame, "AI / Computer Vision Lab #1", 
                        setup.xy_cord_title, setup.font, setup.font_scale, 
                        setup.font_color, setup.font_thickness)
    cv2.imshow(setup.window_title, frame)
    cv2.waitKey(0)
    return None

#Display original image set #1
def show01_display_original_image_set(setup: SetUp) -> None:
    """Display original image set"""
    for i in range(setup.frame_count):
        frame_name = setup.image_file_name(i)
        frame = cv2.imread(frame_name)
        frame = cv2.putText(frame, "Original images", setup.xy_cord_title, 
                            setup.font, setup.font_scale, setup.font_color, 
                            setup.font_thickness)
        frame = cv2.putText(frame, "Frame index: " + str(i), 
                            setup.xy_cord_index, setup.font, setup.font_scale,
                            setup.font_color, setup.font_thickness)
        cv2.imshow(setup.window_title, frame)
        cv2.waitKey(setup.frame_delay)

    cv2.waitKey(0)
    return None

#Display image RGB and gray scale intensity histogram
def show02_display_rgb_histogram(setup: SetUp) -> None:
    """Display gray and RGB intensity histograms"""
    #Set up canvas for histogram
    fig, ax = plt.subplots(figsize = (setup.hist_width_inch, 
                                      setup.hist_height_inch), 
                                      dpi = setup.hist_dpi)
    for i in range(setup.frame_count):
        frame_name = setup.image_file_name(i)
        frame = cv2.imread(frame_name)

        #gray histogram
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])
        ax.plot(hist_gray, color = 'black')
        
        #RGB histogram
        rgb_channel = cv2.split(frame)
        for (channel, color) in zip(rgb_channel, setup.rgb_color):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            ax.plot(hist, color = color)
        ax.set_xlim([0, 256])
        ax.set_ylim([0, 125000])

        #convert canvas to np, then cv2
        fig.canvas.draw()
        img_plot = np.array(fig.canvas.renderer.buffer_rgba())
        img_plot_cv2 = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
        
        #insert plot to image
        anker_y1 = setup.hist_anker_y
        anker_y2 = setup.hist_anker_y + setup.hist_height
        anker_x1 = setup.hist_anker_x
        anker_x2 = setup.hist_anker_x + setup.hist_width
        frame[ anker_y1:anker_y2, anker_x1:anker_x2, :] = img_plot_cv2

        #add description
        frame = cv2.putText(frame, "RGB and gray histogram", 
                            setup.xy_cord_title, setup.font, setup.font_scale, 
                            setup.font_color, setup.font_thickness)
        frame = cv2.putText(frame, "Frame index: " + str(i), 
                            setup.xy_cord_index, setup.font, setup.font_scale, 
                            setup.font_color, setup.font_thickness)
        
        cv2.imshow(setup.window_title, frame)
        cv2.waitKey(setup.frame_delay)
        ax.cla() #clear plot

    cv2.waitKey(0)
    #close histogram canvas
    plt.close()
    return None

#Blurring
def show03_display_blur_response(setup: SetUp) -> None:
    """Display blur response at 4 vertical windows"""
    for i in range(setup.frame_count):
        frame_name = setup.image_file_name(i)
        frame = cv2.imread(frame_name)
        
        #empty frame to collect blurred sections
        frame_collect = np.zeros((setup.frame_size_y, setup.frame_size_x, 3), 
                                 np.uint8)

        #Gaussian blur
        y025=setup.image_size_quart_y
        frame_collect[0:y025, :, :] = \
            cv2.GaussianBlur(frame[0:y025, :, :], (3, 3), 0)
        frame_collect[y025:y025*2, :, :] = \
            cv2.GaussianBlur(frame[y025:y025*2, :, :], (5, 5), 0)
        frame_collect[y025*2:y025*3, :, :] = \
            cv2.GaussianBlur(frame[y025*2:y025*3,:, :], (9, 9), 0)
        frame_collect[y025*3:y025*4, :, :] = \
            cv2.GaussianBlur(frame[y025*3:y025*4, :, :], (15, 15), 0)
        
        #Draw rectangles
        for j in range(4):
            frame_collect = \
                cv2.rectangle(frame_collect, (0, y025 * j), 
                              (setup.frame_size_x, y025 * (j + 1)), 
                              setup.font_color, setup.font_thickness)
        
        #Add description
        frame_collect = \
            cv2.putText(frame_collect, "Blurring response", setup.xy_cord_title, 
                        setup.font, setup.font_scale, setup.font_color, 
                        setup.font_thickness)
        frame_collect = \
            cv2.putText(frame_collect, "Image index: " + str(i), 
                        setup.xy_cord_index, setup.font, setup.font_scale, 
                        setup.font_color, setup.font_thickness)
        cv2.imshow(setup.window_title, frame_collect)
        cv2.waitKey(setup.frame_delay)

    cv2.waitKey(0)
    return None

#Thresholding on gray and RGB
def show04_display_threshold(setup: SetUp) -> None:
    """Display gray and RGB threshold output at 4 sub-windows"""
    for i in range(setup.frame_count):
        frame_name = setup.image_file_name(i)
        frame = cv2.imread(frame_name)

        #Gray and threshold
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_gray = \
            cv2.adaptiveThreshold(image_gray, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 3)
        
        #RGB and threshold
        rgb_channel = cv2.split(frame)
        thresh_blue = \
            cv2.adaptiveThreshold(rgb_channel[0], 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 3)
        thresh_green = \
            cv2.adaptiveThreshold(rgb_channel[1], 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 3)
        thresh_red = \
            cv2.adaptiveThreshold(rgb_channel[2], 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 3)

        #Shrink channels by half  
        thresh_gray_half = cv2.resize(thresh_gray, (0, 0), fx = 0.5, fy = 0.5)
        thresh_red_half = cv2.resize(thresh_red, (0, 0), fx = 0.5, fy = 0.5)
        thresh_green_half = cv2.resize(thresh_green, (0, 0), fx = 0.5, fy = 0.5)
        thresh_blue_half = cv2.resize(thresh_blue, (0, 0), fx = 0.5, fy = 0.5)

        frame_collect = np.zeros((setup.frame_size_y, setup.frame_size_x, 3), 
                                 np.uint8)
        frame_collect[:, :, 0] = 0
        frame_collect[:, :, 1] = 0
        frame_collect[:, :, 2] = 0
        
        #Assemble threshold results
        x05 = setup.frame_size_half_x
        x10 = setup.frame_size_x
        y05 = setup.frame_size_half_y
        y10 = setup.frame_size_y
        frame_collect[ 0:y05, 0:x05, :] = \
            cv2.cvtColor(thresh_gray_half,cv2.COLOR_GRAY2RGB) 
        frame_collect[   0:y05, x05:x10, 0] = thresh_blue_half
        frame_collect[ y05:y10,   0:x05, 1] = thresh_green_half
        frame_collect[ y05:y10, x05:x10, 2] = thresh_red_half
        
        #Add description
        frame_collect = \
            cv2.putText(frame_collect, "Gray and RGB threshold", 
                        setup.xy_cord_title, setup.font, setup.font_scale, 
                        setup.font_color, setup.font_thickness)
        frame_collect = \
            cv2.putText(frame_collect, "Image index: " + str(i), 
                        setup.xy_cord_index, setup.font, setup.font_scale, 
                        setup.font_color, setup.font_thickness)
        cv2.imshow(setup.window_title, frame_collect)
        cv2.waitKey(setup.frame_delay)

    cv2.waitKey(0)
    return None

#Gradient and edge detection - Canny edge detection
def show05_display_canny_edge(setup: SetUp) -> None:
    """Perform and display Canny edge detection"""
    for i in range(setup.frame_count):
        frame_name = setup.image_file_name(i)
        frame = cv2.imread(frame_name)
        #Canny edge detector
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_canny = cv2.Canny(frame_gray, 10, 250, 3)
        
        #Add description
        edge_canny = \
            cv2.putText(edge_canny, "Canny edge detection", setup.xy_cord_title, 
                        setup.font, setup.font_scale, setup.font_color, 
                        setup.font_thickness)
        edge_canny = \
            cv2.putText(edge_canny, "Image index: " + str(i), 
                        setup.xy_cord_index, setup.font, setup.font_scale, 
                        setup.font_color, setup.font_thickness)
        cv2.imshow(setup.window_title, edge_canny)
        cv2.waitKey(setup.frame_delay)
    
    cv2.waitKey(0)
    return None

#Laplacian gradient
def show06_display_laplacian_gradient(setup: SetUp) -> None:
    """Perform and display Laplacian gradient"""
    for i in range(setup.frame_count):
        frame_name = setup.image_file_name(i)
        frame = cv2.imread(frame_name)

        #Laplacian gradient
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gradient_laplacian = cv2.Laplacian(frame_gray, cv2.CV_64F)
        gradient_laplacian = np.uint8(np.absolute(gradient_laplacian))
        
        #Add description
        gradient_laplacian = \
            cv2.putText(gradient_laplacian, "Laplacian gradient", 
                        setup.xy_cord_title, setup.font, setup.font_scale, 
                        setup.font_color, setup.font_thickness)
        gradient_laplacian = \
            cv2.putText(gradient_laplacian, "Image index: " + str(i), 
                        setup.xy_cord_index, setup.font, setup.font_scale, 
                        setup.font_color, setup.font_thickness)
        cv2.imshow(setup.window_title, gradient_laplacian)
        cv2.waitKey(setup.frame_delay)

    cv2.waitKey(0)
    return None

#Contour detection
def show07_display_canny_contour(setup: SetUp) -> None:
    """Perform Canny edge detection and display contours"""
    for i in range(setup.frame_count):
        frame_name = setup.image_file_name(i)
        frame = cv2.imread(frame_name)

        #Canny edge and contour detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_canny = cv2.Canny(frame_gray, 10, 250, 3)
        (waves, _) = cv2.findContours(edge_canny.copy(), cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, waves, -1, (0, 0, 0), 2)    

        #Add description
        cv2.putText(frame, "Wave contour detection", setup.xy_cord_title, 
                        setup.font, setup.font_scale, setup.font_color, 
                        setup.font_thickness)
        cv2.putText(frame, "Image index: " + str(i), 
                        setup.xy_cord_index, setup.font, setup.font_scale, 
                        setup.font_color, setup.font_thickness)
        cv2.imshow(setup.window_title, frame)
        cv2.waitKey(setup.frame_delay)

    cv2.waitKey(0)
    return None

#SIFT (Scale Invariant Feature Transform) feature detection
def show08_display_sift_feature(setup: SetUp) -> None:
    """Perform and display SIFT feature detection"""
    for i in range(setup.frame_count):
        frame_name = setup.image_file_name(i)
        frame = cv2.imread(frame_name)

        #SIFT feature detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keyPoints = sift.detect(frame_gray, None)
        frame = cv2.drawKeypoints(frame_gray, keyPoints, frame, flags = 
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #Add description
        cv2.putText(frame, "SIFT feature detection", setup.xy_cord_title, 
                        setup.font, setup.font_scale, setup.font_color, 
                        setup.font_thickness)
        cv2.putText(frame, "Image index: " + str(i), 
                        setup.xy_cord_index, setup.font, setup.font_scale, 
                        setup.font_color, setup.font_thickness)
        cv2.imshow(setup.window_title, frame)
        cv2.waitKey(setup.frame_delay)
    
    cv2.waitKey(0)
    return None

#Farneback dense optical flow
def show09_display_farneback_dense_optical_flow(setup: SetUp) -> None:
    """Perform Farneback dense optical flow"""
    #Load first frame, then start from second frame
    frame_name = setup.image_file_name(0)
    frame = cv2.imread(frame_name)
    frame_gray_previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(setup.frame_count - 1):
        frame_name = setup.image_file_name(i+1)
        frame = cv2.imread(frame_name)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Compare two frames for flow detection
        opticalFlow = \
            cv2.calcOpticalFlowFarneback(frame_gray_previous, frame_gray, None, 
                                         0.5, 3, 15, 3, 5, 1.2, 0) 

        #Draw optical flow samples
        for j in range(20,1060,20):
            for k in range(20,1900,20):
                point1 = (k, j)
                point2 = (k + int(opticalFlow[j, k, 0]), j + 
                          int(opticalFlow[j, k, 1]))
                if opticalFlow[j, k, 0] < 0:
                    flow_color = (255, 0, 0)
                else:
                    flow_color = (0, 0, 255)
                cv2.line(frame, point1, point2, flow_color, 
                         setup.font_thickness)

        frame_gray_previous = frame_gray
        #Add description
        cv2.putText(frame, "Dense optical flow", setup.xy_cord_title, 
                        setup.font, setup.font_scale, setup.font_color, 
                        setup.font_thickness)
        cv2.putText(frame, "Image index: " + str(i), 
                        setup.xy_cord_index, setup.font, setup.font_scale, 
                        setup.font_color, setup.font_thickness)
        cv2.imshow(setup.window_title, frame)
        cv2.waitKey(setup.frame_delay)

    cv2.waitKey(0)
    return None

#Display image set #2
#Detect surfer and sea animal
def show10_display_surfer_detection(setup: SetUp) -> None:
    """Surfer detection with Canny detection and contour"""
    for i in range(setup.frame_count):
        frame_name = setup.image_file_name(i)
        frame = cv2.imread(frame_name)
        
        #Detect objects
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_blurred = cv2.GaussianBlur(frame_gray, (7, 7), 0)
        edge_canny = cv2.Canny(frame_gray_blurred, 10, 250, 5)
        (objects, _) = cv2.findContours(edge_canny.copy(), cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)

        #Filter out objects by size
        for obj in objects:
            minX = setup.frame_size_x
            maxX = 0
            minY = setup.frame_size_y
            maxY = 0
            for point in obj:
                if point[0][0] < minX:
                    minX = point[0][0]
                if point[0][1] < minY:
                    minY = point[0][1]
                if point[0][0] > maxX:
                    maxX = point[0][0]
                if point[0][1] > maxY:
                    maxY = point[0][1]
            rangeX = maxX - minX
            rangeY = maxY - minY
            if rangeX < 75 and rangeY < 75 and (rangeX > 5 or rangeY > 5):
                cv2.drawContours(frame, [obj], -1, (0, 0, 0), 2)
        
        frame = cv2.putText(frame, "Surfer detection", setup.xy_cord_title, 
                        setup.font, setup.font_scale, setup.font_color, 
                        setup.font_thickness)
        frame = cv2.putText(frame, "Image index: " + str(i), 
                        setup.xy_cord_index, setup.font, setup.font_scale, 
                        setup.font_color, setup.font_thickness)
        cv2.imshow(setup.window_title, frame)
        cv2.waitKey(setup.frame_delay)

    cv2.waitKey(0)
    return None

#Main setup
def main():
    #load arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-f", "--framedelay", required = False,
                            help = "OpenCV waitKey frame delay", default = 1)
    arguments = vars(arg_parser.parse_args())

    #Work on image set #1
    image_path = "./lab01_image_set_1_scout_09/"
    image_prefix = "gimbal0_"
    image_suffix = ".jpg"
    frame_start_index = 2000
    frame_count = 101
    setup = SetUp(image_path, image_prefix, image_suffix, frame_start_index, 
                  frame_count)
    setup.frame_delay = int(arguments["framedelay"])

    #Display first screen
    show00_display_title(setup)

    #Show image set #1
    show01_display_original_image_set(setup)

    #Histogram display
    show02_display_rgb_histogram(setup)

    #Incremental blurring
    show03_display_blur_response(setup)

    #Thresholding on gray and RGB channels
    show04_display_threshold(setup)

    #Canny edge detection
    show05_display_canny_edge(setup)

    #Laplacian gradient
    show06_display_laplacian_gradient(setup)

    #Canny edge and contour detection
    show07_display_canny_contour(setup)

    #SIFT (Scale Invariant Feature Transform) feature detection
    show08_display_sift_feature(setup)

    #Farneback dense optical flow
    show09_display_farneback_dense_optical_flow(setup)
    
    #Display image set #2 for object detection
    setup.image_path = "./lab01_image_set_2_scout_06/"
    setup.frame_start_index = 1500
    setup.frame_count = 101
    show01_display_original_image_set(setup)

    #Detect surfer and sea animal
    show10_display_surfer_detection(setup)

    #Terminate
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
