#Ch3: Loading, diplaying, saving
#Ch4: Image basics -> image data manipulation
#Ch5: Drawing -> draw optical flow, rip current channel
#Ch6: Image processing -> image processing arithmatics and logical operations
#Ch7: Histograms -> display RGB channel distribution
#Ch8: Smoothing and blurring -> try smoothing for object detection
#Ch9: Thresholding -> try wave feature extraction with thresholding
#Ch10: Gradients and edge detection -> gradient for information density weight calculation
#Ch11: Contours -> try detect wave front
#Extra: detect object
#Extra: track object
#Extra: optical flow between two images

#Part 1: optical flow demonstration
#Ocean image display
#Ocean image characteristics
#Wave front detection
#Optical flow
#Information weighted optical flow
#Rip current channel analysis

#Part 2: object detection and tracking
#object detection
#object tracking

from __future__ import print_function
import numpy as np
import argparse
import cv2
import time
from matplotlib import pyplot as plt

#Display first screen
def show00_display_title(window_setup, image_data_setup, title_setup):
    frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, \
        image_size_quart_y, frame_delay, rgb_color = window_setup
    image_path, image_prefix, image_suffix, frame_start_index, frame_count \
        = image_data_setup
    xy_cord_title, xy_cord_index, font, font_scale, font_color, \
        font_thickness, window_title = title_setup

    frame_index = frame_start_index
    frame_name = image_path + image_prefix + str(frame_index) + image_suffix
    frame = cv2.imread(frame_name)
    frame = cv2.putText(frame, "AI / Computer Vision Lab #1", xy_cord_title, 
                        font, font_scale, font_color, font_thickness)
    cv2.imshow(window_title, frame)
    cv2.waitKey(0)

#Display original image set #1
def show01_display_original_image_set(window_setup, image_data_setup, 
                                      title_setup):
    frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, \
        image_size_quart_y, frame_delay, rgb_color = window_setup
    image_path, image_prefix, image_suffix, frame_start_index, frame_count \
        = image_data_setup
    xy_cord_title, xy_cord_index, font, font_scale, font_color, \
        font_thickness, window_title = title_setup
    
    for i in range(frame_count):
        frame_index = frame_start_index + i
        frame_name = image_path + image_prefix + str(frame_index) + image_suffix
        frame = cv2.imread(frame_name)
        frame = cv2.putText(frame, "Original images", xy_cord_title, font, 
                            font_scale, font_color, font_thickness)
        frame = cv2.putText(frame, "Frame index: " + str(frame_index), 
                            xy_cord_index, font, font_scale, font_color, 
                            font_thickness)
        cv2.imshow(window_title, frame)
        cv2.waitKey(frame_delay)

    cv2.waitKey(0)

#Display image RGB and gray scale intensity histogram
def show02_display_rgb_histogram(window_setup, image_data_setup, title_setup, 
                                 hist_dimension):
    frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, \
        image_size_quart_y, frame_delay, rgb_color = window_setup
    image_path, image_prefix, image_suffix, frame_start_index, frame_count \
        = image_data_setup
    xy_cord_title, xy_cord_index, font, font_scale, font_color, \
        font_thickness, window_title = title_setup
    #Histogram variables
    hist_height_inch, hist_width_inch, hist_dpi, hist_anker_x, hist_anker_y \
        = hist_dimension
    hist_height = hist_height_inch * hist_dpi
    hist_width = hist_width_inch * hist_dpi

    #Set up canvas for histogram
    fig, ax = plt.subplots(figsize = (hist_width_inch, hist_height_inch), 
                           dpi = hist_dpi)
    for i in range(frame_count):
        frame_index = frame_start_index + i
        frame_name = image_path + image_prefix + str(frame_index) + image_suffix
        frame = cv2.imread(frame_name)

        #gray histogram
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])
        ax.plot(hist_gray, color = 'black')
        
        #RGB histogram
        rgb_channel = cv2.split(frame)
        for (channel, color) in zip(rgb_channel, rgb_color):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            ax.plot(hist, color = color)
        ax.set_xlim([0, 256])
        ax.set_ylim([0, 125000])

        #convert canvas to np, then cv2
        fig.canvas.draw()
        img_plot = np.array(fig.canvas.renderer.buffer_rgba())
        img_plot_cv2 = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
        
        #insert plot to image
        frame[ hist_anker_y:hist_anker_y + hist_height, 
              hist_anker_x:hist_anker_x + hist_width, :] = img_plot_cv2

        #add description
        frame = cv2.putText(frame, "RGB and gray histogram", xy_cord_title, 
                            font, font_scale, font_color, font_thickness)
        frame = cv2.putText(frame, "Frame index: " + str(frame_index), 
                            xy_cord_index, font, font_scale, font_color, 
                            font_thickness)
        
        cv2.imshow(window_title, frame)
        cv2.waitKey(frame_delay)
        ax.cla() #clear plot

    cv2.waitKey(0)
    #close histogram canvas
    plt.close()

#Blurring
def show03_display_blur_response(window_setup, image_data_setup, title_setup):
    frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, \
        image_size_quart_y, frame_delay, rgb_color = window_setup
    image_path, image_prefix, image_suffix, frame_start_index, frame_count \
        = image_data_setup
    xy_cord_title, xy_cord_index, font, font_scale, font_color, \
        font_thickness, window_title = title_setup
    
    for i in range(frame_count):
        frame_index = frame_start_index + i
        frame_name = image_path + image_prefix + str(frame_index) + image_suffix
        frame = cv2.imread(frame_name)
        
        #empty frame to collect blurred sections
        frame_collect = np.zeros((frame_size_y, frame_size_x, 3), np.uint8)

        #Gaussian blur
        frame_collect[0:image_size_quart_y,:,:] = \
            cv2.GaussianBlur(frame[0:image_size_quart_y,:,:], (3, 3), 0)
        frame_collect[image_size_quart_y:image_size_quart_y*2,:,:] = \
            cv2.GaussianBlur(frame[image_size_quart_y:image_size_quart_y*2,:,:], 
                             (5, 5), 0)
        frame_collect[image_size_quart_y*2:image_size_quart_y*3,:,:] = \
            cv2.GaussianBlur(frame[image_size_quart_y*2:image_size_quart_y*3,:,
                                   :], (9, 9), 0)
        frame_collect[image_size_quart_y*3:image_size_quart_y*4,:,:] = \
            cv2.GaussianBlur(frame[image_size_quart_y*3:image_size_quart_y*4,:,
                                   :], (15, 15), 0)
        
        #Add description
        for i in range(4):
            frame_collect = \
                cv2.rectangle(frame_collect, (0, image_size_quart_y*i), 
                              (frame_size_x, image_size_quart_y*(i+1)), 
                              font_color, font_thickness)
        frame_collect = \
            cv2.putText(frame_collect, "Blurring response", xy_cord_title, font, 
                        font_scale, font_color, font_thickness)
        frame_collect = \
            cv2.putText(frame_collect, "Image index: " + str(frame_index), 
                        xy_cord_index, font, font_scale, font_color, 
                        font_thickness)
        cv2.imshow(window_title, frame_collect)
        cv2.waitKey(frame_delay)

    cv2.waitKey(0)


#Thresholding on gray and RGB
def show04_display_threshold(window_setup, image_data_setup, title_setup):
    frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, \
        image_size_quart_y, frame_delay, rgb_color = window_setup
    image_path, image_prefix, image_suffix, frame_start_index, frame_count \
        = image_data_setup
    xy_cord_title, xy_cord_index, font, font_scale, font_color, \
        font_thickness, window_title = title_setup
    
    for i in range(frame_count):
        frame_index = frame_start_index + i
        frame_name = image_path + image_prefix + str(frame_index) + image_suffix
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

        frame_collect = np.zeros((frame_size_y, frame_size_x, 3), np.uint8)
        frame_collect[:, :, 0] = 0
        frame_collect[:, :, 1] = 0
        frame_collect[:, :, 2] = 0
        
        #Assemble threshold results
        frame_collect[ 0:frame_size_half_y, 0:frame_size_half_x, :] = \
            cv2.cvtColor(thresh_gray_half,cv2.COLOR_GRAY2RGB) 
        frame_collect[ 0:frame_size_half_y, 
                      frame_size_half_x:frame_size_x, 0] = thresh_blue_half
        frame_collect[ frame_size_half_y:frame_size_y, 
                      0:frame_size_half_x, 1] = thresh_green_half
        frame_collect[ frame_size_half_y:frame_size_y, 
                      frame_size_half_x:frame_size_x, 2] = thresh_red_half
        
        #Add description
        frame_collect = \
            cv2.putText(frame_collect, "Gray and RGB threshold", xy_cord_title, 
                        font, font_scale, font_color, font_thickness)
        frame_collect = \
            cv2.putText(frame_collect, "Image index: " + str(frame_index), 
                        xy_cord_index, font, font_scale, font_color, 
                        font_thickness)
        cv2.imshow(window_title, frame_collect)
        cv2.waitKey(frame_delay)

    cv2.waitKey(0)

#Gradient and edge detection - Canny edge detection
def show05_display_canny_edge(window_setup, image_data_setup, title_setup):
    frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, \
        image_size_quart_y, frame_delay, rgb_color = window_setup
    image_path, image_prefix, image_suffix, frame_start_index, frame_count \
        = image_data_setup
    xy_cord_title, xy_cord_index, font, font_scale, font_color, \
        font_thickness, window_title = title_setup
    
    for i in range(frame_count):
        frame_index = frame_start_index + i
        frame_name = image_path + image_prefix + str(frame_index) + image_suffix
        frame = cv2.imread(frame_name)

        #Canny edge detector
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_canny = cv2.Canny(frame_gray, 10, 250, 3)
        
        #Add description
        edge_canny = \
            cv2.putText(edge_canny, "Canny edge detection", xy_cord_title, font, 
                        font_scale, font_color, font_thickness)
        edge_canny = \
            cv2.putText(edge_canny, "Image index: " + str(frame_index), 
                        xy_cord_index, font, font_scale, font_color, 
                        font_thickness)
        cv2.imshow(window_title, edge_canny)
        cv2.waitKey(frame_delay)
    
    cv2.waitKey(0)

#Laplacian gradient
def show06_display_laplacian_gradient(window_setup, image_data_setup, 
                                      title_setup):
    frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, \
        image_size_quart_y, frame_delay, rgb_color = window_setup
    image_path, image_prefix, image_suffix, frame_start_index, frame_count \
        = image_data_setup
    xy_cord_title, xy_cord_index, font, font_scale, font_color, \
        font_thickness, window_title = title_setup
    
    for i in range(frame_count):
        frame_index = frame_start_index + i
        frame_name = image_path + image_prefix + str(frame_index) + image_suffix
        frame = cv2.imread(frame_name)

        #Laplacian gradient
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gradient_laplacian = cv2.Laplacian(frame_gray, cv2.CV_64F)
        gradient_laplacian = np.uint8(np.absolute(gradient_laplacian))
        
        #Add description
        gradient_laplacian = \
            cv2.putText(gradient_laplacian, "Laplacian gradient", xy_cord_title, 
                        font, font_scale, font_color, font_thickness)
        gradient_laplacian = \
            cv2.putText(gradient_laplacian, "Image index: " + str(frame_index), 
                        xy_cord_index, font, font_scale, font_color, 
                        font_thickness)
        cv2.imshow(window_title, gradient_laplacian)
        cv2.waitKey(frame_delay)

    cv2.waitKey(0)


#Contour detection
def show07_display_canny_contour(window_setup, image_data_setup, title_setup):
    frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, \
        image_size_quart_y, frame_delay, rgb_color = window_setup
    image_path, image_prefix, image_suffix, frame_start_index, frame_count \
        = image_data_setup
    xy_cord_title, xy_cord_index, font, font_scale, font_color, \
        font_thickness, window_title = title_setup
    
    for i in range(frame_count):
        frame_index = frame_start_index + i
        frame_name = image_path + image_prefix + str(frame_index) + image_suffix
        frame = cv2.imread(frame_name)

        #Canny edge and contour detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_canny = cv2.Canny(frame_gray, 10, 250, 3)
        (waves, _) = cv2.findContours(edge_canny.copy(), cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, waves, -1, (0, 0, 0), 2)    

        #Add description
        cv2.putText(frame, "Wave contour detection", xy_cord_title, font, 
                    font_scale, font_color, font_thickness)
        cv2.putText(frame, "Image index: " + str(frame_index), xy_cord_index, 
                    font, font_scale, font_color, font_thickness)
        cv2.imshow(window_title, frame)
        cv2.waitKey(frame_delay)

    cv2.waitKey(0)

#SIFT (Scale Invariant Feature Transform) feature detection
def show08_display_sift_feature(window_setup, image_data_setup, title_setup):
    frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, \
        image_size_quart_y, frame_delay, rgb_color = window_setup
    image_path, image_prefix, image_suffix, frame_start_index, frame_count \
        = image_data_setup
    xy_cord_title, xy_cord_index, font, font_scale, font_color, \
        font_thickness, window_title = title_setup
    
    for i in range(frame_count):
        frame_index = frame_start_index + i
        frame_name = image_path + image_prefix + str(frame_index) + image_suffix
        frame = cv2.imread(frame_name)

        #SIFT feature detection
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keyPoints = sift.detect(frame_gray, None)
        frame = cv2.drawKeypoints(frame_gray, keyPoints, frame, flags = 
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        #Add description
        cv2.putText(frame, "SIFT feature detection", xy_cord_title, font, 
                    font_scale, font_color, font_thickness)
        cv2.putText(frame, "Image index: " + str(frame_index), xy_cord_index, 
                    font, font_scale, font_color, font_thickness)
        cv2.imshow(window_title, frame)
        cv2.waitKey(frame_delay)
    
    cv2.waitKey(0)

#Farneback dense optical flow
def show09_display_farneback_dense_optical_flow(window_setup, image_data_setup, 
                                                title_setup):
    frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, \
        image_size_quart_y, frame_delay, rgb_color = window_setup
    image_path, image_prefix, image_suffix, frame_start_index, frame_count \
        = image_data_setup
    xy_cord_title, xy_cord_index, font, font_scale, font_color, \
        font_thickness, window_title = title_setup
    
    #Load first frame, then start from second frame
    frame_index = frame_start_index + 0
    frame_name = image_path + image_prefix + str(frame_index) + image_suffix
    frame = cv2.imread(frame_name)
    frame_gray_previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(frame_count - 1):
        frame_index = frame_start_index + 1 + i
        frame_name = image_path + image_prefix + str(frame_index) + image_suffix
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
                cv2.line(frame, point1, point2, flow_color, font_thickness)

        frame_gray_previous = frame_gray
        #Add description
        cv2.putText(frame, "Dense optical flow", xy_cord_title, font, 
                    font_scale, font_color, font_thickness)
        cv2.putText(frame, "Image index: " + str(frame_index), xy_cord_index, 
                    font, font_scale, font_color, font_thickness)
        cv2.imshow(window_title, frame)
        cv2.waitKey(frame_delay)

    cv2.waitKey(0)

#Display image set #2
#Detect surfer and sea animal
def show10_display_surfer_detection(window_setup, image_data_setup, title_setup):
    frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, \
        image_size_quart_y, frame_delay, rgb_color = window_setup
    image_path, image_prefix, image_suffix, frame_start_index, frame_count \
        = image_data_setup
    xy_cord_title, xy_cord_index, font, font_scale, font_color, \
        font_thickness, window_title = title_setup

    for i in range(frame_count):
        frame_index = frame_start_index + i
        frame_name = image_path + image_prefix + str(frame_index) + image_suffix
        frame = cv2.imread(frame_name)
        
        #Detect objects
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_blurred = cv2.GaussianBlur(frame_gray, (7, 7), 0)
        edge_canny = cv2.Canny(frame_gray_blurred, 10, 250, 5)
        (objects, _) = cv2.findContours(edge_canny.copy(), cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)

        #Filter out objects by size
        for obj in objects:
            minX = frame_size_x
            maxX = 0
            minY = frame_size_y
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
        
        frame = cv2.putText(frame, "Surfer detection", xy_cord_title, font, 
                            font_scale, font_color, font_thickness)
        frame = cv2.putText(frame, "Image index: " + str(frame_index), 
                            xy_cord_index, font, font_scale, font_color, 
                            font_thickness)
        cv2.imshow(window_title, frame)
        cv2.waitKey(frame_delay)

    cv2.waitKey(0)

#Main setup
def main():
    #Window setup
    frame_size_x = 1920
    frame_size_y = 1080
    frame_size_half_x = int(frame_size_x / 2)
    frame_size_half_y = int(frame_size_y / 2)
    image_size_quart_y = int(frame_size_y / 4)
    frame_delay = 1
    rgb_color = ("b", "g", "r")
    window_setup = (frame_size_x, frame_size_y, frame_size_half_x, frame_size_half_y, image_size_quart_y, frame_delay, rgb_color)

    #Title set up
    xy_cord_title = (100, 75)
    xy_cord_index = (1300, 75)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 255, 255)
    font_thickness = 2
    window_title = "AI lab 01"
    title_setup = (xy_cord_title, xy_cord_index, font, font_scale, font_color, font_thickness, window_title)

    #Work on image set #1
    image_path = "./lab01_image_set_1_scout_09/"
    image_prefix = "gimbal0_"
    image_suffix = ".jpg"
    frame_start_index = 2000
    frame_count = 101
    image_data_setup = (image_path, image_prefix, image_suffix, 
                        frame_start_index, frame_count)
    #Display first screen
    show00_display_title(window_setup, image_data_setup, title_setup)
    #Show image set #1
    show01_display_original_image_set(window_setup, image_data_setup, 
                                      title_setup)

    #Histogram setup
    hist_height_inch = 3
    hist_width_inch = 6
    hist_dpi = 100
    hist_anker_x = 1300
    hist_anker_y = 100
    hist_dimension = (hist_height_inch, hist_width_inch, hist_dpi, hist_anker_x, 
                      hist_anker_y)
    show02_display_rgb_histogram(window_setup, image_data_setup, title_setup, 
                                 hist_dimension)

    #Incremental blurring
    show03_display_blur_response(window_setup, image_data_setup, title_setup)

    #Thresholding on gray and RGB channels
    show04_display_threshold(window_setup, image_data_setup, title_setup)

    #Canny edge detection
    show05_display_canny_edge(window_setup, image_data_setup, title_setup)

    #Laplacian gradient
    show06_display_laplacian_gradient(window_setup, image_data_setup, 
                                      title_setup)

    #Canny edge and contour detection
    show07_display_canny_contour(window_setup, image_data_setup, title_setup)

    #SIFT (Scale Invariant Feature Transform) feature detection
    show08_display_sift_feature(window_setup, image_data_setup, title_setup)

    #Farneback dense optical flow
    show09_display_farneback_dense_optical_flow(window_setup, image_data_setup, 
                                                title_setup)
    
    #Display image set #2 for object detection
    image_path = "./lab01_image_set_2_scout_06/"
    frame_start_index = 1500
    frame_count = 101
    image_data_setup = (image_path, image_prefix, image_suffix, 
                        frame_start_index, frame_count)
    show01_display_original_image_set(window_setup, image_data_setup, 
                                      title_setup)

    #Detect surfer and sea animal
    show10_display_surfer_detection(window_setup, image_data_setup, title_setup)

    #Terminate
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
    












