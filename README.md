# 2024 CS AI and Computer Vision Lab #1 - OpenCV
Demosntrate OpenCV operation on ocean images sets<br>
Main function call demonstration functions over two sets of ocean images.<br>
Demonstration functions are:
```
show00_display_title
show01_display_original_image_set
show02_display_rgb_histogram
show03_display_blur_response
show04_display_threshold
show05_display_canny_edge
show06_display_laplacian_gradient
show07_display_canny_contour
show08_display_sift_feature
show09_display_farneback_dense_optical_flow
show10_display_surfer_detection
```
There are two sets of images. The `set_1` is for wave analysis, and `set_2` is for surfer detection. Each set has 101 images.
```
lab01_image_set_1_scout_09
lab01_image_set_2_scout_06
```
## How to run
`
python3 csai_lab1.py
`
## Coverage
|chapter\show                      | 00    | 01    | 02    | 03    | 04    | 05    | 06    | 07    | 08    | 09    | 10    |
|----------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|3. Loading, displaying, and saving|&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|
|4. Image basics                   |&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|
|5. Drawing                        |&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|
|6. Image processing               |       |       |&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|&check;|
|7. Histogram                      |       |       |&check;|       |       |       |       |       |       |       |&check;|
|8. Smoothing and Blurring         |       |       |       |&check;|       |       |       |       |       |       |&check;|
|9. Thresholding                   |       |       |       |       |&check;|       |       |       |       |       |       |
|10. Gradients and edge detection  |       |       |       |       |       |&check;|&check;|&check;|       |       |&check;|
|11. Contours                      |       |       |       |       |       |       |       |&check;|       |       |&check;|
|Extra1: SIFT feature detection    |       |       |       |       |       |       |       |       |&check;|       |       |
|Extra2: Farneback optical flow    |       |       |       |       |       |       |       |       |       |&check;|       |
