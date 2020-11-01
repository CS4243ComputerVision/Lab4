from skimage.feature import corner_harris, corner_peaks

from lab4 import *
import cv2
import numpy as np
import time

def mean_shift_test():
    # load frames
    path = "BlurBody/img/"
    frames = load_frames_rgb(path)

    # first frame in the video
    frame1 = frames[0]

    # windows position for human body in frame 1.
    x, y, w, h = 400, 48, 87, 319
    # track_window = (x, y, w, h)
    # get region of interest in the frame
    roi = frame1[y:y + h, x:x + w]
    # convert to hsv color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    """ PARAMETERS HERE """
    # Histogram masking parameters.
    thresh_value1 = 80.
    thresh_value2 = 20.
    # index of channel for which we calculate Histogram.
    # 0:hue, 1:saturation, 2:lightness
    channel = 1
    """ PARAMETERS HERE """

    mask = cv2.inRange(hsv_roi, np.array((0., thresh_value1, thresh_value2,)), np.array((180., 255., 255.,)))
    roi_hist = cv2.calcHist([hsv_roi], [channel], mask, [180], [0, 180])
    # roi_hist_nomask = cv2.calcHist([hsv_roi], [channel], None, [180], [0, 180])

    # hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
    # dst = cv2.calcBackProject([hsv_frame1], [channel], roi_hist, [0, 180], 1)
    # Initailze bounding box
    bboxes = []

    # tracking windows position for human body in frame 1.
    track_window = (x, y, w, h)

    print("performing mean shift")

    for frame in frames[:1]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        dst = cv2.calcBackProject([hsv], [channel], roi_hist, [0, 180], 1)
        # c = np.reshape(dst, (-1, 1))

        # apply meanshift to get the optimal tracking window location(mode)
        track_window = meanShift(dst, track_window, max_iter=10, stop_thresh=0.01)

        ## OpenCV "model" answer
        #     term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        #     ret, track_window = cv2.meanShift(dst,track_window,term_crit)
        x, y, w, h = track_window
        bboxes.append((x, y, w, h))
    # ani = animated_bbox(frames, bboxes)
    # HTML(ani.to_html5_video())


def lucas_kanade_test():
    # load frames
    path = "TeaCan"
    frames_rgb = load_frames_rgb(path)
    frames = load_frames_as_float_gray(path)

    # Detect keypoints to track in first frame
    keypoints = corner_peaks(corner_harris(frames[0]),
                             exclude_border=5,
                             threshold_rel=0.01)

    flow_vectors = lucas_kanade(frames[0], frames[1], keypoints, window_size=5)

    # Plot flow vectors
    plt.figure(figsize=(15, 12))
    plt.imshow(frames[0], cmap="gray")
    plt.axis('off')
    plt.title('Optical Flow Vectors')

    for y, x, vy, vx in np.hstack((keypoints, flow_vectors)):
        plt.arrow(x, y, vx, vy, head_width=3, head_length=3, color='b')

def compute_error_test():
    path = "TeaCan"
    frames_rgb = load_frames_rgb(path)
    frames = load_frames_as_float_gray(path)

    # Detect keypoints to track in the first frame
    keypoints = corner_peaks(corner_harris(frames[0]),
                             exclude_border=5,
                             threshold_rel=0.01)

    trajs = track_features(frames, keypoints,
                           error_thresh=1.5,
                           optflow_fn=lucas_kanade,
                           window_size=5)


def iterative_lucas_kanade_test():
    # load frames
    path = "TeaCan"
    frames_rgb = load_frames_rgb(path)
    frames = load_frames_as_float_gray(path)

    # Detect keypoints to track in first frame
    keypoints = corner_peaks(corner_harris(frames[0]),
                             exclude_border=5,
                             threshold_rel=0.01)

    flow_vectors = iterative_lucas_kanade(frames[0], frames[1], keypoints, window_size=5)


def pyramid_lucas_kanade_test():
    # load frames
    path = "TeaCan"
    frames_rgb = load_frames_rgb(path)
    frames = load_frames_as_float_gray(path)

    # Detect keypoints to track in first frame
    keypoints = corner_peaks(corner_harris(frames[0]),
                             exclude_border=5,
                             threshold_rel=0.01)

    # Lucas-Kanade method for optical flow
    flow_vectors = pyramid_lucas_kanade(frames[0], frames[1], keypoints)

def pyrLKtrack_test():
    # load frames
    path = "TeaCan"
    frames_rgb = load_frames_rgb(path)
    frames = load_frames_as_float_gray(path)

    keypoints = corner_peaks(corner_harris(frames[0]),
                             exclude_border=5,
                             threshold_rel=0.01)

    trajs = track_features(frames, keypoints,
                           error_thresh=1.5,
                           optflow_fn=pyramid_lucas_kanade,
                           window_size=5)


if __name__ == '__main__':
    print("starting")
    start = time.time()
    # mean_shift_test()
    # lucas_kanade_test()
    # compute_error_test()
    # iterative_lucas_kanade_test()
    # pyramid_lucas_kanade_test()
    pyrLKtrack_test()
    print("done: {:.3f}s".format(time.time() - start))
