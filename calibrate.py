"""
This .py is to capture checkerboard images and calibrate the intrinsics and distortion coefficients of the fisheye camera.
You can also use our captured images in folder imgs/ to calculate the params os our camera.
"""
import cv2
import numpy as np
import os
import glob

# Size of the checkerboard pattern
CHECKERBOARD = (9, 6)
BLOCK_SIZE = (30,30)

# Criteria for corner refinement
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# Calibration flags for fisheye calibration
calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_CHECK_COND +
        cv2.fisheye.CALIB_FIX_SKEW
)

# 3D coordinates of the checkerboard corners in real-world space
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0]*BLOCK_SIZE[0]:BLOCK_SIZE[0], 0:CHECKERBOARD[1]*BLOCK_SIZE[1]:BLOCK_SIZE[1]].T.reshape(-1, 2)

# Placeholder for the shape of the images
_img_shape = None

# Lists to store object points (3D) and image points (2D)
objpoints = []
imgpoints = []

# Camera capture
cap = cv2.VideoCapture(1)  # Use 0 for the default camera, or specify the camera index

# Folder path to save captured images
# You can modify it to "imgs/", and be free from capturing images
# to see the params of our fisheye camera
save_folder = 'qimgs/'
os.makedirs(save_folder, exist_ok=True)

image_counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        # If the user presses 'c', capture and save the image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # If corners are found, draw them on the frame
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
            cv2.imshow('Calibration', frame)

            # If the shape of the first image is not set, set it
            if _img_shape is None:
                _img_shape = frame.shape[:2]
            else:
                assert _img_shape == frame.shape[:2], "All images must share the same size."

            # Refine corner points
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)

            # Append object and image points
            objpoints.append(objp)
            imgpoints.append(corners)

            # Save the image
            image_counter += 1
            image_filename = os.path.join(save_folder, f"{image_counter}.jpg")
            cv2.imwrite(image_filename, frame)
            print(f"Image captured and saved: {image_filename}")

            cv2.waitKey(1000)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        print("Can not find chessboard corners!\n")

    # Display the frame
    cv2.imshow('Calibration', frame)

    # If the user presses 'q', exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# Get all image files in the folder
images = glob.glob(os.path.join(save_folder, '*.jpg'))

# Compute K and D using images just captured
print("Image")
for fname in images:
    img = cv2.imread(fname)

    # If the shape of the first image is not set, set it
    if _img_shape is None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # If corners are found, add object points and image points (after refining them)
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)

# Number of valid images for calibration
N_OK = len(objpoints)

# Arrays to store intrinsic and extrinsic parameters
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

# Perform fisheye calibration
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

# Display calibration results
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

# Write camera intrinsic matrix K and distortion coefficients D into the XML file
fs = cv2.FileStorage("camera_params.xml", cv2.FILE_STORAGE_WRITE)
fs.write("K", K)
fs.write("D", D)

fs.release()

img = cv2.imread(save_folder+"1.jpg")
undistorted_img = cv2.fisheye.undistortImage(img, K, D,None, K)
cv2.imshow("undistortedTestImg", undistorted_img)
cv2.waitKey(0)