"""
This .py is to compute H between two planes:image plane and ground plane
"""
import cv2
import numpy as np

# imgps = [[503,384],[183,391],[422,347],[258,352]] #mm
# objps = [[62.5,112],[-60,112],[62.5,205],[-60,205]]  #mm
imgps = [] #mm
objps = []  #mm
# mouse callback function
# This function is to get and store pairs of pixel coordinates and world coordinates.
# On clicking at a point on the image, the function gets its pixel coordinates,
# and require us input the world coordinates of the point in cmd line.
def click_corner(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(groundImg, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(groundImg, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)

        world_coord = input("Enter world coordinates for selected point (format: x y): ")
        world_coord = list(map(float, world_coord.split()))

        imgps.append([x, y])
        objps.append(world_coord)
        
if __name__ == '__main__':
    # capture a image
    inputVideo = cv2.VideoCapture(1)
    if not inputVideo.isOpened():
        print("打开相机失败")
    else:
        print("相机已打开")

    fs = cv2.FileStorage("camera_params.xml", cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()

    while True:
        ret, frame = inputVideo.read()
        if frame is None:
            continue
        # Undistort frame
        undistorted_frame = cv2.fisheye.undistortImage(frame, K, D, None, K)
        cv2.imshow("groundImg", undistorted_frame)
        key = cv2.waitKey(1)
        if key == 27:  # Stop reading video when escape key entered
            break
        elif key == ord('c') or key == ord('C'):# Freeze the frame
            groundImg = frame
            cv2.setMouseCallback("groundImg", click_corner)
            while True:
                cv2.imshow("groundImg", groundImg)
                key = cv2.waitKey(1)
                if key == ord('q') or key == ord('Q'):# Unfreeze the frame
                    break

    imgps = np.array(imgps, dtype=np.float32)
    objps = np.array(objps,dtype=np.float32)
    # print(imgps)
    # print(objps)
    if len(imgps) >=4:
        H , _ = cv2.findHomography(imgps,objps)

        # Write the homography matrix H into the XML file
        fs = cv2.FileStorage("extrinsic.xml", cv2.FILE_STORAGE_WRITE)
        fs.write("H", H)
        fs.release()
        print("外参已保存至extrinsic.xml")

    cv2.destroyAllWindows()

    