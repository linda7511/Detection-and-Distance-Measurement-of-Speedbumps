# Detection-and-Distance-Measurement-of-Speedbumps
## Program structure
Detection and Distance Measurement of speedbumps
|——imgs   #Store checkerboard pictures captured by us using our fisheye camera.
|——inference
|  |——images/
|  |——Loutput/    #Store the result vedio of each distance measurement.
|——models    #Models of yolov5
|——runs
|  |——train/    #Store the weights of speedbump person detect model trained by us.
|——utils    #Modules of yolov5
|——weights    #Same as the weights in runs/trains. fFor easier import.
|——calibrate.py    #Capture checkerboard images and solve the intrinsics and distortion coefficients of the 
                   fisheye camera
|——compute_homography.py #Compute the homography matrix between image plane and ground plane.
|——estimate_distance.py    #Detect speedbump(and person), and measure the distance from camera to them.
|——train.py    #Train a YOLOv5 model on a custom dataset. lt's result are saved to "runs!/".
|——camParams.xml    #Save the intrinsics and distortion coeffitients of our fisheye camera.
|——extrinsics.xml    # Save the homography matrix between image plane and ground plane.
