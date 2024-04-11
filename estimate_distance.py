"""
This .py is to detect speedbump(and person), and measure the distance to them.
"""
import pandas as pd
import numpy as np
import cv2
import math
import argparse
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

# Importing custom modules and functions
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# Record the start time of the programme
start_time = time.time()
print('Pandas Version:', pd.__version__)
print('Nunpy Version:', np.__version__)

cam_params_file = "camera_params.xml"
fs = cv2.FileStorage(cam_params_file, cv2.FILE_STORAGE_READ)
K = fs.getNode("K").mat()
D = fs.getNode("D").mat()


@torch.no_grad()
class DistanceEstimation:
    def __init__(self):
        self.W = 640  # breadth
        self.H = 480  # high degree

    def object_point_world_position(self, u, v, h):
        u1 = u  # The horizontal coordinate u is unchanged
        v1 = v + h / 2  # 垂The vertical coordinate v moves down half a height h
        extrinsic_file = "extrinsic.xml"
        fs = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
        H = fs.getNode("H").mat()
        # H = np.array([[-9.28937918e-02, -6.95684039e-03, 3.37193124e+01],
        #               [-7.08935280e-03, -4.38287061e-02, -7.69825566e+00],
        #               [-1.22483567e-04, -3.09696310e-03, 1.00000000e+00]])
        # H_inv = np.linalg.inv(H)
        point_c = np.array([u1, v1, 1])  # Construct the chi-square coordinates of a point
        point_w = np.matmul(H, point_c)  # Mapping to the world coordinate system using the single response matrix H
        point_w /= point_w[2]  # Normalisation to ensure that the last bit of the chi-square coordinate is 1

        d1 = np.array((point_w[0], point_w[1]), dtype=float)  # Extract x and y coordinates
        return d1  # Returns the calculated position in the world coordinate system.

    def distance(self, kuang):
        print('=' * 50)  # Print dividers for aesthetics and to differentiate output
        print('开始测距')  # Outputs a message to start ranging

        if len(kuang):  # If kuang (the array containing the target box information) is not empty
            # Calculate the actual pixel coordinates of the target frame in the image
            u, v, w, h = kuang[1] * self.W, kuang[2] * self.H, kuang[3] * self.W, kuang[4] * self.H
            print('目标框', u, v, w, h)  # Print the coordinates and size of the target box

            d1 = self.object_point_world_position(u, v, h)  # Call the previously defined function to calculate the world coordinates of the target

        print('世界坐标', d1 / 100)  # Print the calculated world coordinates

        distance = (math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))) / 100  # Calculating distances in a two-dimensional plane
        print('距离', distance)

        return distance

    def Detect(self, weights='best.pt',
               source='data/images',  # Image/Video Source
               imgsz=640,  # Reasoning about image size
               conf_thres=0.25,  # confidence threshold
               iou_thres=0.45,  # IOU threshold
               max_det=1000,  # Maximum number of detections per image
               device='',  # Equipment used
               view_img=False,  # Whether to display images
               save_txt=False,  # Whether to save results to text
               save_conf=False,  # Whether to preserve confidence
               save_crop=False,  # Whether to save the cropped prediction box
               nosave=False,  # Whether to save images/videos
               classes=None,  # Categories of filtration
               agnostic_nms=False,  # Category agnostic NMS
               augment=False,  # Whether to engage in augmented reasoning
               update=False,  # Whether to update all models
               project='inference/output',  # Path to save results
               name='exp',  # Save the name of the result
               exist_ok=False,  # Whether to allow already existing items/names
               line_thickness=3,  # Boundary frame thickness
               hide_labels=False,  # Whether to hide labels
               hide_conf=False,  # Whether to hide confidence
               half=False  # Whether to use half-precision reasoning
               ):

        save_img = not nosave and not source.endswith('.txt')  # If it is not a no-image-save mode and the source file is not a text file, the inference image is saved
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))  # Determine if the data source is a webcam or a video stream

        save_dir = Path(project)  # Setting the directory where the results are saved

        # initialisation
        set_logging()  # Setting up logging
        device = select_device(device)  # Select device (CPU or GPU)
        half &= device.type != 'cpu'  # Enable half-precision if not CPU (FP16)

        # Loading Models
        model = attempt_load(weights, map_location=device)  # Load model (FP32 precision)
        stride = int(model.stride.max())  # Get the model's stride
        names = model.module.names if hasattr(model, 'module') else model.names  # Get the class name of the model
        if half:
            model.half()  # Converting Models to FP16 Accuracy

        # Second-stage classifiers (if required)
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # Initialise the classifier
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Setting up the data loader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()  # Check if the image can be displayed
            cudnn.benchmark = True  # To speed things up, set True
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)  # Loading video streams
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)  # Load Image

        # running reasoning
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # Warming up models on non-CPU devices

        t0 = time.time()  # Record the start time of reasoning
        for path, img, im0s, vid_cap in dataset:
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), K, (self.W, self.H), cv2.CV_16SC2)
            for i, im0 in enumerate(im0s):
                im0s[i] = cv2.remap(im0, map1, map2, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
            im0 = cv2.remap(im0, map1, map2, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
            img = torch.from_numpy(img).to(device)  # Convert image data to PyTorch tensor and send to device
            img = img.half() if half else img.float()  # Adjusts the image data type depending on whether half-precision is used or not
            img /= 255.0  # Normalise image data to 0-1

            if img.ndimension() == 3:
                img = img.unsqueeze(0)  # Adding a batch dimension to image data

            # reasoning
            t1 = time_synchronized()  # Record the start time of reasoning
            pred = model(img, augment=augment)[0]  # Run model inference and get predictions

            # Applying non-extreme value suppression
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            t2 = time_synchronized()  # Record the end time of reasoning

            # Applying the second stage classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Processing of test results
            for i, det in enumerate(pred):  # Iterate through the detection results for each image
                if webcam:  # If it's a webcam or video stream
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # Convert a path string to a Path object
                save_path = str(save_dir / p.name)  # Setting the image save path
                txt_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # Setting the text save path

                s += '%gx%g ' % img.shape[2:]  # Adding image size information to a string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Get the normalisation factor for the image size
                imc = im0.copy() if save_crop else im0  # If you need to save the crop, copy the original image

                if len(det):
                    # Resize the detection frame
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        if not names[int(c)] in ['person', 'speedbump']:
                            continue
                        n = (det[:, -1] == c).sum()  # Statistics on the number of tests per category
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Adds the result to the string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if not names[int(cls)] in ['person', 'speedbump']:
                            continue
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # Convert coordinates to aspect format
                        kuang = [int(cls), xywh[0], xywh[1], xywh[2], xywh[3]]  # Constructing frame data
                        if save_txt:  # If you need to save the text
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (int(cls), *xywh))  # Write to text file

                        distance = self.distance(kuang)  # Calculate the distance to the detected object

                        # Adding a Border to an Image
                        if save_img or save_crop or view_img:  # If you need to save an image, crop or display an image
                            c = int(cls)  # Converting categories to integers
                            # Decide whether to display labels and confidence levels based on configuration
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            if label is not None and distance != 0:
                                label = label + ' ' + str('%.2f' % distance) + 'm'  # Adding distance information to labels
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                         line_thickness=line_thickness)  # Drawing borders and labels on images
                            if save_crop:
                                # If you need to save the crop, save the portion of the image inside the detection box
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Time to print inference and non-extremely large value suppression
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Show results
                if view_img:
                    cv2.imshow(str(p), im0)  # Show image
                    cv2.waitKey(1)  # Wait 1 millisecond

                # Save results (images with detection frames)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)  # Save image
                    else:  # If it's video or streaming
                        if vid_path != save_path:  # If it's a new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # Release the previous video writer
                            if vid_cap:  # If it's a video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # If it's a stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)  # Write video frames

        # Save results
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''  # If a text file is saved, output the number of saved labels
            print(f"Results saved to {save_dir}{s}")  # Where the printout is saved

        # Updating the model
        if update:
            strip_optimizer(weights)  # Updating the model to fix potential warnings, such as SourceChangeWarning

        # Print Completion Information
        print(f'Done. ({time.time() - t0:.3f}s)')  # Total Printing Time


# Main function entry
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='1', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default='mydata/meeting2.mp4', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1440, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save_txt', default=False, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='inference/output', help='save results to project/name')  # save address
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()  # Parsing command line arguments
    print(opt)  # Print the parsed parameters
    check_requirements(exclude=('tensorboard', 'thop'))  # Check that the required dependencies are satisfied

    print('开始进行目标检测和单目测距！')  # Output start message
    DE = DistanceEstimation()  # Creating a distance estimation object
    DE.Detect(**vars(opt))  # Calling the detection method

    # If more than 10 seconds have elapsed, wait for a keystroke and close all windows
    if time.time() > (start_time + 10):
        cv2.waitKey(0)
        cv2.destroyAllWindows()