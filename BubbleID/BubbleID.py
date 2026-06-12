# BubbleID
# Import libraries:
import torch, detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# import some common detectron2 utilities
from detectron2 import model_zoo, structures
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog, DatasetCatalog

import torch

torch.cuda.is_available()

from detectron2.engine import DefaultTrainer
import os
import glob

import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import time
from tqdm import tqdm
import numpy as np
import cv2
import filterpy
import torch
import super_gradients as sg
import matplotlib.pyplot as plt
from .ocsort import ocsort
from .indexing import filter_segmentation_outputs
import colorsys
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import PIL.Image
from scipy.spatial import cKDTree

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftfreq
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import cv2
import numpy as np

import torch, detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2.data import transforms as T
# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo, structures
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import copy
import torch

from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

import torch
from torchvision import models, transforms
from PIL import Image


# Define Helper Functions
def get_image_paths(directory):
    image_extensions = ['*.jpg']  # Add more extensions as needed

    image_paths = []
    for extension in image_extensions:
        pattern = os.path.join(directory, '**', extension)
        image_paths.extend(glob.glob(pattern, recursive=True))

    return sorted(image_paths)  # Sort the list of image paths alphabetically


def get_color(number):
    """ Converts an integer number to a color """
    # change these however you want to
    hue = number * 30 % 180
    saturation = number * 103 % 256
    value = number * 50 % 256

    # expects normalized values
    color = colorsys.hsv_to_rgb(hue / 179, saturation / 255, value / 255)

    return [int(c * 255) for c in color]


# Define Function for computing iou
def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
              + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return (o)


class DataAnalysis:
    def __init__(self, imagesfolder, videopath, savefolder, extension, modelweightsloc, device, height, width, fps, px):
        self.imagesfolder = imagesfolder
        self.videopath = videopath
        self.savefolder = savefolder
        self.extension = extension
        self.modeldirectory, self.modelweights = os.path.split(modelweightsloc)
        self.device = device
        self.height = height
        self.width = width
        self.fps = fps
        self.px = px

    def GenerateData(self, thres=0.5):
        directory_path = self.imagesfolder
        video_file = self.videopath

        # define save locations
        file_path = self.savefolder + f'/bb-Boiling-{self.extension}.txt'
        output_file_path = self.savefolder + f'/bb-Boiling-output-{self.extension}.txt'
        vapor_file = self.savefolder + f'vapor_{self.extension}.npy'
        vapor_base_file = self.savefolder + f'/vaporBase_bt-{self.extension}.npy'
        bubble_size_file = self.savefolder + f'/bubble_size_bt-{self.extension}.npy'
        bubind_file = self.savefolder + f'/bubind_{self.extension}.npy'
        frameind_file = self.savefolder + f'/frames_{self.extension}.npy'
        classind_file = self.savefolder + f'/class_{self.extension}.npy'
        bubclassind_file = self.savefolder + f'/bubclass_{self.extension}.npy'

        # Make save folder if it does not exist
        if not os.path.exists(self.savefolder):
            os.makedirs(self.savefolder)

        # load model
        print("Load model")
        cfg = get_cfg()
        cfg.OUTPUT_DIR = self.modeldirectory
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 1000  # 1000 iterations seems good enough for this dataset
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Default is 512, using 256 for this dataset.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, self.modelweights)  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
        if self.device == 'cpu':
            cfg.MODEL.DEVICE = 'cpu'
        predictor = DefaultPredictor(cfg)

        # Get list of sorted image files
        print("Load image paths")
        image_paths = get_image_paths(directory_path)
        image_paths_sub = image_paths[0:]

        print("Run instance segmentation model and save data")
        Bounding_Box = np.empty((0, 7))
        bubble_size = []
        vapor = []
        vapor_base = []
        for i in tqdm(range(len(image_paths_sub))):
            # for i in [117,118,119]:
            new_im = cv2.imread(image_paths_sub[i])
            outputs = predictor(new_im)
            box = outputs["instances"].pred_boxes
            box = box.tensor
            box = box.cpu().tolist()
            masks = outputs["instances"].pred_masks.cpu()
            scores = outputs["instances"].scores
            # scores=scores.tensor
            scores = scores.cpu().tolist()
            class_val = outputs["instances"].pred_classes.cpu().tolist()

            converted_bounding_box = []
            if len(scores) > 0:
                for j in range(len(box)):

                    x1, y1, x2, y2 = box[j]

                    # if y2 > 502 and y2<533 and x1>320 and x1<515:
                    # if y2 > 502 and y2<680 and x1>250 and x1<580:
                    if y2 > 0:

                        converted_bounding_box.append([x1, y1, x2, y2])
                    elif y2 > 502 and y2 < 533 and x2 > 320 and x2 < 515:
                        # elif y2 > 502 and y2<680 and x2>250 and x2<580:
                        converted_bounding_box.append([x1, y1, x2, y2])

                box = converted_bounding_box
                box_data = [[i + 1] + box[j] + [scores[j]] + [class_val[j]] for j in range(len(box))]

                if len(box_data) != 0:
                    Bounding_Box = np.vstack([Bounding_Box, box_data])
                else:
                    print('Error')
                    outputs = predictor(new_im)
                    box = outputs["instances"].pred_boxes
                    box = box.tensor
                    box = box.cpu().tolist()
                    x1, y1, x2, y2 = box[0]
                    box_data = [[i + 1] + box[j] + [scores[j]] + [class_val[j]] for j in range(len(box))]

                    Bounding_Box = np.vstack([Bounding_Box, box_data])

                masks = outputs['instances'].pred_masks.cpu()
                scores = outputs['instances'].scores.cpu()
                masks, class_val, masks_base = filter_segmentation_outputs(
                    masks, scores, class_val, thres
                )
                combined_mask = torch.any(masks, axis=0)
                vapor.append(torch.sum(combined_mask).item())
                combined_mask = torch.any(masks_base, axis=0)
                vapor_base.append(torch.sum(combined_mask).item())
                pixel_count = torch.sum(masks, dim=(1, 2)).numpy()
                bubble_size.append(pixel_count)
            else:
                print(i)

        # Save vapor and bubble size
        np.save(vapor_file, vapor)
        np.save(vapor_base_file, vapor_base)
        np.save(bubble_size_file, bubble_size, allow_pickle=True)

        # Save bounding box data
        with open(file_path, 'w') as file:
            for sublist in Bounding_Box:
                formatted_values = []
                for value in sublist:
                    if isinstance(value, int):
                        formatted_values.append(f'{int(value)}')  # Format integers as strings
                    elif isinstance(value, float):
                        formatted_values.append(f'{value:.4f}')  # Format floats with 4 decimal places
                    else:
                        formatted_values.append(str(value))  # Keep other types as they are

                # Manually format the first and last elements as integers
                formatted_values[0] = str(int(float(formatted_values[0])))
                formatted_values[-1] = str(int(float(formatted_values[-1])))
                # Join the formatted values with spaces
                line = ','.join(formatted_values)

                # Write the line to the file and add a newline character
                file.write(line + '\n')

        print("Perform ocsort tracking on saved data")

        tracker = ocsort.OCSort(det_thresh=thres, max_age=10, min_hits=20)
        img_info = (self.height, self.width)
        img_size = (self.height, self.width)
        boiling_test = 78
        hf = 120
        # cap=cv2.VideoCapture(f"/mnt/share/zdrive/Hari/Bubble hydro acoustic analysis/B83_210_Bubble/Bubble/video.avi")
        # cap=cv2.VideoCapture(f"/mnt/share/zdrive/Christy/Boiling-{boiling_test}/{hf}W-full.avi")
        cap = cv2.VideoCapture(video_file)

        if (cap.isOpened() == False):
            print("Error opening video file")

        frames = []
        i = 0
        counter, fps, elapsed = 0, 0, 0
        frame_data = {}
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into components
                parts = line.strip().split(',')

                # Extract frame ID and data
                frame_id = int(parts[0])
                data = list(map(float, parts[1:-1]))

                # Check if the frame ID is already in the dictionary
                if frame_id not in frame_data:
                    frame_data[frame_id] = []

                # Append the data to the corresponding frame ID
                frame_data[frame_id].append(data)

        with open(output_file_path, 'w') as file:

            while (cap.isOpened()):
                ret, frame = cap.read()

                if ret == True:
                    # load data from previously configured text file that has
                    # bounding boxes. (x1,y1,x2,y2,c)
                    xyxyc = frame_data.get(i + 1, [])
                    i += 1
                    if np.array(xyxyc).size == 0:
                        xyxyc = np.empty((0, 5))
                    tracks = tracker.update(np.array(xyxyc), img_info, img_size)
                    for track in tracker.trackers:
                        track_id = track.id
                        hits = track.hits
                        color = get_color(track_id * 15)
                        x1, y1, x2, y2 = np.round(track.get_state()).astype(int).squeeze()
                        '''
                        #cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                        #cv2.putText(frame, f"{track_id}-{hits}",
                         #           (x1+10, y1-5),
                          #          cv2.FONT_HERSHEY_SIMPLEX,
                           #         0.5,
                            #        color,
                             #       1,
                              #      cv2.LINE_AA)
                        #frames.append(frame)
                        '''
                        file.write(f'{i},{track_id},{hits},{x1},{y1},{x2},{y2}\n')
                else:
                    break
            cap.release()
            del cap

        print("Match tracking results to bubble indexs")
        # load data
        real_data = np.loadtxt(file_path, delimiter=",")
        real = real_data[real_data[:, -2] >= thres]
        real = np.round(real[:, 0:-2]).astype(int)
        # del real_data
        pred_data = np.loadtxt(output_file_path, delimiter=",").astype(int)

        # Initialize an empty dictionary
        my_dict = {}

        # Assign keys ranging from 1 to 10 using a for loop
        for i in range(1, real[-1][0] + 1):
            my_dict[i] = []

        for item in real:
            key = item[0]
            value = item[1:]
            if my_dict[key] is []:
                my_dict[key] = (list(value))
            else:
                my_dict[key].append(list(value))
        realg = [values for values in my_dict.values()]

        # Initialize an empty dictionary
        my_dict = {}

        # Assign keys ranging from 1 to 10 using a for loop
        for i in range(1, real[-1][0] + 1):
            my_dict[i] = []

        for item in real:
            key = item[0]
            value = item[1]
            if my_dict[key] is []:
                my_dict[key] = ((value))
            else:
                my_dict[key].append((value))
        realgG = [values for values in my_dict.values()]

        # Initialize an empty dictionary
        my_dict = {}

        # Assign keys ranging from 1 to 10 using a for loop
        for i in range(1, real[-1][0] + 1):
            my_dict[i] = []

        for item in pred_data:
            key = item[0]
            value = item[1:]
            if my_dict[key] is []:
                my_dict[key] = (list(value))
            else:
                my_dict[key].append(list(value))
        predg = [values for values in my_dict.values()]

        import copy
        values = copy.deepcopy(realg)
        tracks = copy.deepcopy(realgG)

        # given previous frame and current frame remove row that doesn't have

        for k in range(len(predg) - 1):
            # for k in range(2):
            if len(predg[k]) > 0 and len(predg[k + 1]) > 0:
                frame1 = predg[k]
                frame2 = predg[k + 1]
                vector1 = np.array(frame2)[:, 0].tolist()
                vector2 = np.array(frame1)[:, 0].tolist()
                result_vector = np.full(len(vector1), -1)
                for i, val in enumerate(vector1):
                    if val in vector2:
                        result_vector[i] = vector2.index(val)
                j = 0
                for i in range(len(result_vector)):
                    if result_vector[i] != -1:
                        if frame2[i][1] != frame1[result_vector[i]][1]:
                            # print(k,j,i)
                            tracks[k + 1][j] = frame2[i][0]
                            values[k + 1][j][:] = frame2[i][2:]
                            j += 1
                    if result_vector[i] == -1:
                        tracks[k + 1][j] = frame2[i][0]
                        values[k + 1][j][:] = frame2[i][2:]
                        j += 1

        # Tracks is of shape (frames, bb in real detection (basically removes tracks with no hits))
        tracks[0] = np.array(predg[0])[:, 0].tolist()
        values[0] = np.array(predg[0])[:, 2:].tolist()

        for i in range(len(values)):
            if len(values[i]) > 0:
                sort = np.argmax(iou_batch(realg[i], values[i]), axis=1).tolist()
                tracks[i] = np.array(tracks[i])[sort].tolist()

        # Original data
        data = tracks

        # Find the maximum number in the original data
        # max_number = max(max(row) for row in data)
        max_number = max(max(sublist, default=float('-inf')) for sublist in data if sublist)

        # Create a list of lists to store the index positions and initial row numbers
        frames = [[] for _ in range(max_number + 1)]

        # Iterate through the original data and populate the result list
        for initial_row, row in enumerate(data):
            for index, number in enumerate(row):
                frames[number].append(initial_row)

        # Original data
        data = tracks
        # Find the maximum number in the original data
        # max_number = max(max(row) for row in data)
        max_number = max(max(sublist, default=float('-inf')) for sublist in data if sublist)

        # Create a list of lists to store the index positions
        bubInd = [[] for _ in range(max_number + 1)]

        # Iterate through the original data and populate the result list
        for row in data:
            for index, number in enumerate(row):
                bubInd[number].append(index)

        # Print the result
        # print(bubInd)

        # Save Data
        bubInd = np.array(bubInd, dtype=object)
        np.save(bubind_file, bubInd)
        frames = np.array(frames, dtype=object)
        np.save(frameind_file, frames)

        classes = real_data[:, -1]

        # Initialize an empty dictionary
        my_dict = {}

        # Assign keys ranging from 1 to 10 using a for loop
        for i in range(1, real[-1][0] + 1):
            my_dict[i] = []

        for item in real_data:
            key = item[0]
            value = item[-1]
            if my_dict[key] is []:
                my_dict[key] = ((value))
            else:
                my_dict[key].append((value))
        realgG = [values for values in my_dict.values()]

        np.save(classind_file, realgG)

        bub_class = copy.deepcopy(frames)
        for j in range(len(bubInd)):
            bub = j
            for i in range(len(frames[bub])):
                bub_class[j][i] = realgG[frames[bub][i]][bubInd[bub][i]]

        bub_class = np.array(bub_class, dtype=object)
        np.save(bubclassind_file, bub_class)
        print("Finish")

    def Plotvf(self):
        vf_path = self.savefolder + f'vapor_{self.extension}.npy'
        vidstart = 0
        vf = np.load(vf_path) / (self.height * self.width)
        time = [(i / self.fps) + vidstart for i in range(len(vf))]
        df = pd.DataFrame(data=vf)
        df['value'] = df.iloc[:, 0].rolling(window=300).mean()
        plt.figure(figsize=(5, 10))
        fig, ax1 = plt.subplots(figsize=(6, 2))
        ax1.plot(time, df, color='lightgray')
        ax1.plot(time, df['value'], color='darkblue', label='Rolling Average')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Vapor Fraction')
        ax1.legend()
        saveloc = self.savefolder + f'vaporfig_{self.extension}.png'
        plt.savefig(saveloc, bbox_inches='tight')
        plt.show()

    def Plotbc(self):
        bs_path = self.savefolder + f'/bubble_size_bt-{self.extension}.npy'
        bs = np.load(bs_path, allow_pickle=True)
        count = []
        for i in range(len(bs)):
            count.append(len(bs[i]))
        df = pd.DataFrame(data=count)

        vidstart = 0
        time = [(i / self.fps) + vidstart for i in range(len(count))]
        df['value'] = df.iloc[:, 0].rolling(window=300).mean()
        plt.figure(figsize=(5, 10))
        fig, ax1 = plt.subplots(figsize=(6, 2))
        ax1.plot(time, df, color='lightgray')
        ax1.plot(time, df['value'], color='darkblue', label='Rolling Average')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Bubble Count')
        ax1.legend()
        saveloc = self.savefolder + f'bcfig_{self.extension}.png'
        plt.savefig(saveloc, bbox_inches='tight')
        plt.show()

    def PlotInterfaceVelocity(self, bubble):
        directory_path = self.imagesfolder

        bubind_file = self.savefolder + f'/bubind_{self.extension}.npy'
        frameind_file = self.savefolder + f'/frames_{self.extension}.npy'

        # load model
        cfg = get_cfg()
        cfg.OUTPUT_DIR = self.modeldirectory
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 1000  # 1000 iterations seems good enough for this dataset
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Default is 512, using 256 for this dataset.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, self.modelweights)  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
        if self.device == 'cpu':
            cfg.MODEL.DEVICE = 'cpu'
        predictor = DefaultPredictor(cfg)

        print("Model Loaded")
        bubInd = np.load(bubind_file, allow_pickle=True)
        frames = np.load(frameind_file, allow_pickle=True)

        def get_image_paths(directory):
            """
            Get a list of file paths for all image files in the specified directory and its subdirectories.

            Args:
            directory (str): The directory to search for image files.

            Returns:
            List[str]: A list of file paths for all image files found, sorted alphabetically.
            """
            image_extensions = ['*.jpg']  # Add more extensions as needed

            image_paths = []
            for extension in image_extensions:
                pattern = os.path.join(directory, '**', extension)
                image_paths.extend(glob.glob(pattern, recursive=True))

            return sorted(image_paths)  # Sort the list of image paths alphabetically

        # Get a list of image file paths sorted alphabetically
        image_paths = get_image_paths(directory_path)

        # Set the output video file name
        output_file = f'./vel_bubble{bubble}1.avi'
        frame_width = self.width
        frame_height = self.height
        '''
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, 10, (frame_width, frame_height), isColor=True)
        '''
        skip = 5
        angles = []
        # Save Contours of a bubble in each frame

        image_paths_sub = image_paths[0:]

        # output_video = 'bubble0.avi'
        # frame_size = (832, 600)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter(output_video, fourcc, 20.0, frame_size, isColor=False)
        # bubble=12
        contours = []
        centroids = []
        values = []
        # for i in tqdm(range(len(frames[bubble]))):
        indexs = []
        print(len(frames[bubble]))
        for i in tqdm(range(len(frames[bubble]))):
            new_im = cv2.imread(image_paths_sub[frames[bubble][i]])
            outputs = predictor(new_im)
            box = outputs["instances"].pred_boxes
            box = box.tensor
            box = box.cpu().tolist()
            masks = outputs["instances"].pred_masks.cpu()
            scores = outputs["instances"].scores
            scores = scores.cpu().tolist()
            k = 0
            for j in range(len(box)):
                x1, y1, x2, y2 = box[j]
                if y2 > -1000:
                    if k == bubInd[bubble][i]:
                        mask = np.uint8(masks[j]) * 255
                        contours1, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                        if len(contours1) > 1:
                            largest_contour = max(contours1, key=cv2.contourArea)
                            contours1 = np.array(largest_contour).reshape((-1, 2))
                        else:
                            contours1 = np.array(contours1).reshape((-1, 2))
                            # out.write(mask)
                        contours.append(contours1)
                        indexs.append(j)
                    k += 1

        avg_mag = []
        mag = []
        contour_connections = []
        for j in range(len(frames[bubble]) - skip - 1):

            newimg = cv2.imread(image_paths_sub[frames[bubble][j]])

            frame1 = frames[bubble][j]
            frame2 = frames[bubble][j + skip]
            tree_set1 = cKDTree(contours[j + skip])
            distances, indices = tree_set1.query(contours[j], k=1)
            if j == 1:
                fig, ax = plt.subplots()
                ax.imshow(newimg)
                ax.scatter(contours[j][:, 0], contours[j][:, 1], label='Set 1', marker='o', s=1)
                ax.scatter(contours[j + skip][:, 0], contours[j + skip][:, 1], label='Set 2', marker='x', s=1)
                plt.show()

            velocitys = (np.array(distances) / self.px) / ((frame2 - frame1) / self.fps)
            distances = velocitys
            avg_mag.append(np.mean(np.array(distances)))
            mag.append(list(distances))
            # Connect the paired points
            ang = []
            # print(len(contours[j]), len(indices), max(indices))
            '''
            if j ==0:
                values.append(indices[0])
            else:
                values.append(indices[values[-1]])
            '''

            for i in range(0, len(contours[j]), 1):
                point1 = contours[j][i]
                point2 = contours[j + skip][indices[i]]
                vector = point1 - point2
                angle_rad = np.arctan2(vector[1], vector[0])
                angle_rad = np.degrees(angle_rad)
                if angle_rad < 0:
                    angle_rad += 360
                ang.append(angle_rad)
                # ax.arrow(point1[0], point1[1], point2[0] - point1[0], point2[1] - point1[1],
                #         head_width=4, head_length=6, fc='k', ec='k', linewidth=0.5)
            angles.append(ang)
            contour_connections.append(contours[j + skip][indices])

            direction = []
        for i in range(len(contours) - skip - 1):
            # for i in range(1):
            new_im = cv2.imread(image_paths_sub[frames[bubble][i]])
            outputs = predictor(new_im)
            masks = outputs["instances"].pred_masks.cpu()
            mask = np.uint8(masks[indexs[i]]) * 255
            class_val = []
            for j in range(len(contours[i])):
                x_coord = contour_connections[i][j][0]
                y_coord = contour_connections[i][j][1]
                if x_coord >= 832:
                    x_coord = 831
                elif x_coord <= 0:
                    x_coord = 0

                if y_coord <= 0:
                    y_coord = 0
                elif y_coord >= 600:
                    y_coord = 599

                if mask[y_coord][x_coord] == 255:
                    class_val.append(1)
                else:
                    class_val.append(0)
            direction.append(class_val)

        for i in range(len(direction)):
            for j in range(len(direction[i])):
                if direction[i][j] == 1:
                    mag[i][j] = mag[i][j] * -1

        length = len(mag[0])
        for i in range(len(mag)):
            if len(mag[i]) <= length:
                length = len(mag[i])

        num_entries = 200
        data = np.empty((len(mag), num_entries))
        # Calculate indices for evenly spaced entries
        for i in range(len(mag)):
            indices = np.linspace(0, len(mag[i]) - 1, num_entries, dtype=int)
            data[i:] = np.array(mag[i])[indices]
        # data=data[:-20]

        data1 = np.empty((len(mag), num_entries))
        # data1=data1[:-20]
        split_val = 100
        data1[:, 0:-split_val] = data[:, split_val:]
        data1[:, -split_val:] = data[:, 0:split_val]

        data = data1

        data_smoothed = gaussian_filter(data, sigma=2)

        # Generate some data
        total_time = len(data) / self.fps
        # Plot the data with a color bar

        image = plt.imshow(data_smoothed.T, extent=[0, total_time, 0, 200], aspect='auto', cmap='turbo', alpha=1)
        cbar = plt.colorbar()
        cbar.set_label('Velocity Magnitude (cm/s)')  # Set label for the color bar
        image.set_clim(vmin=-30, vmax=30)  # Set the range of values to display

        plt.xticks(np.arange(0, total_time, 0.01), [str(i) for i in np.arange(0, total_time, 0.01)])

        plt.xlabel('Time (s)')
        plt.ylabel('Location Along Bubble \n Perimeter')
        # Control the range of values displayed on the color bar
        saveloc = self.savefolder + f'velocity_{self.extension}_{bubble}.png'
        plt.savefig(saveloc, bbox_inches='tight')
        plt.show()

    def GenerateDataWClasses(self, classmodelweights, amt, thres=0.5):
        # Define CNN model
        import os

        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=6, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 128, kernel_size=6, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
                self.classifier = nn.Sequential(
                    nn.Linear(256 * 7 * 7, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 2)
                )

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        # Initialize the model
        model = CNN()
        import torch
        model.load_state_dict(torch.load(classmodelweights))
        transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # save locations
        file_path = self.savefolder + f'/bb-Boiling-{self.extension}.txt'
        output_file_path = self.savefolder + f'/bb-Boiling-output-{self.extension}.txt'
        vapor_file = self.savefolder + f'/vapor_{self.extension}.npy'
        vapor_base_file = self.savefolder + f'/vaporBase_bt-{self.extension}.npy'
        bubble_size_file = self.savefolder + f'/bubble_size_bt-{self.extension}.npy'
        bubind_file = self.savefolder + f'/bubind_{self.extension}.npy'
        frameind_file = self.savefolder + f'/frames_{self.extension}.npy'
        classind_file = self.savefolder + f'/class_{self.extension}.npy'
        bubclassind_file = self.savefolder + f'/bubclass_{self.extension}.npy'

        if not os.path.exists(self.savefolder):
            os.makedirs(self.savefolder)

        import torch
        torch.cuda.is_available()

        from detectron2.engine import DefaultTrainer
        # load model
        cfg = get_cfg()
        cfg.OUTPUT_DIR = self.modeldirectory
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 1000  # 1000 iterations seems good enough for this dataset
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Default is 512, using 256 for this dataset.
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, self.modelweights)  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
        predictor = DefaultPredictor(cfg)

        # Load Images
        import os
        import glob

        def get_image_paths(directory):
            """
            Get a list of file paths for all image files in the specified directory and its subdirectories.

            Args:
            directory (str): The directory to search for image files.

            Returns:
            List[str]: A list of file paths for all image files found, sorted alphabetically.
            """
            image_extensions = ['*.jpg']  # Add more extensions as needed

            image_paths = []
            for extension in image_extensions:
                pattern = os.path.join(directory, '**', extension)
                image_paths.extend(glob.glob(pattern, recursive=True))

            return sorted(image_paths)  # Sort the list of image paths alphabetically

        # hf=60
        # Specify the directory you want to search for image files
        # directory_path=f"/mnt/share/zdrive/Christy/Boiling-78/{hf}W"

        # Get a list of image file paths sorted alphabetically
        image_paths = get_image_paths(self.imagesfolder)

        # Print the sorted list of image file paths
        print(image_paths[0:10])

        import matplotlib.pyplot as plt
        image_paths_sub = image_paths[0:amt]

        from tqdm import tqdm
        Bounding_Box = np.empty((0, 7))
        bubble_size = []
        vapor = []
        vapor_base = []
        for i in tqdm(range(len(image_paths_sub))):
            # for i in [117,118,119]:
            new_im = cv2.imread(image_paths_sub[i])
            outputs = predictor(new_im)
            box = outputs["instances"].pred_boxes
            box = box.tensor
            box = box.cpu().tolist()
            masks = outputs["instances"].pred_masks.cpu()
            scores = outputs["instances"].scores
            # scores=scores.tensor
            scores = scores.cpu().tolist()
            new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
            new_im = new_im.reshape((new_im.shape[0], new_im.shape[1], 1))
            class_val = []
            for k in range(len(outputs['instances'].pred_masks)):
                if outputs['instances'].scores[k] > 0:
                    mask = outputs['instances'].pred_masks[k].cpu()
                    mask = np.reshape(np.array(mask, dtype=np.uint8), (mask.shape[0], mask.shape[1], 1))

                    object_image1 = np.zeros_like(new_im)
                    object_image1[mask > 0] = new_im[mask > 0]

                    object_image2 = np.zeros_like(new_im)
                    object_image2[mask <= 0] = new_im[mask <= 0]

                    mask = np.where(mask == 0, 0, 255).astype(np.uint8)

                    combined_image = np.concatenate((mask, object_image1, object_image2), axis=2)
                    mask_filename = f'./mask.png'
                    cv2.imwrite(mask_filename, combined_image)

                    input_tensor = transform(Image.open('./mask.png')).unsqueeze(0)

                    model.eval()
                    with torch.no_grad():
                        output = model(input_tensor)

                    _, predicted = torch.max(output, 1)

                    class_val.append(predicted.item())

                    # Map the predicted class to the class name
            class_val = np.array(class_val)

            converted_bounding_box = []
            for j in range(len(box)):
                x1, y1, x2, y2 = box[j]

                # if y2 > 502 and y2<533 and x1>320 and x1<515:
                # if y2 > 502 and y2<680 and x1>250 and x1<580:
                if y2 > 0:

                    converted_bounding_box.append([x1, y1, x2, y2])
                elif y2 > 502 and y2 < 533 and x2 > 320 and x2 < 515:
                    # elif y2 > 502 and y2<680 and x2>250 and x2<580:
                    converted_bounding_box.append([x1, y1, x2, y2])

            box = converted_bounding_box
            box_data = [[i + 1] + box[j] + [scores[j]] + [class_val[j]] for j in range(len(box))]

            if len(box_data) != 0:
                Bounding_Box = np.vstack([Bounding_Box, box_data])
            else:
                print('Error')
                outputs = predictor(new_im)
                box = outputs["instances"].pred_boxes
                box = box.tensor
                box = box.cpu().tolist()
                x1, y1, x2, y2 = box[0]
                box_data = [[i + 1] + box[j] + [scores[j]] + [class_val[j]] for j in range(len(box))]

                Bounding_Box = np.vstack([Bounding_Box, box_data])

            masks = outputs['instances'].pred_masks.cpu()
            scores = outputs['instances'].scores.cpu()
            masks, class_val, masks_base = filter_segmentation_outputs(
                masks, scores, class_val, thres
            )

            combined_mask = torch.any(masks, axis=0)
            vapor.append(torch.sum(combined_mask).item())
            combined_mask = torch.any(masks_base, axis=0)
            vapor_base.append(torch.sum(combined_mask).item())
            pixel_count = torch.sum(masks, dim=(1, 2)).numpy()
            bubble_size.append(pixel_count)

        np.save(vapor_file, vapor)
        np.save(vapor_base_file, vapor_base)
        np.save(bubble_size_file, bubble_size)

        # Define the file path where you want to save the data
        # file_path = 'ocsort-78-120W-base.txt'
        # file_path=f'./bb-Boiling78-{hf}W-base.txt'
        # Open the file for writing
        with open(file_path, 'w') as file:
            for sublist in Bounding_Box:
                formatted_values = []
                for value in sublist:
                    if isinstance(value, int):
                        formatted_values.append(f'{int(value)}')  # Format integers as strings
                    elif isinstance(value, float):
                        formatted_values.append(f'{value:.4f}')  # Format floats with 4 decimal places
                    else:
                        formatted_values.append(str(value))  # Keep other types as they are

                # Manually format the first and last elements as integers
                formatted_values[0] = str(int(float(formatted_values[0])))
                formatted_values[-1] = str(int(float(formatted_values[-1])))
                # Join the formatted values with spaces
                line = ','.join(formatted_values)

                # Write the line to the file and add a newline character
                file.write(line + '\n')

        # Confirmation message
        print(f'Data saved to {file_path}')
        tracker = ocsort.OCSort(det_thresh=thres, max_age=10, min_hits=20)
        img_info = (self.height, self.width)
        img_size = (self.height, self.width)
        boiling_test = 78
        hf = 120
        # cap=cv2.VideoCapture(f"/mnt/share/zdrive/Hari/Bubble hydro acoustic analysis/B83_210_Bubble/Bubble/video.avi")
        # cap=cv2.VideoCapture(f"/mnt/share/zdrive/Christy/Boiling-{boiling_test}/{hf}W-full.avi")
        video_file = self.videopath
        cap = cv2.VideoCapture(video_file)
        if (cap.isOpened() == False):
            print("Error opening video file")

        frames = []
        i = 0
        counter, fps, elapsed = 0, 0, 0
        frame_data = {}
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line into components
                parts = line.strip().split(',')

                # Extract frame ID and data
                frame_id = int(parts[0])
                data = list(map(float, parts[1:-1]))

                # Check if the frame ID is already in the dictionary
                if frame_id not in frame_data:
                    frame_data[frame_id] = []

                # Append the data to the corresponding frame ID
                frame_data[frame_id].append(data)

        with open(output_file_path, 'w') as file:

            while (cap.isOpened()):
                ret, frame = cap.read()

                if ret == True:
                    # load data from previously configured text file that has
                    # bounding boxes. (x1,y1,x2,y2,c)
                    xyxyc = frame_data.get(i + 1, [])
                    i += 1
                    tracks = tracker.update(np.array(xyxyc), img_info, img_size)

                    for track in tracker.trackers:
                        track_id = track.id
                        hits = track.hits
                        color = get_color(track_id * 15)
                        x1, y1, x2, y2 = np.round(track.get_state()).astype(int).squeeze()
                        '''
                        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                        cv2.putText(frame, f"{track_id}-{hits}",
                                    (x1+10, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    1,
                                    cv2.LINE_AA)
                        #frames.append(frame)
                        '''
                        file.write(f'{i},{track_id},{hits},{x1},{y1},{x2},{y2}\n')
                else:
                    break
            cap.release()
            del cap
            # load data
            real_data = np.loadtxt(file_path, delimiter=",")
            real = real_data[real_data[:, -2] >= thres]
            real = np.round(real[:, 0:-2]).astype(int)
            # del real_data
            pred_data = np.loadtxt(output_file_path, delimiter=",").astype(int)
            my_dict = {}

            # Assign keys ranging from 1 to 10 using a for loop
            for i in range(1, real[-1][0] + 1):
                my_dict[i] = []

            for item in real:
                key = item[0]
                value = item[1:]
                if my_dict[key] is []:
                    my_dict[key] = (list(value))
                else:
                    my_dict[key].append(list(value))
            realg = [values for values in my_dict.values()]

            # Initialize an empty dictionary
            my_dict = {}

            # Assign keys ranging from 1 to 10 using a for loop
            for i in range(1, real[-1][0] + 1):
                my_dict[i] = []

            for item in real:
                key = item[0]
                value = item[1]
                if my_dict[key] is []:
                    my_dict[key] = ((value))
                else:
                    my_dict[key].append((value))
            realgG = [values for values in my_dict.values()]

            # Initialize an empty dictionary
            my_dict = {}

            # Assign keys ranging from 1 to 10 using a for loop
            for i in range(1, real[-1][0] + 1):
                my_dict[i] = []

            for item in pred_data:
                key = item[0]
                value = item[1:]
                if my_dict[key] is []:
                    my_dict[key] = (list(value))
                else:
                    my_dict[key].append(list(value))
            predg = [values for values in my_dict.values()]

            import copy
            values = copy.deepcopy(realg)
            tracks = copy.deepcopy(realgG)

            # given previous frame and current frame remove row that doesn't have

            for k in range(len(predg) - 1):
                # for k in range(2):
                if len(predg[k]) > 0 and len(predg[k + 1]) > 0:
                    frame1 = predg[k]
                    frame2 = predg[k + 1]
                    vector1 = np.array(frame2)[:, 0].tolist()
                    vector2 = np.array(frame1)[:, 0].tolist()
                    result_vector = np.full(len(vector1), -1)
                    for i, val in enumerate(vector1):
                        if val in vector2:
                            result_vector[i] = vector2.index(val)
                    j = 0
                    for i in range(len(result_vector)):
                        if result_vector[i] != -1:
                            if frame2[i][1] != frame1[result_vector[i]][1]:
                                # print(k,j,i)
                                tracks[k + 1][j] = frame2[i][0]
                                values[k + 1][j][:] = frame2[i][2:]
                                j += 1
                        if result_vector[i] == -1:
                            tracks[k + 1][j] = frame2[i][0]
                            values[k + 1][j][:] = frame2[i][2:]
                            j += 1

            # Tracks is of shape (frames, bb in real detection (basically removes tracks with no hits))
            tracks[0] = np.array(predg[0])[:, 0].tolist()
            values[0] = np.array(predg[0])[:, 2:].tolist()

            for i in range(len(values)):
                if len(values[i]) > 0:
                    sort = np.argmax(iou_batch(realg[i], values[i]), axis=1).tolist()
                    tracks[i] = np.array(tracks[i])[sort].tolist()

            # Original data
            data = tracks

            # Find the maximum number in the original data
            # max_number = max(max(row) for row in data)
            max_number = max(max(sublist, default=float('-inf')) for sublist in data if sublist)

            # Create a list of lists to store the index positions and initial row numbers
            frames = [[] for _ in range(max_number + 1)]

            # Iterate through the original data and populate the result list
            for initial_row, row in enumerate(data):
                for index, number in enumerate(row):
                    frames[number].append(initial_row)

            # Print the result with initial row numbers
            print(frames[0])

            # Original data
            data = tracks
            # Find the maximum number in the original data
            # max_number = max(max(row) for row in data)
            max_number = max(max(sublist, default=float('-inf')) for sublist in data if sublist)

            # Create a list of lists to store the index positions
            bubInd = [[] for _ in range(max_number + 1)]

            # Iterate through the original data and populate the result list
            for row in data:
                for index, number in enumerate(row):
                    bubInd[number].append(index)

            # Print the result
            # print(bubInd)

            # Save Data
            bubInd = np.array(bubInd, dtype=object)
            np.save(bubind_file, bubInd)
            frames = np.array(frames, dtype=object)
            np.save(frameind_file, frames)

            classes = real_data[:, -1]

            # Initialize an empty dictionary
            my_dict = {}

            # Assign keys ranging from 1 to 10 using a for loop
            for i in range(1, real[-1][0] + 1):
                my_dict[i] = []

            for item in real_data:
                key = item[0]
                value = item[-1]
                if my_dict[key] is []:
                    my_dict[key] = ((value))
                else:
                    my_dict[key].append((value))
            realgG = [values for values in my_dict.values()]

            np.save(classind_file, realgG)

            bub_class = copy.deepcopy(frames)
            for j in range(len(bubInd)):
                bub = j
                for i in range(len(frames[bub])):
                    bub_class[j][i] = realgG[frames[bub][i]][bubInd[bub][i]]

            bub_class = np.array(bub_class, dtype=object)
            np.save(bubclassind_file, bub_class)


def TrainSegmentationModel(datapath, savename):
    register_coco_instances("my_dataset_train", {}, datapath, "")
    train_metadata = MetadataCatalog.get("my_dataset_train")
    train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

    cfg = get_cfg()
    cfg.OUTPUT_DIR = "./Models"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_MATLAB1.pth")  # path to the model we just trained
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 1000  # 1000 iterations seems good enough for this dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Default is 512, using 256 for this dataset.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # We have 1 classes.
    # NOTE: this config means the number of classes, without the background. Do not use num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    def custom_mapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        mean = 0
        std_dev = 25
        gaussian_noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
        # noisy_image = cv2.add(image, gaussian_noise)

        transform_list = [
            # T.Resize((800,600)),
            T.RandomBrightness(0.8, 1.8),
            T.RandomContrast(0.6, 1.3),
            T.RandomSaturation(0.8, 1.4),
            # T.RandomRotation(angle=[90, 90]),
            # T.RandomNoise(mean=0.0, std=0.1),
            T.RandomLighting(0.7),
            T.RandomFlip(prob=0.4, horizontal=True, vertical=False),
        ]
        image, transforms = T.apply_transform_gens(transform_list, image)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

    class CustomTrainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, cfg):
            return build_detection_train_loader(cfg, mapper=custom_mapper)

    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()  # Start the training process

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, savename)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)


def FineTuneModel(savefolder, pastmodel, traincoco, valcoco,device):
    register_coco_instances("train", {}, traincoco, "")
    register_coco_instances("val", {}, valcoco, "")

    # Custom data mapper with augmentations
    def custom_mapper(dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        augs = [
            T.ResizeShortestEdge(short_edge_length=512, max_size=800),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomRotation(angle=[-10, 10]),
            T.RandomBrightness(0.8, 1.2),  # Brightness jitter
            T.RandomContrast(0.8, 1.2),  # Contrast jitter
            T.RandomSaturation(0.8, 1.2),  # Saturation jitter
            T.RandomCrop("relative", (0.8, 0.8))
        ]

        image, transforms = T.apply_transform_gens(augs, image)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
        ]
        dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[:2])
        return dataset_dict

    # Config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)  # Enable validation
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = pastmodel  # pretrained or previous checkpoint
    cfg.MODEL.BACKBONE.FREEZE_AT = 5
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 1e-5
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.DEVICE= device
    cfg.TEST.EVAL_PERIOD = 100  # Evaluate every 100 iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = savefolder
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Custom trainer with evaluation and best model saving
    class BestModelTrainer(DefaultTrainer):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.best_ap = -1.0

        @classmethod
        def build_train_loader(cls, cfg):
            return build_detection_train_loader(cfg, mapper=custom_mapper)

        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
            return COCOEvaluator(dataset_name, cfg, False, output_folder)

        def after_step(self):
            super().after_step()
            if self.iter % self.cfg.TEST.EVAL_PERIOD == 0 and self.iter > 0:
                results = self.test(self.cfg, self.model)
                score = results.get("bbox", results.get("segm", {})).get("AP", 0)
                if score > self.best_ap:
                    self.best_ap = score
                    self.checkpointer.save("model_best")
                    print(f"New best model saved at iter {self.iter} with AP={score:.3f}")

    # Train the model
    trainer = BestModelTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def TrainCNNClassification(savename):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms for data augmentation and normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((480, 640)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Set data directory
    data_dir = './output'

    # Create datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    # Create dataloaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in
                   ['train', 'val']}

    # Get dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # Define CNN model
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=6, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=6, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(256 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 2)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # Initialize the model
    model = CNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initialize variables to keep track of best accuracy and corresponding model weights
    best_accuracy = 0.0
    best_model_weights = None

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Check if current phase is validation and if current accuracy is better than the best accuracy
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                # Save the model weights
                best_model_weights = model.state_dict()

    # Save the best model weights
    torch.save(best_model_weights, savename)
    print("Training complete!")
