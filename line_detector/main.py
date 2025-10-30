from os.path import join
import numpy as np
import operator
import cv2
import fnmatch
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import glob
import imutils
from math import *
from scipy.stats import mode
from distutils.dir_util import copy_tree
from IPython.display import Image
import PIL.Image
import os.path
import shutil

import torch
from IPython.display import Image, clear_output  # to display images

from line_detector.utils import sorting, rotation

def plot_fig(img, size = 15):
  plt.figure(figsize=(size, size))
  plt.imshow(imutils.opencv2matplotlib(img))
  plt.show()
  
def yolo_detection(img_path, img_size, conf, img_name, detections_path):
    import subprocess
    subprocess.run([
        'python', './yolov5/detect.py', 
        '--weights', './model/line_model_best.pt', 
        '--img', str(img_size), 
        '--conf', str(conf), 
        '--source', img_path, 
        '--save-conf', 
        '--save-txt', 
        '--project', detections_path, 
        '--name', img_name  # Specifica il nome della cartella
    ])
    
def sort_initial_detection(detection_dir, ):
    # Sorting Labels of 1st detection on the basis of y...
    txt_loc = f"{detection_dir}/labels/"
    new_sort_label = f'{detection_dir}/sorted_line_after_1st_detection/'
    flag = 0
    sorting.sort_detection_label(txt_loc, new_sort_label, flag)
    
def initial_line_segmentation(img, img_lb, label, segmented_img_path, flag = 0):
  pred_lb = os.listdir(label)
  print(pred_lb)
  pred_lb2 = str(pred_lb[0])
  pred_lb3 = label + pred_lb[0]

  dir = segmented_img_path
  # dir = "/content/initial_line_segmantation"
  os.mkdir(dir)
  img1 = cv2.imread(img)
  dh, dw, _ = img1.shape
  txt_lb = open(pred_lb3, 'r')
  txt_lb_data = txt_lb.readlines()
  txt_lb.close()
  img_lb2 = img_lb
  
  k=1
  for dt in txt_lb_data:
      if flag != 0:
        _, x, y, w, h = map(float, dt.split(' '))
      else:
        _, x, y, w, h, conf = map(float, dt.split(' '))
        
      if w > 0.50 and w < 0.80 and flag == 0:
        x = 0.5
        w = 1.0
      l = int((x - w / 2) * dw)
      r = int((x + w / 2) * dw)
      t = int((y - h / 2) * dh)
      b = int((y + h / 2) * dh)

      crop = img1[t:b, l:r]
      cv2.imwrite("{}/{}_{}.jpg".format(dir, img_lb2, k), crop)
      k += 1

def crop_image(bb_data, destination, image, img_lb, dh, dw):
    x = float(bb_data[1])
    y = float(bb_data[2])
    w = float(bb_data[3])
    h = float(bb_data[4])
  
    # x = 0.5
    # w  = 1.0
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    crop = image[t:b, l:r]
    filename = destination+img_lb
    cv2.imwrite(filename,crop)
    print("Segmented successfully!\n")
    
def final_segmentation(img, img_path, label, label_path, segmented_img_path):
  dir = segmented_img_path
  print("Image path -> ",img_path)
  img1 = cv2.imread(img_path)
  dh, dw, _ = img1.shape
  txt_lb = open(label_path, 'r')
  txt_lb_data = txt_lb.readlines()
  txt_lb.close()
  img_name = img
  
  max_w = 0
  data1 = []
  for line in txt_lb_data:
      token = line.split()
      data1.append(token)
  
  if len(data1)==1:
      bb_data = data1[0]
      wdth = float(bb_data[3])
      if wdth>0.4:
          crop_image(bb_data,dir,img1,img_name,dh,dw,)
      else:
          filename = dir+img_name
          cv2.imwrite(filename,img1)
  elif len(data1)==2:
      bb_data1 = data1[0]
      bb_data2 = data1[1]
      w1 = float(bb_data1[3]) 
      w2 = float(bb_data2[3])
      c1 = float(bb_data1[5]) 
      c2 = float(bb_data2[5])
      if w1 <= 0.5 and w2 <= 0.5:
        if c1 >= 0.8 and c2 >= 0.8:
          sorted_bb_data = sorted(data1, key=operator.itemgetter(5))
          bb_data = sorted_bb_data[-1]
          crop_image(bb_data,dir,img1,img_name,dh,dw,)
        else:
          filename = dir+img_name
          cv2.imwrite(filename,img1)
      else:
        sorted_bb_data = sorted(data1, key=operator.itemgetter(3))
        bb_data = sorted_bb_data[-1]
        crop_image(bb_data,dir,img1,img_name,dh,dw,)
  elif len(data1)==3:
      sorted_bb_data = sorted(data1, key=operator.itemgetter(2))
      bb_data = sorted_bb_data[1]
      crop_image(bb_data,dir,img1,img_name,dh,dw,) 
  else:
      sorted_bb_data = sorted(data1, key=operator.itemgetter(3))
      bb_data = sorted_bb_data[-1]
      crop_image(bb_data,dir,img1,img_name,dh,dw,)
      
def rotate_lines(first_detection, directory):
  line_path = first_detection
  line_dir = sorting.line_sort(os.listdir(line_path))
  print(line_dir)

  for img in line_dir:
      rotation.ready_for_rotate(line_path, img, directory)

if __name__ == '__main__':
    test_images_dir = '/home/filippo/projects/line_detector/test_images/'
    
    # create initial and final detections directories
    os.makedirs('./initial_detections', exist_ok=True)
    os.makedirs('./final_detections', exist_ok=True)
    os.makedirs('./lines', exist_ok=True)

    for img_file in os.listdir(test_images_dir):
        img_path = os.path.join(test_images_dir, img_file)
        img_size = 640
        conf = 0.30
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # yolo detection
        yolo_detection(img_path, img_size, conf, img_name, f"./initial_detections/")
        sort_initial_detection(f"./initial_detections/{img_name}")
        
        # Line Segmentation after 1st Detection
        sorted_label = f"./initial_detections/{img_name}/sorted_line_after_1st_detection/"
        filename = f"./initial_detections/{img_name}/initial_line_segmentation/"
        initial_line_segmentation(img_path, img_name, sorted_label, filename)

        # Rotation..Rotating image after 1st line Segmentation...
        rotate_line = f"./initial_detections/{img_name}/Rotated_line_by_HaughLine_Affine/"
        os.mkdir(rotate_line)

        rotate_line_Dskew = f"./initial_detections/{img_name}/DSkew/"
        os.mkdir(rotate_line_Dskew)

        rotate_line_Haughline = f"./initial_detections/{img_name}/HaughLine_Affine/"
        os.mkdir(rotate_line_Haughline)

        rotate_lines(filename, f"./initial_detections/{img_name}")
        
        # Yolo 2nd Detection...
        rotated_img_path = f"./initial_detections/{img_name}/Rotated_line_by_HaughLine_Affine/"
        img_size = 640
        conf = 0.50
        yolo_detection(rotated_img_path, img_size, conf, img_name, f"./final_detections/")
        
        # Showing 2nd Detection result...
        second_det = f"./final_detections/{img_name}/"
        x = os.listdir(second_det)
        x.remove('labels')
        x = sorting.line_sort(x)

        # Cropping Rotated image with 2nd detection label...
        # Target Images
        target_image_path1 = rotated_img_path
        target_image_path = f"./final_detections/{img_name}/2nd line detection for rotated images/"
        if os.path.exists(target_image_path1):  # Copying and removing 2nd detection label if exists...
            to_dir = target_image_path
            shutil.copytree(target_image_path1, to_dir)

        # Target Images labels
        target_label_path = f"./final_detections/{img_name}/labels/"

        target_image = os.listdir(target_image_path)
        target_label = os.listdir(target_label_path)

        new_dir = f"./final_detections/{img_name}/final_line_segmentation/"
        os.mkdir(new_dir)

        for i in target_image:
            for j in target_label:
                fn_i = i.split(".")
                fn_j = j.split(".")
                if fn_i[0] == fn_j[0]:
                    # Line Segmentation after 1st Detection...
                    # Final Line Segmentation...
                    image_path = target_image_path + i
                    img = i
                    sorted_label = j
                    sorted_label_path = target_label_path + j
                    final_segmentation(img, image_path, sorted_label, sorted_label_path, new_dir)
                   
        # copy final line segmentation to a new directory
        to_dir = f"./lines/{img_name}/final_line_segmentation/"
        shutil.copytree(new_dir, to_dir)
        to_dir = f"./lines/{img_name}/initial_line_segmentation/"
        shutil.copytree(rotated_img_path, to_dir)
        # copy initial image file to a new directory
        to_dir = f"./lines/{img_name}/"
        shutil.copy(img_path, to_dir)
                
    # delete initial detections
    shutil.rmtree(f"./initial_detections/{img_name}")
    # delete final detections
    shutil.rmtree(f"./final_detections/{img_name}")