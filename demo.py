import os
import sys
import re
import numpy as np
import operator
# Keep the existing backend but add explicit matplotlib configuration
import matplotlib
from torchvision.transforms import ToTensor
matplotlib.rcParams['figure.figsize'] = [6, 8]
matplotlib.rcParams['figure.dpi'] = 100
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt, QTimer, QSize
import cv2
import torch
from model.HTR_VT import create_model
from utils.utils import CTCLabelConverter
from data.PONTALTO.pontalto import PONTALTO

import torchvision.transforms.functional as T

import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
							 QHBoxLayout, QPushButton, QFileDialog, QLabel, 
							 QSplitter, QMessageBox, QSizePolicy, QScrollArea)

import line_detector.utils.sorting as sorting
import line_detector.utils.rotation as rotation
import shutil

# Path to your model - adjust as needed
CHECKPOINT_PATH = "./checkpoints/finetune_vmamba_pontalto2_full/best_CER.pth"

# Constants for operation status messages
STATUS_IDLE = "Ready"
STATUS_LOADING = "Loading model..."
STATUS_PROCESSING = "Processing image..."
STATUS_SEGMENTING = "Segmenting lines..."
STATUS_RECOGNIZING = "Recognizing text..."
STATUS_COMPLETE = "Recognition complete"
STATUS_ERROR = "Error during processing"

class HTRDemo(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Handwritten Text Recognition Demo")
		self.setGeometry(100, 100, 1200, 800)
		
		# Initialize variables
		self.image_path = None
		self.loaded_image = None
		self.lines = []
		self.predictions = []
		
		# Set up the UI
		self.init_ui()
		
		# Try to load model
		self.model = self.load_model()
		self.model.eval()
		print("Model loaded successfully")
		
	def init_ui(self):
		# Main layout

		main_widget = QWidget()
		main_layout = QVBoxLayout()
		main_layout.setContentsMargins(0, 0, 0, 0)
		main_layout.setSpacing(0)
		
		# Splitter for left (original image) and right (predictions) sides
		self.splitter = QSplitter()
		self.splitter.setChildrenCollapsible(False)
		
		# Left side - Original Image
		self.left_widget = QWidget()
		left_layout = QVBoxLayout()
		left_layout.setContentsMargins(4, 4, 4, 4)
		self.left_widget.setLayout(left_layout)
		self.left_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		
		# Create a label for status/debugging
		self.status_label1 = QLabel("Ready")
		
		# Create matplotlib figures
		self.orig_fig = plt.figure()
		self.orig_canvas = FigureCanvas(self.orig_fig)
		self.orig_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		self.orig_canvas.setMinimumSize(QSize(300, 400))
		
		# Create a label for status/debugging
		self.status_label2 = QLabel("Waiting for an image.")
		
		# Add widgets to layout
		left_layout.addWidget(self.orig_canvas, 1)
		left_layout.addWidget(self.status_label1, 0)
		
		# Right side - Predictions with scrolling
		self.right_widget = QWidget()
		
		# Create a scroll area for the right widget
		self.scroll_area = QScrollArea()
		self.scroll_area.setWidgetResizable(True)
		self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
		self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
		
		# Create content widget for the scroll area
		self.prediction_content = QWidget()
		self.prediction_layout = QVBoxLayout(self.prediction_content)
		self.prediction_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		self.prediction_layout.setContentsMargins(4, 4, 4, 4)
		self.prediction_layout.setSpacing(10)
		
		# Set the content widget to the scroll area
		self.scroll_area.setWidget(self.prediction_content)
		
		 # Create status label with better visibility
		self.status_label2 = QLabel("Waiting for an image.")
		self.status_label2.setMinimumHeight(30)
		self.status_label2.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-top: 1px solid #cccccc;")
		self.status_label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
		
		# Add scroll area and status label to right widget with proper layout settings
		right_layout = QVBoxLayout(self.right_widget)
		right_layout.setContentsMargins(0, 0, 0, 0)
		right_layout.setSpacing(0)
		right_layout.addWidget(self.scroll_area, 1)  # Add weight of 1
		right_layout.addWidget(self.status_label2, 0)  # Add weight of 0 (fixed height)
		self.right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		
		# Add widgets to splitter
		self.splitter.addWidget(self.left_widget)
		self.splitter.addWidget(self.right_widget)
		self.splitter.setSizes([self.width() // 2, self.width() // 2])
		
		# Bottom controls
		bottom_widget = QWidget()
		bottom_layout = QHBoxLayout(bottom_widget)
		bottom_layout.setContentsMargins(10, 10, 10, 10)
		
		# Load button
		self.load_btn = QPushButton("Load Image")
		self.load_btn.clicked.connect(self.load_image)
		self.load_btn.setMinimumHeight(40)
		bottom_layout.addWidget(self.load_btn)
		
		# Process button (disabled initially)
		self.process_btn = QPushButton("PROCESS")
		self.process_btn.clicked.connect(self.process_image)
		self.process_btn.setEnabled(False)
		self.process_btn.setMinimumHeight(40)
		bottom_layout.addWidget(self.process_btn)
		
		# Add splitter and bottom controls to main layout
		main_layout.addWidget(self.splitter, 9)
		main_layout.addWidget(bottom_widget, 1)
		
		main_widget.setLayout(main_layout)
		self.setCentralWidget(main_widget)
	
	def load_image(self):
		#options = QFileDialog.Option.
		file_path, _ = QFileDialog.getOpenFileName(
			self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
		)

		print("Attempting to load image: ", file_path)
		
		if file_path:
			try:
				self.image_path = file_path
				# Load and display image
				self.loaded_image = cv2.imread(file_path)
				if self.loaded_image is None:
					raise Exception(f"Failed to load image from {file_path}")
					
				self.loaded_image = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB)
				
				# Set status
				self.status_label1.setText(f"Loaded image: {os.path.basename(file_path)} - {self.loaded_image.shape}")
				
				# Use a timer to delay the display slightly - sometimes helps with refresh issues
				QTimer.singleShot(100, self.display_original_image)
				self.status_label2.setText("Ready to process.")
				
				self.process_btn.setEnabled(True)
				
				# Clear predictions display when loading new image
				self.clear_predictions()
				print('Drawing finished')
				
			except Exception as e:
				QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
				self.process_btn.setEnabled(False)
				
	def clear_predictions(self):
		# Clear all items from prediction layout
		while self.prediction_layout.count():
			item = self.prediction_layout.takeAt(0)
			widget = item.widget()
			if widget is not None:
				widget.deleteLater()
	
	def display_original_image(self):
		if self.loaded_image is None:
			print("No image loaded")
			self.status_label1.setText("Error: No image loaded")
			return
			
		# Try completely recreating the figure - sometimes helps with refresh issues
		plt.close(self.orig_fig)
		self.orig_fig = plt.figure()
		
		# Get the layout and replace the canvas
		layout = self.left_widget.layout()
		if self.orig_canvas is not None:
			layout.removeWidget(self.orig_canvas)
			self.orig_canvas.deleteLater()
			
		self.orig_canvas = FigureCanvas(self.orig_fig)
		self.orig_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
		self.orig_canvas.setMinimumSize(QSize(300, 400))
		layout.insertWidget(0, self.orig_canvas, 1)  # Add stretch factor 1
		
		# Now plot on the fresh figure
		ax = self.orig_fig.add_subplot(111)
		ax.imshow(self.loaded_image)
		ax.axis('off')
		self.orig_fig.tight_layout()
		
		# Force draw
		self.orig_canvas.draw_idle()
		self.orig_canvas.update()
		
		# Force Qt to process events
		QApplication.processEvents()
		
		print(f"Canvas updated with image shape {self.loaded_image.shape}")
		self.status_label1.setText(f"Displayed: {os.path.basename(self.image_path)} - {self.loaded_image.shape}")
	
	def process_image(self):
		if self.loaded_image is None:
			return
		
		# Step 1: Segment lines
		self.status_label2.setText("Starting segmentation of the page in lines...")
		QApplication.processEvents()
		self.segment_lines()
		print('Segmented')
		
		# Step 2: Predict text for each line
		self.status_label2.setText(f"Segmentation completed, starting prediction process...")
		QApplication.processEvents()
		self.predict_lines()
		print('Predicted')

		# Step 3: Display predictions
		self.status_label2.setText(f"Displaying the results.")
		QApplication.processEvents()
		self.display_predictions()
		print('Displayed')
	
	def sort_initial_detection(self, detection_dir, ):
		# Sorting Labels of 1st detection on the basis of y...
		txt_loc = f"{detection_dir}/labels/"
		new_sort_label = f'{detection_dir}/sorted_line_after_1st_detection/'
		flag = 0
		sorting.sort_detection_label(txt_loc, new_sort_label, flag)

	def initial_line_segmentation(self, img, img_lb, label, segmented_img_path, flag = 0):
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

	def rotate_lines(self, first_detection, directory):
		line_path = first_detection
		line_dir = sorting.line_sort(os.listdir(line_path))
		print(line_dir)

		for img in line_dir:
			rotation.ready_for_rotate(line_path, img, directory)

	def crop_image(self, bb_data, destination, image, img_lb, dh, dw):
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

	def final_segmentation(self, img, img_path, label, label_path, segmented_img_path):
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
				self.crop_image(bb_data,dir,img1,img_name,dh,dw,)
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
					self.crop_image(bb_data,dir,img1,img_name,dh,dw,)
				else:
					filename = dir+img_name
					cv2.imwrite(filename,img1)
			else:
				sorted_bb_data = sorted(data1, key=operator.itemgetter(3))
				bb_data = sorted_bb_data[-1]
				self.crop_image(bb_data,dir,img1,img_name,dh,dw,)
		elif len(data1)==3:
			sorted_bb_data = sorted(data1, key=operator.itemgetter(2))
			bb_data = sorted_bb_data[1]
			self.crop_image(bb_data,dir,img1,img_name,dh,dw,) 
		else:
			sorted_bb_data = sorted(data1, key=operator.itemgetter(3))
			bb_data = sorted_bb_data[-1]
			self.crop_image(bb_data,dir,img1,img_name,dh,dw,)

	def segment_lines(self):

		img_path = self.image_path
	
		# create initial and final detections directories
		if os.path.exists('./initial_detections'):
			shutil.rmtree('./initial_detections')
		if os.path.exists('./final_detections'):
			shutil.rmtree('./final_detections')
		if os.path.exists('./lines'):
			shutil.rmtree('./lines')		

		os.makedirs('./initial_detections', exist_ok=True)
		os.makedirs('./final_detections', exist_ok=True)
		os.makedirs('./lines', exist_ok=True)

		conf = 0.30
		img_name = os.path.splitext(os.path.basename(img_path))[0]
		img_size = 640
		
		# yolo detection
		self.yolo_detection(img_path, img_size, conf, img_name, f"./initial_detections/")
		self.sort_initial_detection(f"./initial_detections/{img_name}")
		
		# Line Segmentation after 1st Detection
		sorted_label = f"./initial_detections/{img_name}/sorted_line_after_1st_detection/"
		filename = f"./initial_detections/{img_name}/initial_line_segmentation/"
		self.initial_line_segmentation(img_path, img_name, sorted_label, filename)

		# Rotation..Rotating image after 1st line Segmentation...
		rotate_line = f"./initial_detections/{img_name}/Rotated_line_by_HaughLine_Affine/"
		os.mkdir(rotate_line)

		rotate_line_Dskew = f"./initial_detections/{img_name}/DSkew/"
		os.mkdir(rotate_line_Dskew)

		rotate_line_Haughline = f"./initial_detections/{img_name}/HaughLine_Affine/"
		os.mkdir(rotate_line_Haughline)

		self.rotate_lines(filename, f"./initial_detections/{img_name}")
		
		#Yolo 2nd Detection...
		# rotated_img_path = f"./initial_detections/{img_name}/Rotated_line_by_HaughLine_Affine/"
		# img_size = 640
		# conf = 0.30
		# self.yolo_detection(rotated_img_path, img_size, conf, img_name, f"./final_detections/")
		
		# # Showing 2nd Detection result...
		# second_det = f"./final_detections/{img_name}/"
		# x = os.listdir(second_det)
		# x.remove('labels')
		# x = sorting.line_sort(x)

		# #Cropping Rotated image with 2nd detection label...
		# #Target Images
		# target_image_path1 = rotated_img_path
		# target_image_path = f"./final_detections/{img_name}/2nd line detection for rotated images/"
		# if os.path.exists(target_image_path1):  # Copying and removing 2nd detection label if exists...
		# 	to_dir = target_image_path
		# 	shutil.copytree(target_image_path1, to_dir)

		# #Target Images labels
		# target_label_path = f"./final_detections/{img_name}/labels/"

		# target_image = os.listdir(target_image_path)
		# target_label = os.listdir(target_label_path)

		# new_dir = f"./final_detections/{img_name}/final_line_segmentation/"
		# os.mkdir(new_dir)

		# for i in target_image:
		# 	for j in target_label:
		# 		fn_i = i.split(".")
		# 		fn_j = j.split(".")
		# 		if fn_i[0] == fn_j[0]:
		# 			# Line Segmentation after 1st Detection...
		# 			# Final Line Segmentation...
		# 			image_path = target_image_path + i
		# 			img = i
		# 			sorted_label = j
		# 			sorted_label_path = target_label_path + j
		# 			self.final_segmentation(img, image_path, sorted_label, sorted_label_path, new_dir)
				
		# copy final line segmentation to a new directory
		# to_dir = f"./lines/{img_name}/final_line_segmentation/"
		# shutil.copytree(new_dir, to_dir)
		# to_dir = f"./lines/{img_name}/initial_line_segmentation/"
		# shutil.copytree(rotated_img_path, to_dir)
		# copy initial image file to a new directory
		to_dir = f"./lines/{img_name}/"
		os.makedirs(to_dir, exist_ok=True)
		shutil.copy(img_path, to_dir)
					
		# delete initial detections
		# shutil.rmtree(f"./initial_detections/{img_name}")
		# # delete final detections
		# shutil.rmtree(f"./final_detections/{img_name}")
	
	def yolo_detection(self, img_path, img_size, conf, img_name, detections_path):
		import subprocess
		subprocess.run([
			'python', 'line_detector/yolov5/detect.py', 
			'--weights', 'line_detector/model/line_model_best.pt', 
			'--img', str(img_size), 
			'--conf', str(conf), 
			'--source', img_path, 
			'--save-conf', 
			'--save-txt', 
			'--project', detections_path, 
			'--name', img_name  # Specifica il nome della cartella
		])
		
	def load_model(self):
		"""Load the vmamba model for HTR"""
			
		# Create model with appropriate parameters
		args = type('', (), {})()  # Create a simple object to hold attributes
		args.use_mamba = True
		
		model = create_model(nb_cls=90, img_size=[512, 64], args=args)
		
		# Load the checkpoint (note correct capitalization: CER not cer)
		checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
		model.load_state_dict(checkpoint['model'])
		
		# Set model to evaluation mode
		model.eval()
		
		# Move to GPU if available
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model = model.to(device)
		self.device = device
		
		print(f"Model loaded from {CHECKPOINT_PATH}")
		print(f"Using device: {device}")
		
		return model
	
	def predict_lines(self):
		"""Run the prediction model on segmented lines"""
		# Get the base name of the image
		img_base = os.path.splitext(os.path.basename(self.image_path))[0]
		lines_dir = f"./initial_detections/{img_base}/initial_line_segmentation"

		# Check if directory exists
		if not os.path.exists(lines_dir):
			self.status_label2.setText(f"Error: Line images directory not found at {lines_dir}")
			return

		# List and sort line images
		line_files = sorted([f for f in os.listdir(lines_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

		if not line_files:
			self.status_label2.setText("No line images found")
			return

		self.lines = []
		tensor_lines = []

		# Sort line files in natural order
		def natural_sort_key(s):
			"""Sort strings with numbers in natural order"""
			return [int(text) if text.isdigit() else text.lower() 
					for text in re.split(r'(\d+)', s)]
					
		line_files = sorted(line_files, key=natural_sort_key)

		# Process each line image
		for line_file in line_files:
			line_path = os.path.join(lines_dir, line_file)
			
			# Read the image
			img = cv2.imread(line_path)
			if img is None:
				print(f"Failed to load {line_path}")
				continue
			
			# Convert to RGB
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			self.lines.append(img)
			
			# Preprocess image for the model
			img_resized = cv2.resize(img, (512, 64))  # Resize to model input size
			
			img_tensor = torch.from_numpy(img_resized).unsqueeze(0)  # Add batch dimension
			img_tensor = img_tensor.to(torch.float32).to(self.device)
			img_tensor = T.rgb_to_grayscale(img_tensor.permute(0,3,1,2))
			tensor_lines.append(img_tensor)

		if not self.lines:
			self.status_label2.setText("No valid line images could be processed")
			return

		self.status_label2.setText(f"Successfully loaded {len(self.lines)} lines")
		self.predictions = []
		
		dset = PONTALTO('./data/PONTALTO' , 'basic', ToTensor(), img_size=[512, 64], nameset='train')
		converter = CTCLabelConverter(dset.charset)
		for i, line_img in enumerate(tensor_lines):
			with torch.no_grad():
				prediction = self.model(line_img)

				preds_size = torch.IntTensor([prediction.size(1)])
				_, preds_index = prediction.max(2)
				preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
				preds_str = converter.decode(preds_index.data, preds_size.data)


				self.predictions.append(preds_str)
				print(f"Line {i+1} prediction: {prediction}")
	
	def display_predictions(self):
		"""Display the predictions alongside the segmented lines"""
		if not self.lines or not self.predictions:
			return
		
		# Clear previous predictions
		self.clear_predictions()
		# For each line and its prediction
		for i, (line_img, prediction) in enumerate(zip(self.lines, self.predictions)):
			# Create a container widget for this line
			line_container = QWidget()
			line_layout = QVBoxLayout(line_container)
			line_layout.setContentsMargins(0, 0, 0, 10)
			
			# Create a figure for the line image that fills width
			line_fig = plt.figure(figsize=(8, 2))
			line_canvas = FigureCanvas(line_fig)
			line_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
			
			# Plot the line image
			ax = line_fig.add_subplot(111)
			ax.imshow(line_img)
			ax.axis('off')
			line_fig.tight_layout(pad=0)
			
			# Create a label for the prediction text
			pred_label = QLabel(prediction[0])
			pred_label.setWordWrap(True)
			pred_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
			pred_label.setStyleSheet("font-size: 22px; font-weight: bold; padding: 5px; subcontrol-position: center;")
			
			# Add widgets to the layout
			line_layout.addWidget(line_canvas)
			line_layout.addWidget(pred_label)
			
			# Add the container to the main prediction layout
			self.prediction_layout.addWidget(line_container)
		
		# Make sure UI updates - force update of status label
		self.status_label2.setText(f"Displayed {len(self.predictions)} lines with predictions")
		self.status_label2.repaint()
		QApplication.processEvents()

if __name__ == "__main__":

	app = QApplication(sys.argv)
	window = HTRDemo()
	window.show()
	sys.exit(app.exec())