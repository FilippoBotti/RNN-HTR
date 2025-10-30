import os
import sys
import re
import numpy as np
import operator
import cv2
import torch
from torchvision.transforms import ToTensor
import gradio as gr
import shutil
import subprocess
from model.HTR_VT import create_model
from utils.utils import CTCLabelConverter
from data.PONTALTO2.pontalto import PONTALTO

import matplotlib.pyplot as plt
import torchvision.transforms.functional as T

import line_detector.utils.sorting as sorting
import line_detector.utils.rotation as rotation

# Path to your model - adjust as needed
CHECKPOINT_PATH = "./checkpoints/finetune_vmamba_pontalto2/best_CER.pth"

def load_model():
    """Load the vmamba model for HTR"""
    # Create model with appropriate parameters
    args = type('', (), {})()  # Create a simple object to hold attributes
    args.use_mamba = True
    
    model = create_model(nb_cls=90, img_size=[512, 64], args=args)
    
    # Load the checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model loaded from {CHECKPOINT_PATH}")
    print(f"Using device: {device}")
    
    return model, device

def sort_initial_detection(detection_dir):
    # Sorting Labels of 1st detection on the basis of y...
    txt_loc = f"{detection_dir}/labels/"
    new_sort_label = f'{detection_dir}/sorted_line_after_1st_detection/'
    flag = 0
    sorting.sort_detection_label(txt_loc, new_sort_label, flag)

def initial_line_segmentation(img, img_lb, label, segmented_img_path, flag=0):
    pred_lb = os.listdir(label)
    
    #pred_lb2 = str(pred_lb[0])
    pred_lb3 = label + pred_lb[0]
    
    dir = segmented_img_path
    os.makedirs(dir, exist_ok=True)
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

def rotate_lines(first_detection, directory):
    line_path = first_detection
    print('line path', first_detection)
    print(os.listdir(line_path))
    line_dir = sorting.line_sort(os.listdir(line_path))
    print('line dir :' ,line_dir)

    for img in line_dir:
        rotation.ready_for_rotate(line_path, img, directory)

def yolo_detection(img_path, img_size, conf, img_name, detections_path):
    subprocess.run([
        'python', 'line_detector/yolov5/detect.py', 
        '--weights', 'line_detector/model/line_model_best.pt', 
        '--img', str(img_size), 
        '--conf', str(conf), 
        '--source', img_path, 
        '--save-conf', 
        '--save-txt', 
        '--project', detections_path, 
        '--name', img_name
    ])

def segment_lines(img_path):
    """Segment the input image into lines of text"""
    # Create initial and final detections directories
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
    
    # YOLO detection
    yolo_detection(img_path, img_size, conf, img_name, f"./initial_detections/")
    sort_initial_detection(f"./initial_detections/{img_name}")
    
    # Line Segmentation after 1st Detection
    sorted_label = f"./initial_detections/{img_name}/sorted_line_after_1st_detection/"
    filename = f"./initial_detections/{img_name}/initial_line_segmentation/"
    initial_line_segmentation(img_path, img_name, sorted_label, filename)

    # Rotation - Rotating image after 1st line Segmentation
    rotate_line = f"./initial_detections/{img_name}/Rotated_line_by_HaughLine_Affine/"
    os.makedirs(rotate_line, exist_ok=True)

    rotate_line_Dskew = f"./initial_detections/{img_name}/DSkew/"
    os.makedirs(rotate_line_Dskew, exist_ok=True)

    rotate_line_Haughline = f"./initial_detections/{img_name}/HaughLine_Affine/"
    os.makedirs(rotate_line_Haughline, exist_ok=True)

    rotate_lines(filename, f"./initial_detections/{img_name}")
    
    # Copy to lines directory
    to_dir = f"./lines/{img_name}/"
    os.makedirs(to_dir, exist_ok=True)
    shutil.copy(img_path, to_dir)
    
    return filename  # Return path to segmented lines

def predict_lines(image_path, model, device):
    """Run the prediction model on segmented lines"""
    # Get the base name of the image
    img_base = os.path.splitext(os.path.basename(image_path))[0]
    lines_dir = f"./initial_detections/{img_base}/initial_line_segmentation"

    # Check if directory exists
    if not os.path.exists(lines_dir):
        return [], []

    # List and sort line images
    line_files = sorted([f for f in os.listdir(lines_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    if not line_files:
        return [], []

    lines = []
    tensor_lines = []

    # Sort line files in natural order
    def natural_sort_key(s):
        """Sort strings with numbers in natural order"""
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split(r'(\d+)', s)]
                
    line_files = sorted(line_files, key=natural_sort_key)
    print(line_files)

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
        lines.append(img)
        
        # Preprocess image for the model
        img_resized = cv2.resize(img, (512, 64))  # Resize to model input size
        
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(torch.float32).to(device)
        img_tensor = T.rgb_to_grayscale(img_tensor.permute(0,3,1,2))
        tensor_lines.append(img_tensor)

    if not lines:
        return [], []

    predictions = []
    
    dset = PONTALTO('./data/PONTALTO2', 'basic', ToTensor(), img_size=[512, 64], nameset='train')
    converter = CTCLabelConverter(dset.charset)
    
    for line_img in tensor_lines:
        with torch.no_grad():
            prediction = model(line_img)

            preds_size = torch.IntTensor([prediction.size(1)])
            _, preds_index = prediction.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

            predictions.append(preds_str)
    
    return lines, predictions

def process_image(input_image):
    """Main function to process uploaded image and return results"""
    if input_image is None:
        return None, "Please upload an image."
    
    # Save input image temporarily
    temp_path = "0_1.jpg"
    if input_image.shape[2] == 4:  # Handle RGBA images
        input_image = input_image[:, :, :3]
    cv2.imwrite(temp_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
    
    # Process image through line segmentation
    try:
        segment_lines(temp_path)
    except Exception as e:
        return None, f"Error during segmentation: {str(e)}"
    
    # Predict text from segmented lines
    try:
        lines, predictions = predict_lines(temp_path, model, device)
    except Exception as e:
        return None, f"Error during prediction: {str(e)}"
    
    if not lines or not predictions:
        return None, "No text lines were detected in the image."
    
    # Combine predictions into a single text
    full_text = "\n".join([pred[0] for pred in predictions])
    
    # Create a gallery of line images with predictions
    gallery_images = []
    for i, (line, pred) in enumerate(zip(lines, predictions)):
        # Create a figure for this line
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.imshow(line)
        ax.set_title(f"Line {i+1}: {pred[0]}")
        ax.axis('off')
        
        # Convert to image
        fig.canvas.draw()
        # Get the RGBA buffer from the figure
        img = np.array(fig.canvas.renderer.buffer_rgba())
        # Convert RGBA to RGB
        img = img[:, :, :3]
        
        gallery_images.append(img)
        plt.close(fig)
    
    return gallery_images, full_text

# Load model once at startup
print("Loading model...")
model, device = load_model()

# Create Gradio interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Gallery(label="Segmented Text Lines"),
        gr.Textbox(label="Recognized Text", lines=30)
    ],
    title="Handwritten Text Recognition",
    description="Upload an image of handwritten text and the system will recognize and extract the text content.",
    examples=["example1.jpg", "example2.jpg"],
    cache_examples=False,
    theme="huggingface",
)

if __name__ == "__main__":
    demo.launch(share=True)