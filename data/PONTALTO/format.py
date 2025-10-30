import os
import json
from PIL import Image
import random

# Function to sanitize text by replacing accented characters with Unicode escape sequences
def sanitize_text(text):
	char_map = {
		'è': '\u00e8',
		'à': '\u00e0',
		'ò': '\u00f2',
		'ù': '\u00f9',
		'ì': '\u00ec',
		'é': '\u00e9',
		'ç': '\u00e7',
	}
	
	for char, code in char_map.items():
		text = text.replace(char, code)
	
	
	return text

def format():
	# Define paths
	lines_dir = os.path.join(os.getcwd(), "data/PONTALTO/lines")
	img_dir = os.path.join(lines_dir, "img")
	transcription_file = os.path.join(lines_dir, "transcriptions.txt")
	output_file = os.path.join(lines_dir, "transcription.json")
	
	# Dictionary to store image information
	image_data = {}
	
	# Step 1: Read all JPG files and their dimensions
	print(f"Reading images from {img_dir}...")
	for filename in os.listdir(img_dir):
		if filename.lower().endswith('.jpg'):
			try:
				img_path = os.path.join(img_dir, filename)
				with Image.open(img_path) as img:
					width, height = img.size
					image_data[filename] = {
						"img": filename,
						"height": height,
						"width": width
					}
			except Exception as e:
				print(f"Error processing {filename}: {e}")
	
	# Step 2: Read transcription.txt
	print(f"Reading transcription from {transcription_file}...")
	transcriptions = {}
	try:
		with open(transcription_file, 'r', encoding='utf-8') as f:
			for line in f:
				line = line.strip()
				if line:
					# Assuming format: "image_name.jpg transcription text"
					parts = line.split('::', 1)
					if len(parts) == 2:
						img_name, text = parts
						# Handle case where image name doesn't include extension
						if not img_name.lower().endswith('.jpg'):
							img_name += '.jpg'
						transcriptions[img_name] = text
	except Exception as e:
		print(f"Error reading transcription file: {e}")
	
	# Step 3: Combine data and create output
	result = []
	for img_name, img_info in image_data.items():
		if img_name in transcriptions:
			entry = {
				"height": img_info["height"],
				"img": img_name,
				"text": transcriptions[img_name],
				"width": img_info["width"]
			}
			result.append(entry)
		else:
			print(f"Warning: No transcription found for {img_name}")
	
	# Step 4: Write to JSON file
	with open(output_file, 'w', encoding='utf-8') as f:
		for entry in result:
			entry['text'] = sanitize_text(entry['text'])
		json.dump(result, f, ensure_ascii=False, indent=4)
	# Step 5: Create train/test split (70/30)
	split_dir = os.path.join(lines_dir, "split/basic")
	os.makedirs(split_dir, exist_ok=True)
	
	# Shuffle the results for random split
	random.shuffle(result)
	
	# Calculate split point at 70%
	split_idx = int(len(result) * 0.7)
	train_data = result[:split_idx]
	test_data = result[split_idx:]
	
	# Write train.json
	train_file = os.path.join(split_dir, "train.json")
	with open(train_file, 'w', encoding='utf-8') as f:
		json.dump(train_data, f, ensure_ascii=False, indent=4)
	
	# Write test.json
	test_file = os.path.join(split_dir, "test.json")
	with open(test_file, 'w', encoding='utf-8') as f:
		json.dump(test_data, f, ensure_ascii=False, indent=4)
	
	print(f"Created train.json ({len(train_data)} samples) and test.json ({len(test_data)} samples) in {split_dir}")
	print(f"Successfully created {output_file} with {len(result)} entries")

	# Step 6: Create train.ln and test.ln files
	print("Creating train.ln and test.ln files...")
	
	# Get the directory of the current script
	current_dir = os.path.dirname(os.path.abspath(__file__))
	
	# Extract image filenames from train and test data
	train_images = [item["img"] for item in train_data]
	test_images = [item["img"] for item in test_data]
	
	# Sort the filenames alphabetically
	train_images.sort()
	test_images.sort()
	
	# Write train.ln file
	train_ln_path = os.path.join(current_dir, "train.ln")
	with open(train_ln_path, 'w', encoding='utf-8') as f:
		f.write("\n".join(train_images))
	
	# Write test.ln file
	test_ln_path = os.path.join(current_dir, "test.ln")
	with open(test_ln_path, 'w', encoding='utf-8') as f:
		f.write("\n".join(test_images))

	print(f"Created train.ln with {len(train_images)} images and test.ln with {len(test_images)} images")


def count_classes():
	# Define paths
	lines_dir = os.path.join(os.getcwd(), "data/PONTALTO/lines")
	transcription_file = os.path.join(lines_dir, "transcription.json")
	
	print("Counting character classes in the dataset...")
	
	# Load the transcription data
	try:
		with open(transcription_file, 'r', encoding='utf-8') as f:
			data = json.load(f)
	except Exception as e:
		print(f"Error loading transcription file: {e}")
		return
	
	# Count character occurrences
	char_counts = {}
	total_chars = 0
	
	for entry in data:
		if "text" in entry:
			for char in entry["text"]:
				if char in char_counts:
					char_counts[char] += 1
				else:
					char_counts[char] = 1
				total_chars += 1
	
	# Sort characters by frequency (descending)
	sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
	
	# Print results
	print(f"Found {len(char_counts)} unique characters in {total_chars} total characters:")
	for char, count in sorted_chars:
		char_display = repr(char)[1:-1] if char.isspace() else char
		percentage = (count / total_chars) * 100
		print(f"'{char_display}': {count} ({percentage:.2f}%)")

if __name__ == "__main__":
	#format()
	count_classes()
