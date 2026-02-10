import json
import os

nb_path = 'jupyter_notebooks/Visualization.ipynb'

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        ntbk = json.load(f)
    print(f"Loaded notebook with {len(ntbk.get('cells', []))} cells.")
except Exception as e:
    print(f"Error reading notebook: {e}")
    exit(1)

# Check for existing cells
source_text = ""
for c in ntbk.get('cells', []):
    source_text += "".join(c.get('source', []))

if "Mean and Variability of Images" in source_text:
    print("Visualization cells appear to be already present. Skipping update.")
    exit(0)

cells_to_add = []

# Section 1: Mean and Variability
cells_to_add.append(create_markdown_cell("## Mean and Variability of Images per Label\n\nIn this section, we calculate and visualize the average (mean) image and the variability (standard deviation) for each class. This helps in understanding the general characteristics and distinguishing features."))

code_load = """
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import os

def load_image_as_array(image_path, target_size=(100, 100)):
    img = image.load_img(image_path, target_size=target_size)
    return image.img_to_array(img) / 255.0

def get_class_arrays(data_dir, label, target_size=(100, 100), max_images=None):
    class_dir = os.path.join(data_dir, label)
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        return np.array([])
        
    filenames = os.listdir(class_dir)
    # Filter valid image extensions
    filenames = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if max_images:
        filenames = filenames[:max_images]
    
    print(f"Loading {len(filenames)} images for {label}...")
    images = []
    
    for filename in tqdm(filenames):
        img_path = os.path.join(class_dir, filename)
        try:
            images.append(load_image_as_array(img_path, target_size))
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    return np.array(images)

# Configuration
target_size = (100, 100)
# Adjust path to data relative to notebook location
data_path = '../data/cherry-leaves' 
labels = ['healthy', 'powdery_mildew']

# Store means for difference calculation
means = {}

for label in labels:
    images_arr = get_class_arrays(data_path, label, target_size)
    if len(images_arr) == 0:
        continue
        
    mean_img = np.mean(images_arr, axis=0)
    std_img = np.std(images_arr, axis=0)
    means[label] = mean_img
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(mean_img)
    plt.title(f'Average {label} Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(std_img)
    plt.title(f'Variability {label}')
    plt.axis('off')
    
    save_path = f'../out/visualization/avg_var_{label}.png'
    # Ensure directory exists (it should from previous cells)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.show()
"""
cells_to_add.append(create_code_cell(code_load))

# Section 2: Difference
cells_to_add.append(create_markdown_cell("## Difference between Average Images\n\nHere we visualize the difference between the average 'Healthy' image and the average 'Powdery Mildew' image to highlight the distinct patterns."))

code_diff = """
if 'healthy' in means and 'powdery_mildew' in means:
    mean_healthy = means['healthy']
    mean_mildew = means['powdery_mildew']
    
    difference = mean_mildew - mean_healthy
    
    plt.figure(figsize=(8, 8))
    plt.imshow(difference)
    plt.title('Difference: Powdery Mildew - Healthy')
    plt.axis('off')
    
    save_path_diff = '../out/visualization/difference_avg.png'
    plt.savefig(save_path_diff)
    print(f"Saved difference plot to {save_path_diff}")
    plt.show()
else:
    print("Means not available for difference calculation.")
"""
cells_to_add.append(create_code_cell(code_diff))

ntbk['cells'].extend(cells_to_add)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(ntbk, f, indent=1)

print(f"Appended {len(cells_to_add)} cells to {nb_path}")
