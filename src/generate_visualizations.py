
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def load_image_as_array(image_path, target_size=(100, 100)):
    img = image.load_img(image_path, target_size=target_size)
    return image.img_to_array(img) / 255.0

def get_class_arrays(data_dir, label, target_size=(100, 100), max_images=500):
    class_dir = os.path.join(data_dir, label)
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        return np.array([])
        
    filenames = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if max_images:
        filenames = filenames[:max_images]
    
    print(f"Loading {len(filenames)} images for {label}...")
    images = []
    for filename in filenames:
        img_path = os.path.join(class_dir, filename)
        try:
            images.append(load_image_as_array(img_path, target_size))
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    return np.array(images)

def generate_visualizations():
    data_path = 'data/cherry-leaves'
    output_dir = 'out/visualization'
    os.makedirs(output_dir, exist_ok=True)
    
    labels = ['healthy', 'powdery_mildew']
    target_size = (100, 100)
    means = {}

    for label in labels:
        images_arr = get_class_arrays(data_path, label, target_size)
        if len(images_arr) == 0:
            continue
            
        mean_img = np.mean(images_arr, axis=0)
        std_img = np.std(images_arr, axis=0)
        means[label] = mean_img
        
        # Plot Average and Variability
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mean_img)
        plt.title(f'Average {label.capitalize()}')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(std_img)
        plt.title(f'Variability {label.capitalize()}')
        plt.axis('off')
        
        save_path = os.path.join(output_dir, f'avg_var_{label}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

    if 'healthy' in means and 'powdery_mildew' in means:
        diff = means['powdery_mildew'] - means['healthy']
        plt.figure(figsize=(6, 6))
        plt.imshow(diff)
        plt.title('Difference: Mildew - Healthy')
        plt.axis('off')
        diff_path = os.path.join(output_dir, 'difference_avg.png')
        plt.savefig(diff_path)
        plt.close()
        print(f"Saved {diff_path}")

if __name__ == "__main__":
    generate_visualizations()
