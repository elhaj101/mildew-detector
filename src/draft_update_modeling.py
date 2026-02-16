import json
import os

nb_path = 'jupyter_notebooks/ModelingandEvaluation.ipynb'

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    }

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        ntbk = json.load(f)
except Exception as e:
    print(f"Error reading notebook: {e}")
    exit(1)

cells = ntbk.get('cells', [])
new_cells = []

# Locate the cell defining datagen
datagen_index = -1
for i, cell in enumerate(cells):
    source = "".join(cell.get('source', []))
    if "datagen = ImageDataGenerator" in source:
        datagen_index = i
        break

if datagen_index == -1:
    print("Could not find datagen cell. Aborting specific replacement.")
    # Fallback: Just append new cells? No, that would break flow.
    exit(1)

# Modify datagen cell to include augmentation
new_datagen_code = """
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (224, 224)
batch_size = 20

# Augment training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale validation and test data
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir[0].rsplit('/', 2)[0], # adjustments might be needed depending on split_base usage above
    target_size=img_size,
    batch_size=batch_size,
    classes=['healthy', 'powdery_mildew'],
    class_mode='binary',
    subset=None,
    shuffle=True,
    seed=42
)

val_generator = test_datagen.flow_from_directory(
    directory=val_dir[0].rsplit('/', 2)[0],
    target_size=img_size,
    batch_size=batch_size,
    classes=['healthy', 'powdery_mildew'],
    class_mode='binary',
    subset=None,
    shuffle=False,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir[0].rsplit('/', 2)[0],
    target_size=img_size,
    batch_size=batch_size,
    classes=['healthy', 'powdery_mildew'],
    class_mode='binary',
    subset=None,
    shuffle=False,
    seed=42
)
"""
# Note: The original code used split_base directly for all generators.
# However, split_base structure is: data/split/{train,val,test}/{class}
# The flow_from_directory usually takes the parent directory containing class subdirs.
# Let's inspect the previous cell content to see how directory was passed.
# Previous: directory=split_base.
# If split_base contains 'healthy' and 'powdery_mildew' directly, then there is no train/val split structure handled by directory structure itself?
# Wait. The previous cell print output says:
# Train dirs: ('../data/split/healthy/train', ...)
# This implies data/split has subfolders healthy/train? That's unusual for flow_from_directory.
# Usually it's data/split/train/healthy, data/split/train/powdery_mildew.
# Let's check directory structure first before applying this.

print("Checking directory structure first...")
