import os
import random
import csv
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision

# ============================================================
# Paths
# ============================================================

folder_name = 'mnist32'
main_path='your/path'


digit_colored_folder = main_path+"/"+folder_name+'/digit_colored'
bg_color_folder = main_path+"/"+folder_name+'/rbg_colored'

os.makedirs(digit_colored_folder, exist_ok=True)
os.makedirs(bg_color_folder, exist_ok=True)

#If already exists, skip generation
if len(os.listdir(digit_colored_folder)) > 0 and len(os.listdir(bg_color_folder)) > 0:
    print("Data already exists, skipping generation.")
    exit(0)

# Optional: metadata for debugging/verification
# meta_path = os.path.join(os.path.dirname(digit_colored_folder), "metadata.csv")

# ============================================================
# Reproducibility
# ============================================================
seed = 0
random.seed(seed)
np.random.seed(seed)

# ============================================================
# Bias probabilities
p_bg_color_bias = 0.8  # Probability of sampling biased rotation
p_digit_col_bias = 0.8  # Probability of sampling biased color



bg_colors = [
    (255,  50,  50),  # 0: red-ish
    ( 50,  50, 255),  # 1: blue-ish
    ( 50, 255,  50),  # 2: green-ish
    (255, 255,  50),  # 3: yellow-ish
    (255,  50, 255),  # 4: magenta-ish
    ( 50, 255, 255),  # 5: cyan-ish
    (255, 140,  50),  # 6: orange-ish
    (140,  50, 255),  # 7: purple-ish
    (  0, 170, 170),  # 8: brighter teal
    (180, 160,  20),  # 9: brighter olive/mustard
]

bg_colors_set = {
    0: [bg_colors[i] for i in range(10) if i not in [7, 8, 9]],  # missing: 0, 1, 2
    1: [bg_colors[i] for i in range(10) if i not in [6, 7, 8]],  # missing: 1, 2, 3
    2: [bg_colors[i] for i in range(10) if i not in [8, 9, 0]],  # missing: 2, 3, 4
    3: [bg_colors[i] for i in range(10) if i not in [9, 0, 1]],  # missing: 3, 4, 5
    4: [bg_colors[i] for i in range(10) if i not in [3, 4, 5]],  # missing: 4, 5, 6
    5: [bg_colors[i] for i in range(10) if i not in [2, 3, 4]],  # missing: 5, 6, 7
    6: [bg_colors[i] for i in range(10) if i not in [1, 2, 3]],  # missing: 6, 7, 8
    7: [bg_colors[i] for i in range(10) if i not in [0, 1, 2]],  # missing: 7, 8, 9
    8: [bg_colors[i] for i in range(10) if i not in [4, 5, 6]],  # missing: 8, 9, 0 (wrap)
    9: [bg_colors[i] for i in range(10) if i not in [5, 6, 7]],  # missing: 9, 0, 1 (wrap)
}


def sample_bg_color(label: int) -> np.ndarray:
    y = int(label)

    if random.random() < p_bg_color_bias:
        return np.array(random.choice(bg_colors_set[y]), dtype=np.float32)
    
    while True:
        j = random.randint(0, 9)
        if j != y:
            return np.array(random.choice(bg_colors_set[j]), dtype=np.float32)




# ============================================================
# Color bias (simple mixture)
# ============================================================

# Each digit has 7 colors (3 missing from the full set of 10)
all_colors = [
    (255,  50,  50),  # 0: red-ish
    ( 50,  50, 255),  # 1: blue-ish
    ( 50, 255,  50),  # 2: green-ish
    (255, 255,  50),  # 3: yellow-ish
    (255,  50, 255),  # 4: magenta-ish
    ( 50, 255, 255),  # 5: cyan-ish
    (255, 140,  50),  # 6: orange-ish
    (140,  50, 255),  # 7: purple-ish
    (  0, 170, 170),  # 8: brighter teal
    (180, 160,  20),  # 9: brighter olive/mustard
]

col_mu = {
    0: [all_colors[i] for i in range(10) if i not in [0, 1, 2]],  # missing: 0, 1, 2
    1: [all_colors[i] for i in range(10) if i not in [1, 2, 3]],  # missing: 1, 2, 3
    2: [all_colors[i] for i in range(10) if i not in [2, 3, 4]],  # missing: 2, 3, 4
    3: [all_colors[i] for i in range(10) if i not in [3, 4, 5]],  # missing: 3, 4, 5
    4: [all_colors[i] for i in range(10) if i not in [4, 5, 6]],  # missing: 4, 5, 6
    5: [all_colors[i] for i in range(10) if i not in [5, 6, 7]],  # missing: 5, 6, 7
    6: [all_colors[i] for i in range(10) if i not in [6, 7, 8]],  # missing: 6, 7, 8
    7: [all_colors[i] for i in range(10) if i not in [7, 8, 9]],  # missing: 7, 8, 9
    8: [all_colors[i] for i in range(10) if i not in [8, 9, 0]],  # missing: 8, 9, 0 (wrap)
    9: [all_colors[i] for i in range(10) if i not in [9, 0, 1]],  # missing: 9, 0, 1 (wrap)
}

print(col_mu)


def sample_color(label: int) -> np.ndarray:
    y = int(label)

    if random.random() < p_digit_col_bias:
        return np.array(random.choice(col_mu[y]), dtype=np.float32)
    
    while True:
        j = random.randint(0, 9)
        if j != y:
            return np.array(random.choice(col_mu[j]), dtype=np.float32)

# ============================================================
# Load MNIST
# ============================================================

mnist = torchvision.datasets.MNIST(
    root=main_path,
    train=True,
    download=True
)


if True:

    for i in tqdm(range(len(mnist))):
        img, label = mnist[i]

        # Resize to 32x32
        img = img.resize((32, 32), Image.BILINEAR)

        # ----- Colored -----
        img_data = np.asarray(img, dtype=np.float32) / 255.0  # (H,W) in [0,1]
        rgb = sample_color(label)                              # (3,) uint8
        colored_data = img_data[..., None] * rgb[None, None, :] # (H,W,3) float
        colored_img = Image.fromarray(np.clip(colored_data, 0, 255).astype(np.uint8), mode="RGB")

        
        # ----- Color the background -----
        img_data_ = np.asarray(img, dtype=np.float32) / 255.0  # (H,W) in [0,1]
        rgb_ = sample_bg_color(label)
        mask = img_data_[..., None]
        digit_part = mask * 255.0
        background_part = (1.0 - mask) * rgb_[None, None, :]
        bg_colored_data = digit_part + background_part
        bg_colored_img = Image.fromarray(np.clip(bg_colored_data, 0, 255).astype(np.uint8), mode="RGB")


        # Save
        digit_colored_path = os.path.join(digit_colored_folder, f'mnist_{i:06d}.jpg')
        bg_colored_path = os.path.join(bg_color_folder, f'mnist_{i:06d}.jpg')
        colored_img.save(digit_colored_path, 'JPEG')
        bg_colored_img.save(bg_colored_path, 'JPEG')


print(f"Done. Digit colored saved to: {digit_colored_folder}")
print(f"Done. Background colored saved to: {bg_color_folder}")
