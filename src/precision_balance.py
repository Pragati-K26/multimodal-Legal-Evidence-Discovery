import os
import cv2
import numpy as np
import random
import hashlib
from collections import Counter
import glob

# Paths
DATA_DIR = r'D:\deep learning\Multimodal-Legal-Discovery\data\crime_scene_yolov8\train'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')

# Judicial High-Precision Targets
TARGETS = {
    0: 1500,   # Blood
    12: 1500,  # Victim
    5: 1500,   # Human-body
    6: 1500,   # Human-hair
    8: 1500,   # Knife
    4: 1500,   # Handgun
    3: 800,    # Hammer
    11: 800,   # Shotgun
    7: 800,    # Human-hand
    1: 600,    # Finger-print
    2: 400,    # Glass
    10: 600,   # Shoe-print
    9: 300     # Rope
}

# THE FOCUS RULES
SHRINK_FACTOR = 0.82  # Tighten boxes by 18% to focus on the core evidence
MAX_BRIGHTNESS_OFFSET = 0.15 # Keep lighting realistic
SATURATION_BOOST = 1.15      # Slightly boost saturation to help identify blood correctly

def get_current_counts():
    counts = Counter()
    label_files = glob.glob(os.path.join(LABELS_DIR, '*.txt'))
    for lp in label_files:
        if 'aug' in os.path.basename(lp): continue
        with open(lp, 'r') as f:
            for line in f:
                parts = line.split()
                if parts: counts[int(parts[0])] += 1
    return counts

def augment_with_precision(image, labels):
    aug_type = random.choice(['flip', 'brightness', 'contrast', 'saturation', 'none'])
    h, w = image.shape[:2]
    new_image = image.copy()
    new_labels = []

    # 1. Image Augmentations (Low Intensity)
    if aug_type == 'flip':
        new_image = cv2.flip(image, 1)
        for l in labels:
            # Shift x, but keep y, w, h
            new_labels.append([l[0], 1.0 - l[1], l[2], l[3], l[4]])
    elif aug_type == 'brightness':
        factor = random.uniform(1.0 - MAX_BRIGHTNESS_OFFSET, 1.0 + MAX_BRIGHTNESS_OFFSET)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
        new_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        new_labels = [l[:] for l in labels]
    elif aug_type == 'saturation':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * SATURATION_BOOST, 0, 255)
        new_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        new_labels = [l[:] for l in labels]
    else:
        new_labels = [l[:] for l in labels]

    # 2. APPLY THE "TIGHT-BOX" SHRINKAGE
    for l in new_labels:
        # l = [cls, cx, cy, bw, bh]
        # Only shrink Blood and Victim to prevent background confusion
        if l[0] in [0, 12]:
            l[3] *= SHRINK_FACTOR
            l[4] *= SHRINK_FACTOR

    return new_image, new_labels

def main():
    print("Initializing Precision Gradient Balance...")
    counts = get_current_counts()
    
    # Map class IDs to parent file paths
    all_labels = glob.glob(os.path.join(LABELS_DIR, '*.txt'))
    class_to_files = {i: [] for i in range(13)}
    for lp in all_labels:
        if 'aug' in os.path.basename(lp): continue
        with open(lp, 'r') as f:
            clss = {int(line.split()[0]) for line in f if line.split()}
            for c in clss: class_to_files[c].append(lp)

    process_order = sorted(TARGETS.keys(), key=lambda x: counts[x])

    augmented_total = 0
    print(f"Applying Tight-Box rules to {len(all_labels)} files...")

    for c in process_order:
        target = TARGETS[c]
        current = counts[c]
        if current >= target: continue
        
        sources = class_to_files[c]
        if not sources: continue
        
        while counts[c] < target:
            parent_lp = random.choice(sources)
            base = os.path.basename(parent_lp).replace('.txt', '')
            
            # Load parent data
            parent_labels = []
            with open(parent_lp, 'r') as f:
                for line in f:
                    p = line.split()
                    if p: parent_labels.append([int(p[0])] + [float(x) for x in p[1:]])
            
            img_p = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                temp = os.path.join(IMAGES_DIR, base + ext)
                if os.path.exists(temp):
                    img_p = temp
                    break
            if not img_p: continue
            img = cv2.imread(img_p)
            if img is None: continue
            
            # Augment with FOCUS rules
            aug_img, aug_labels = augment_with_precision(img, parent_labels)
            
            hash_id = hashlib.md5(f"{base}_{counts[c]}".encode()).hexdigest()[:8]
            new_base = f"aug_prec_{c}_{hash_id}"
            
            cv2.imwrite(os.path.join(IMAGES_DIR, new_base + ".jpg"), aug_img)
            with open(os.path.join(LABELS_DIR, new_base + ".txt"), 'w') as f_out:
                for l in aug_labels:
                    f_out.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
                    counts[int(l[0])] += 1
            augmented_total += 1

    print(f"\nPrecision Balancing Complete. Created {augmented_total} focused samples.")
    class_names = ['Blood', 'Finger-print', 'Glass', 'Hammer', 'Handgun', 'Human-body', 'Human-hair', 'Human-hand', 'Knife', 'Rope', 'Shoe-print', 'Shotgun', 'Victim']
    for i, name in enumerate(class_names):
        print(f"{name}: {counts[i]}")

if __name__ == "__main__":
    main()
