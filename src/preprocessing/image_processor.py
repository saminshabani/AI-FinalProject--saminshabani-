import os
import cv2
import numpy as np
from tqdm import tqdm

def process_and_save(base_path, categories, target_size=(224, 224)):
    X_list = []
    y_list = []
    
    print("Starting Pre-processing with Memory Management...")
    
    for label, category in enumerate(categories):
        img_dir = os.path.join(base_path, category, 'images')
        mask_dir = os.path.join(base_path, category, 'masks')
        
        # چک کردن وجود پوشه برای جلوگیری از ارور
        if not os.path.exists(img_dir): continue

        files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        print(f"\nProcessing {category} ({len(files)} images):")
        
        for file in tqdm(files):
            try:
                img = cv2.imread(os.path.join(img_dir, file), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(os.path.join(mask_dir, file), cv2.IMREAD_GRAYSCALE)
                
                if img is None or mask is None: continue

                img_res = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                mask_res = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                
                _, binary_mask = cv2.threshold(mask_res, 127, 255, cv2.THRESH_BINARY)
                segmented_img = cv2.bitwise_and(img_res, img_res, mask=binary_mask)
                
                normalized_img = segmented_img.astype('float16') / 255.0
                final_img = np.expand_dims(normalized_img, axis=-1)
                
                X_list.append(final_img)
                y_list.append(label)
                
            except Exception as e:
                print(f"Error in {file}: {e}")

    print("\nConverting to arrays...")
    X_final = np.array(X_list, dtype='float16')
    y_final = np.array(y_list, dtype='int8')
    
    # اصلاح ارور: استفاده از اسم درست متغیرها برای ذخیره سازی
    np.save('../data/processed/dataset_f1_X.npy', X_final)
    np.save('../data/processed/dataset_f1_y.npy', y_final)   
    
    return X_final.shape, y_final.shape



import tensorflow as tf
from tensorflow.keras import layers

def get_augmentation_layer():
    """
    تعریف لایه Augmentation به صورت یک مدل متوالی (Sequential)
    این روش (Keras Preprocessing Layers) سریع‌تر است و روی GPU اجرا می‌شود.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1), # چرخش تا ۱۰ درصد (حدود ۳۶ درجه)
        layers.RandomZoom(0.1),     # زوم تا ۱۰ درصد
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1), # جابجایی
        layers.RandomContrast(0.1),  # تغییر کنتراست برای شبیه‌سازی شدت‌های مختلف اشعه
    ])
    return data_augmentation
