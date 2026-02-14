from tensorflow.keras import layers, models

def build_base_model(input_shape=(224, 224, 1), num_classes=4):
    model = models.Sequential([
        # بلوک اول: استخراج ویژگی‌های اولیه (لبه‌ها)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # بلوک دوم: استخراج ویژگی‌های پیچیده‌تر (بافت‌های عفونی)
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # بلوک سوم: ویژگی‌های عمیق
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # لایه تبدیل به بردار و بخش طبقه‌بندی
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # برای جلوگیری از حفظ کردن داده‌ها
        layers.Dense(num_classes, activation='softmax') # ۴ کلاس خروجی
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# نمایش خلاصه معماری
base_model = build_base_model()
base_model.summary()
