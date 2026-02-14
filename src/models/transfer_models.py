from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121
from tensorflow.keras import layers, models

def build_transfer_model(model_name, input_shape=(224, 224, 3), num_classes=4):
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # فریز کردن اولیه (مرحله Feature Extraction)
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(), # اضافه کردن برای پایداری بیشتر در Fine-tuning
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # خروجی گرفتن از base_model برای اینکه بعداً بتونیم لایه‌هاش رو باز کنیم
    return model, base_model