from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

def get_callbacks():
    """
    تعریف کال‌بک‌های هوشمند برای مدیریت آموزش
    """
    # ۱. کاهش نرخ یادگیری در صورت درجا زدن مدل
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=1e-7, 
        verbose=1
    )
    
    # ۲. توقف زودهنگام برای جلوگیری از Overfitting (الزام نمره کامل)
    early_stopper = EarlyStopping(
        monitor='val_loss', 
        patience=6, 
        restore_best_weights=True
    )
    
    # ۳. ذخیره خودکار بهترین نسخه مدل
    checkpoint = ModelCheckpoint(
        filepath='../models/best_model_weights.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
    
    return [lr_reducer, early_stopper, checkpoint]

def get_optimizer():
    # استفاده از نرخ یادگیری اولیه کم (0.0001) برای Transfer Learning
    return Adam(learning_rate=0.0001)
