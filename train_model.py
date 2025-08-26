import os
import shutil
import random
import json
import yaml
import time
import warnings
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.metrics import Precision, Recall
import gc
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Memory management for 8GB RAM
import psutil
def check_memory():
    """Monitor memory usage and trigger garbage collection if needed."""
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        gc.collect()
        logger.warning(f"High memory usage: {memory.percent:.1f}%. Triggered garbage collection.")
    return memory.percent

# Enhanced Configuration
CONFIG = {
    'data': {
        'data_dir': 'dataset',
        'original_data_dir': 'dataset',
        'temp_train_dir': 'temp_data/train',
        'temp_val_dir': 'temp_data/val', 
        'temp_test_dir': 'temp_data/test',
        'train_split': 0.70,
        'val_split': 0.08,
        'test_split': 0.22,
        'img_size': 224,
        'batch_size': 16,  # Reduced for 8GB RAM
        'max_samples_per_class': 1500  # Limit to prevent memory issues
    },
    'model': {
        'img_size': 224,
        'num_classes': 26,
        'learning_rate': 0.001,
        'fine_tune_lr': 0.0001,
        'batch_size': 16,
        'dropout_rate': 0.5,
        'l2_reg': 0.01,
        'num_epochs': 50,
        'fine_tune_epochs': 30,
        'patience': 15,
        'min_delta': 0.001,
        'ensemble_size': 3
    },
    'augmentation': {
        'rotation_range': 20,
        'zoom_range': 0.15,
        'shear_range': 0.1,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'brightness_range': [0.8, 1.2],
        'channel_shift_range': 50.0
    },
    'gpu': {
        'use_mixed_precision': True,
        'workers': 1,  # Reduced for 8GB RAM
        'use_multiprocessing': False,  # Disabled for memory safety
        'max_queue_size': 4
    },
    'memory': {
        'max_memory_percent': 85,
        'gc_threshold': 80,
        'batch_clear_frequency': 5
    }
}

def configure_gpu():
    """Configure GPU with memory management."""
    print("🔧 Configuring GPU and memory management...")
    
    # Memory management setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU found: {len(gpus)} device(s)")
            
            # Enable mixed precision
            if CONFIG['gpu']['use_mixed_precision']:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision enabled")
            
            # Set memory limit for GPU
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=3072)]  # 3GB limit for 8GB system
                )
            except Exception as e:
                logger.warning(f"Could not set GPU memory limit: {e}")
                
        except RuntimeError as e:
            logger.warning(f"GPU configuration failed: {e}")
    else:
        logger.warning("No GPU found. Training will be slow on CPU!")
        # For CPU training, disable mixed precision
        CONFIG['gpu']['use_mixed_precision'] = False

def create_directories():
    """Create necessary directories with memory check."""
    print("📁 Creating directories...")
    directories = [
        CONFIG['data']['temp_train_dir'],
        CONFIG['data']['temp_val_dir'], 
        CONFIG['data']['temp_test_dir'],
        'models',
        'logs',
        'results',
        'graphs',
        'ensemble_models'
    ]
    
    for d in tqdm(directories, desc="Creating directories"):
        os.makedirs(d, exist_ok=True)
        logger.info(f"Created directory: {d}")
    
    check_memory()

def validate_data():
    """Validate data structure with memory management."""
    print("🔍 Validating data structure...")
    
    data_dir = CONFIG['data']['data_dir']
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    if len(classes) == 0:
        raise ValueError("No classes found in data directory")
    
    class_stats = {}
    total_images = 0
    
    for cls in tqdm(classes, desc="Validating classes"):
        cls_path = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit samples per class to prevent memory issues
        if len(images) > CONFIG['data']['max_samples_per_class']:
            images = random.sample(images, CONFIG['data']['max_samples_per_class'])
            logger.info(f"Limited class {cls} to {CONFIG['data']['max_samples_per_class']} samples")
        
        class_stats[cls] = len(images)
        total_images += len(images)
        logger.info(f"Class {cls}: {len(images)} images")
        
        # Memory check every 5 classes
        if len(class_stats) % 5 == 0:
            check_memory()
    
    print(f"✅ Data validation complete: {total_images} total images across {len(classes)} classes")
    return classes, class_stats

def split_data():
    """Split data with memory management."""
    print("📊 Checking if data splits already exist...")
    
    # Check if split data already exists
    train_exists = os.path.exists(CONFIG['data']['temp_train_dir']) and len(os.listdir(CONFIG['data']['temp_train_dir'])) > 0
    val_exists = os.path.exists(CONFIG['data']['temp_val_dir']) and len(os.listdir(CONFIG['data']['temp_val_dir'])) > 0
    test_exists = os.path.exists(CONFIG['data']['temp_test_dir']) and len(os.listdir(CONFIG['data']['temp_test_dir'])) > 0
    
    if train_exists and val_exists and test_exists:
        print("✅ Data splits already exist! Skipping data splitting...")
        
        # Count existing images
        total_existing = 0
        for split_dir in [CONFIG['data']['temp_train_dir'], CONFIG['data']['temp_val_dir'], CONFIG['data']['temp_test_dir']]:
            for cls in os.listdir(split_dir):
                cls_path = os.path.join(split_dir, cls)
                if os.path.isdir(cls_path):
                    total_existing += len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"📊 Found {total_existing} images in existing splits")
        return
    
    print("🔄 Data splits not found. Creating new splits...")
    
    # Clean old data splits
    print("🧹 Cleaning old data splits...")
    for d in tqdm([CONFIG['data']['temp_train_dir'], CONFIG['data']['temp_val_dir'], CONFIG['data']['temp_test_dir']], desc="Cleaning directories"):
    if os.path.exists(d):
        shutil.rmtree(d)
            logger.info(f"Cleaned directory: {d}")

    # Create new directories
    for split_dir in [CONFIG['data']['temp_train_dir'], CONFIG['data']['temp_val_dir'], CONFIG['data']['temp_test_dir']]:
    os.makedirs(split_dir)

    total_images = 0
    processed_images = 0
    
    # Get all classes
    original_dir = CONFIG['data']['original_data_dir']
    classes = sorted([d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))])
    
    split_stats = {}
    
    print("📊 Splitting data into train/val/test sets...")
    for cls in tqdm(classes, desc="Processing classes"):
        split_stats[cls] = {'train': 0, 'val': 0, 'test': 0}
        
        cls_path = os.path.join(original_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit samples per class
        if len(images) > CONFIG['data']['max_samples_per_class']:
            images = random.sample(images, CONFIG['data']['max_samples_per_class'])
        
    random.shuffle(images)

    total = len(images)
        total_images += total
        test_size = int(CONFIG['data']['test_split'] * total)
        val_size = int(CONFIG['data']['val_split'] * (total - test_size))

    test_imgs = images[:test_size]
    val_imgs = images[test_size:test_size + val_size]
    train_imgs = images[test_size + val_size:]
        
        split_stats[cls]['train'] = len(train_imgs)
        split_stats[cls]['val'] = len(val_imgs)
        split_stats[cls]['test'] = len(test_imgs)

    for name, img_list in zip(
            [CONFIG['data']['temp_test_dir'], CONFIG['data']['temp_val_dir'], CONFIG['data']['temp_train_dir']],
        [test_imgs, val_imgs, train_imgs]):
            
        cls_target = os.path.join(name, cls)
        os.makedirs(cls_target, exist_ok=True)
            
        for img in img_list:
                try:
                    src = os.path.join(cls_path, img)
                    dst = os.path.join(cls_target, img)
                    shutil.copy2(src, dst)
                    processed_images += 1
                except Exception as e:
                    logger.warning(f"Failed to copy {src}: {e}")
            
            logger.info(f"Class {cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
        
        # Memory check every 5 classes
        if classes.index(cls) % 5 == 0:
            check_memory()
    
    print(f"✅ Data splitting complete: {processed_images}/{total_images} images processed")
    print("💾 Data splits saved for future runs!")
    
    # Save split statistics
    split_stats_file = os.path.join('results', 'split_statistics.json')
    with open(split_stats_file, 'w') as f:
        json.dump(split_stats, f, indent=2)
    print(f"📊 Split statistics saved to: {split_stats_file}")

def create_data_generators():
    """Create data generators with memory management."""
    print("🔄 Creating data generators...")
    
train_datagen = ImageDataGenerator(
    rescale=1./255,
        rotation_range=CONFIG['augmentation']['rotation_range'],
        zoom_range=CONFIG['augmentation']['zoom_range'],
        shear_range=CONFIG['augmentation']['shear_range'],
        width_shift_range=CONFIG['augmentation']['width_shift_range'],
        height_shift_range=CONFIG['augmentation']['height_shift_range'],
        horizontal_flip=CONFIG['augmentation']['horizontal_flip'],
        brightness_range=CONFIG['augmentation']['brightness_range'],
        channel_shift_range=CONFIG['augmentation']['channel_shift_range'],
        fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Memory-optimized data generators
train_gen = train_datagen.flow_from_directory(
        CONFIG['data']['temp_train_dir'], 
        target_size=(CONFIG['model']['img_size'], CONFIG['model']['img_size']), 
        batch_size=CONFIG['model']['batch_size'], 
        class_mode='categorical',
        shuffle=True
    )

val_gen = val_test_datagen.flow_from_directory(
        CONFIG['data']['temp_val_dir'], 
        target_size=(CONFIG['model']['img_size'], CONFIG['model']['img_size']), 
        batch_size=CONFIG['model']['batch_size'], 
        class_mode='categorical',
        shuffle=False
    )

test_gen = val_test_datagen.flow_from_directory(
        CONFIG['data']['temp_test_dir'], 
        target_size=(CONFIG['model']['img_size'], CONFIG['model']['img_size']), 
        batch_size=1, 
        class_mode='categorical', 
        shuffle=False
    )
    
    logger.info(f"Data generators created. Classes: {train_gen.num_classes}")
    logger.info(f"Batch size: {CONFIG['model']['batch_size']}")
    
    print(f"✅ Data generators ready: {train_gen.num_classes} classes, batch size {CONFIG['model']['batch_size']}")
    return train_gen, val_gen, test_gen

def build_advanced_model(num_classes, model_type='mobilenet'):
    """Build advanced model with regularization and ensemble support."""
    print(f"🏗️ Building {model_type} model with advanced features...")
    
    # Base model selection
    if model_type == 'mobilenet':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(CONFIG['model']['img_size'], CONFIG['model']['img_size'], 3)
        )
    elif model_type == 'resnet':
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(CONFIG['model']['img_size'], CONFIG['model']['img_size'], 3)
        )
    else:
        # Fallback to MobileNet for any other type
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(CONFIG['model']['img_size'], CONFIG['model']['img_size'], 3)
        )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Advanced architecture with regularization
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(CONFIG['model']['dropout_rate']),
        Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=CONFIG['model']['l2_reg'])),
        BatchNormalization(),
        Dropout(CONFIG['model']['dropout_rate']),
        Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=CONFIG['model']['l2_reg'])),
        BatchNormalization(),
        Dropout(CONFIG['model']['dropout_rate']),
        Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def create_advanced_callbacks():
    """Create advanced callbacks with early stopping."""
    print("⚙️ Creating advanced callbacks...")
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=CONFIG['model']['patience'],
            min_delta=CONFIG['model']['min_delta'],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='models/best_model_{epoch:02d}_{val_accuracy:.4f}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=f'logs/fit/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    return callbacks

def train_advanced_model(model, train_gen, val_gen, callbacks, model_name):
    """Train model with advanced techniques."""
    print(f"🚀 Training {model_name}...")
    
    # Compile with advanced metrics
    optimizer = Adam(learning_rate=CONFIG['model']['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )
    
    print(f"📊 Training on {len(train_gen)} batches per epoch")
    print(f"📊 Validation on {len(val_gen)} batches per epoch")
    
    start_time = time.time()
history = model.fit(
    train_gen,
        epochs=CONFIG['model']['num_epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"✅ {model_name} training complete in {training_time/60:.1f} minutes")
    
    return history

def fine_tune_advanced_model(model, base_model, train_gen, val_gen, callbacks, model_name):
    """Fine-tune model with advanced techniques."""
    print(f"🔧 Fine-tuning {model_name}...")
    
    # Unfreeze top layers
    base_model.trainable = True
    
    # Freeze bottom layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    optimizer = Adam(learning_rate=CONFIG['model']['fine_tune_lr'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )
    
    print(f"🔧 Fine-tuning with lower learning rate...")
    
    start_time = time.time()
    fine_tune_history = model.fit(
        train_gen,
        epochs=CONFIG['model']['fine_tune_epochs'],
    validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    fine_tune_time = time.time() - start_time
    print(f"✅ {model_name} fine-tuning complete in {fine_tune_time/60:.1f} minutes")
    
    return fine_tune_history

def evaluate_advanced_model(model, test_gen, model_name):
    """Evaluate model with comprehensive metrics."""
    print(f"📊 Evaluating {model_name}...")
    
    # Predict
    y_pred = model.predict(test_gen, verbose=0)
y_true = test_gen.classes
    
    # Convert predictions
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_classes)
    
    print(f"🎯 {model_name} Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save results
    results = {
        'model_name': model_name,
        'test_accuracy': accuracy,
        'predictions': y_pred_classes.tolist(),
        'true_labels': y_true.tolist()
    }
    
    with open(f'results/{model_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return accuracy, y_pred, y_true

def create_ensemble_models():
    """Create ensemble of different model architectures."""
    print("🎯 Creating ensemble models...")
    
    models = []
    model_types = ['mobilenet', 'resnet']  # Removed efficientnet due to compatibility issues
    
    for i, model_type in enumerate(model_types):
        print(f"🏗️ Building {model_type} model {i+1}/{len(model_types)}...")
        model, base_model = build_advanced_model(CONFIG['model']['num_classes'], model_type)
        models.append((model, base_model, model_type))
        
        # Memory check
        check_memory()
    
    return models

def train_ensemble():
    """Train ensemble of models."""
    print("🎯 Training ensemble models...")
    
    # Create ensemble models
    ensemble_models = create_ensemble_models()
    
    results = {}
    
    for i, (model, base_model, model_type) in enumerate(ensemble_models):
        model_name = f"{model_type}_model"
        print(f"\n🎯 Training {model_name} ({i+1}/{len(ensemble_models)})...")
        
        # Create data generators
        train_gen, val_gen, test_gen = create_data_generators()
        
        # Create callbacks
        callbacks = create_advanced_callbacks()
        
        # Train model
        history = train_advanced_model(model, train_gen, val_gen, callbacks, model_name)
        
        # Fine-tune model
        fine_tune_history = fine_tune_advanced_model(model, base_model, train_gen, val_gen, callbacks, model_name)
        
        # Evaluate model
        accuracy, y_pred, y_true = evaluate_advanced_model(model, test_gen, model_name)
        
        # Save model
        model.save(f'ensemble_models/{model_name}.keras')
        
        results[model_name] = {
            'accuracy': accuracy,
            'history': history.history,
            'fine_tune_history': fine_tune_history.history
        }
        
        # Memory cleanup
        del model, base_model, train_gen, val_gen, test_gen
        gc.collect()
        check_memory()
    
    return results

def create_ensemble_predictions(results):
    """Create ensemble predictions."""
    print("🎯 Creating ensemble predictions...")
    
    # Load test data
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        CONFIG['data']['temp_test_dir'],
        target_size=(CONFIG['model']['img_size'], CONFIG['model']['img_size']),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    
    ensemble_predictions = []
    
    for model_name in results.keys():
        model = tf.keras.models.load_model(f'ensemble_models/{model_name}.keras')
        predictions = model.predict(test_gen, verbose=0)
        ensemble_predictions.append(predictions)
        
        # Reset generator
        test_gen.reset()
    
    # Average predictions
    ensemble_pred = np.mean(ensemble_predictions, axis=0)
    ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
    
    # Calculate ensemble accuracy
    ensemble_accuracy = accuracy_score(test_gen.classes, ensemble_pred_classes)
    
    print(f"🎯 Ensemble Test Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    
    return ensemble_accuracy, ensemble_pred_classes, test_gen.classes

def main():
    """Main training pipeline with advanced techniques."""
    try:
        print("🚀 Starting Advanced Sign Language Recognition Training Pipeline")
        print("=" * 70)
        
        # Configure GPU and memory
        configure_gpu()
        
        # Create directories
        create_directories()
        
        # Validate data
        classes, class_stats = validate_data()
        
        # Split data
        split_data()
        
        # Train ensemble
        results = train_ensemble()
        
        # Create ensemble predictions
        ensemble_accuracy, ensemble_pred_classes, true_classes = create_ensemble_predictions(results)
        
        # Save final results
        final_results = {
            'ensemble_accuracy': ensemble_accuracy,
            'individual_results': results,
            'ensemble_predictions': ensemble_pred_classes.tolist(),
            'true_labels': true_classes.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open('results/final_ensemble_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print("🎉 Advanced training pipeline completed successfully!")
        print(f"🎯 Final Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")
        
        # Check if target accuracy achieved
        if ensemble_accuracy >= 0.90:
            print("🎉 TARGET ACCURACY ACHIEVED! (>90%)")
        else:
            print("⚠️ Target accuracy not achieved. Consider hyperparameter tuning.")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
