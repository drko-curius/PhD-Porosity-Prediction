####################
#####LIBRARIES######
####################

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Define Paths of classes - Point to the parent 'exploitable' and 'defective'
data_dir = ''
exploitable_base = os.path.join(data_dir, 'exploitable')
defective_base = os.path.join(data_dir, 'defective')

#Load Data with Cube Groups
image_paths = []
labels = []
groups = []

print("Loading exploitable images")
for cube_folder in os.listdir(exploitable_base):
    cube_path = os.path.join(exploitable_base, cube_folder)
    if os.path.isdir(cube_path):
        for img_name in os.listdir(cube_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(cube_path, img_name)
                image_paths.append(img_path)
                labels.append(1)
                groups.append(cube_folder)

print("Loading defective images")
for cube_folder in os.listdir(defective_base):
    cube_path = os.path.join(defective_base, cube_folder)
    if os.path.isdir(cube_path):
        for img_name in os.listdir(cube_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(cube_path, img_name)
                image_paths.append(img_path)
                labels.append(0)
                groups.append(cube_folder)

image_paths = np.array(image_paths)
labels = np.array(labels)
groups = np.array(groups)

#Configure K-Fold Cross-Validation
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
fold_histories = [] # Store history for plotting
fold_reports = []

#Iterate through each fold
for fold, (train_idx, val_idx) in enumerate(gkf.split(image_paths, labels, groups=groups)):
    print(f"\n=== Training Fold {fold+1}/{n_splits} ===")
    X_train_paths, X_val_paths = image_paths[train_idx], image_paths[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    print(f"Validation cubes: {np.unique(groups[val_idx])}")

    #Create TensorFlow Datasets
    def load_and_preprocess_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [224, 224])
        image = tf.keras.applications.vgg16.preprocess_input(image)
        return image, label

    batch_size = 16
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_paths, y_train))
    train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=len(X_train_paths)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_paths, y_val))
    val_ds = val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    #Updated strategy: Using transfer learning with VGG 16 from ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #Train the model
    print("Training with Transfer Learning (VGG16 features)...")
    history = model.fit(train_ds,
                        epochs=15,
                        validation_data=val_ds,
                        verbose=1)
    fold_histories.append(history.history) # Save history for plotting

    #Evaluate on the validation set for this fold
    print(f"\n--- Evaluation for Fold {fold+1} ---")
    y_val_true = []
    y_val_pred_probs = []
    for images, true_labels in val_ds.unbatch().batch(1024):
        y_val_true.extend(true_labels.numpy())
        preds = model.predict(images, verbose=0)
        y_val_pred_probs.extend(preds.flatten())

    y_val_true = np.array(y_val_true)
    y_val_pred = (np.array(y_val_pred_probs) > 0.5).astype(int)

    #Generate and print classification report
    report = classification_report(y_val_true, y_val_pred, target_names=['Defective', 'Exploitable'], output_dict=True)
    fold_reports.append(report)
    print(classification_report(y_val_true, y_val_pred, target_names=['Defective', 'Exploitable'], digits=4))

    #Plotting graphs
    #Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {fold+1} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold+1} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'fold_{fold+1}_training_history.png')
    plt.close()

    #Plot Confusion Matrix
    cm = confusion_matrix(y_val_true, y_val_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Defective', 'Exploitable'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - Fold {fold+1}')
    plt.savefig(f'fold_{fold+1}_confusion_matrix.png') # Saves the plot to a file
    plt.close()

#Analyze the overall cross-validation results
print("\n" + "="*60)
print("CROSS-VALIDATION SUMMARY (Transfer Learning)")
print("="*60)
fold_accuracies = [report['accuracy'] for report in fold_reports]
fold_f1_defective = [report['Defective']['f1-score'] for report in fold_reports]
fold_f1_exploitable = [report['Exploitable']['f1-score'] for report in fold_reports]

print(f"\nMean Accuracy across {n_splits} folds: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print(f"Mean F1-Score (Defective): {np.mean(fold_f1_defective):.4f} (+/- {np.std(fold_f1_defective):.4f})")
print(f"Mean F1-Score (Exploitable): {np.mean(fold_f1_exploitable):.4f} (+/- {np.std(fold_f1_exploitable):.4f})")
