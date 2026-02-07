import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers

DATASET_PATH = "Enhanced Audio"
SR = 22050
N_MFCC = 40
MAX_LEN = 44
RANDOM_STATE = 42

def extract_mfcc(path):
    y, sr = librosa.load(path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc

X = []
y = []

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    label = 1 if folder == "car_crash" else 0
    print("Loading:", folder)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            try:
                feat = extract_mfcc(os.path.join(folder_path, file))
                X.append(feat)
                y.append(label)
            except:
                print("Skipped:", file)

X = np.array(X)[..., np.newaxis]
y = np.array(y)

print("Samples:", len(X))
print("Class counts:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)


def augment_mfcc(mfcc):
    mfcc = mfcc.copy()
    mfcc += np.random.normal(0, 0.01, mfcc.shape)  
    shift = np.random.randint(0, mfcc.shape[1] // 8)
    mfcc = np.roll(mfcc, shift, axis=1)          
    return mfcc

X_aug = []
y_aug = []

for i in range(len(X_train)):
    X_aug.append(X_train[i])
    y_aug.append(y_train[i])

    X_aug.append(augment_mfcc(X_train[i]))
    y_aug.append(y_train[i])

X_train = np.array(X_aug)
y_train = np.array(y_aug)

print("Training samples after augmentation:", len(X_train))

neg, pos = np.bincount(y_train)
class_weight = {
    0: 1.0,
    1: neg / pos
}

model = models.Sequential([

    layers.Conv2D(
        16, (3,3),
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        input_shape=X_train.shape[1:]
    ),
    layers.MaxPooling2D(2),
    layers.BatchNormalization(),

    layers.Conv2D(
        32, (3,3),
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    ),
    layers.MaxPooling2D(2),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-3)
    ),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

model.summary()


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[early_stop]
)

loss, acc, prec, rec = model.evaluate(X_test, y_test)
avg_accuracy = np.mean(history.history['accuracy'])
print(f"\nAverage Accuracy over {len(history.history['accuracy'])} epochs: {avg_accuracy:.3f}")