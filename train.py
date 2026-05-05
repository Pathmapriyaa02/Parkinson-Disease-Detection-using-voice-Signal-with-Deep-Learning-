import os
import numpy as np
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from utils import extract_features

# ============================
# DATASET PATH
# ============================

DATASET_PATH = "voice_sample"

if not os.path.exists(DATASET_PATH):
    print("❌ Folder not found:", DATASET_PATH)
    exit()

# ============================
# LOAD DATA
# ============================

X = []

files = os.listdir(DATASET_PATH)
print("Files found:", files)

for file in files:
    if file.lower().endswith(".wav"):
        file_path = os.path.join(DATASET_PATH, file)
        print("Processing:", file)

        # 🔥 IMPORTANT: Get MFCC ONLY (176x40)
        features = extract_features(file_path)

        if features is not None:

            # If your new function returns flattened → reshape back
            if len(features.shape) == 2 and features.shape[1] > 1000:
                mfcc = features[:, :7040]   # take only MFCC part
                mfcc = mfcc.reshape(176, 40)
            else:
                mfcc = features  # already (176,40)

            X.append(mfcc)

if len(X) == 0:
    print("❌ No valid audio files loaded!")
    exit()

X = np.array(X)

print("✅ Dataset Loaded")
print("Dataset Shape:", X.shape)  # (samples, 176, 40)

# ============================
# MODEL ARCHITECTURE
# ============================

timesteps = X.shape[1]   # 176
features = X.shape[2]    # 40

input_layer = Input(shape=(timesteps, features))

# -------- Encoder --------
x = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
x = MaxPooling1D(2, padding='same')(x)

x = Conv1D(32, 3, activation='relu', padding='same')(x)
x = MaxPooling1D(2, padding='same')(x)

x = Bidirectional(LSTM(32, return_sequences=True))(x)

# -------- Decoder --------
x = Bidirectional(LSTM(32, return_sequences=True))(x)

x = UpSampling1D(2)(x)
x = Conv1D(32, 3, activation='relu', padding='same')(x)

x = UpSampling1D(2)(x)
output_layer = Conv1D(features, 3, activation='linear', padding='same')(x)

model = Model(input_layer, output_layer)

model.compile(
    optimizer=Adam(0.001),
    loss="mse"
)

model.summary()

# ============================
# TRAIN
# ============================

model.fit(
    X, X,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

# ============================
# AUTO THRESHOLD
# ============================

reconstructions = model.predict(X)
mse = np.mean(np.square(X - reconstructions), axis=(1, 2))

threshold = np.mean(mse) + 3 * np.std(mse)

print("✅ Auto Threshold:", threshold)

# ============================
# SAVE MODEL
# ============================

if not os.path.exists("saved_model"):
    os.makedirs("saved_model")

model.save("saved_model/cnn_bilstm_autoencoder.keras")
joblib.dump(threshold, "saved_model/threshold.pkl")

print("✅ Model & Threshold Saved Successfully!")





