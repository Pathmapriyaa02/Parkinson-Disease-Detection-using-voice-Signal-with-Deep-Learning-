import librosa
import numpy as np

def extract_features(file_path, max_pad_len=176):

    try:
        audio, sample_rate = librosa.load(file_path, sr=None)

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        # Padding / trimming
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(
                mfccs,
                pad_width=((0, 0), (0, pad_width)),
                mode='constant'
            )
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.T   # (176, 40)

    except Exception as e:
        print(f"❌ Error loading file {file_path}")
        print("Reason:", e)
        return None


