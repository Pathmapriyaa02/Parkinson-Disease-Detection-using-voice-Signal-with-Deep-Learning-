# Supervised Voice Classifier Training

Use this when you want the Voice Analysis page to say:

```text
Normal
Low Risk
Moderate Risk
High Risk
```

based on a trained Parkinson-vs-healthy voice classifier.

## 1. Prepare Folders

Create this structure in the same folder as `app.py`:

```text
voice_dataset
  healthy
    healthy_001.wav
    healthy_002.wav
  parkinson
    pd_001.wav
    pd_002.wav
```

Use clear voice files. WAV is best.

Recommended audio format:

```text
WAV
Mono
16000 Hz
4 to 8 seconds
Same phrase for every person, for example "aaaaaa"
```

## 2. Train

Run:

```powershell
python train_voice_classifier.py
```

It will save:

```text
dataset_model/voice_classifier.pkl
dataset_model/voice_scaler.pkl
dataset_model/voice_features.pkl
dataset_model/voice_metrics.json
```

## 3. Restart App

Run:

```powershell
python app.py
```

Now the Voice Analysis page will use the supervised voice classifier first.

Risk mapping:

```text
PD probability < 40%  = Normal
40% to 54%            = Low Risk
55% to 74%            = Moderate Risk
75% and above         = High Risk
```

## Important

This is still a screening system, not a final medical diagnosis.
Use the phrase "Parkinsonian voice risk" instead of "confirmed Parkinson's disease".
