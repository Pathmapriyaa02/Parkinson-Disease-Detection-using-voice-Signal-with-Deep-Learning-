"""
dataset_classifier.py
Wraps the trained RandomForest/GradientBoosting/KNN ensemble that was
trained on PD-Dataset.csv.

Usage
-----
from dataset_classifier import DatasetClassifier
clf = DatasetClassifier()
result = clf.predict(feature_dict)
# result = {
#   'label':       'Parkinson\'s Detected' | 'Normal (Healthy)',
#   'confidence':  0.82,          # probability 0-1
#   'risk_level':  'High' | 'Moderate' | 'Low' | 'Normal',
#   'proba_pd':    0.82,
#   'proba_norm':  0.18,
#   'feature_comparison': [{name, value, pd_mean, norm_mean, deviation}, ...]
# }
"""

import os
import pickle
import numpy as np
import pandas as pd

MODEL_DIR    = os.path.join(os.path.dirname(__file__), 'dataset_model')
CLF_PATH     = os.path.join(MODEL_DIR, 'classifier.pkl')
SCALER_PATH  = os.path.join(MODEL_DIR, 'scaler.pkl')
REF_PATH     = os.path.join(MODEL_DIR, 'feature_reference.csv')
FEAT_PATH    = os.path.join(MODEL_DIR, 'features.pkl')

FEATURE_NAMES = [
    'Jitter', 'Shimmer', 'ZCR', 'HNR', 'RPDE', 'PPE',
    'MFCC0', 'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6',
    'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12'
]


class DatasetClassifier:
    def __init__(self):
        with open(CLF_PATH, 'rb')   as f: self.clf    = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f: self.scaler = pickle.load(f)
        self.ref = pd.read_csv(REF_PATH, index_col=0)

    def predict(self, feature_dict: dict) -> dict:
        """
        Parameters
        ----------
        feature_dict : dict  {feature_name: float}

        Returns
        -------
        dict with keys: label, confidence, risk_level,
                        proba_pd, proba_norm, feature_comparison
        """
        # Build input vector in correct order
        vec = np.array([feature_dict.get(f, 0.0) for f in FEATURE_NAMES],
                       dtype=np.float32).reshape(1, -1)

        vec_scaled = self.scaler.transform(vec)
        proba      = self.clf.predict_proba(vec_scaled)[0]

        # clf was trained with Status 0=healthy, 1=PD
        # VotingClassifier preserves class order from training data
        classes = self.clf.classes_          # [0, 1]
        idx_pd   = list(classes).index(1)
        idx_norm = list(classes).index(0)

        proba_pd   = float(proba[idx_pd])
        proba_norm = float(proba[idx_norm])

        # Risk level
        if proba_pd >= 0.75:
            risk_level = 'High'
        elif proba_pd >= 0.55:
            risk_level = 'Moderate'
        elif proba_pd >= 0.40:
            risk_level = 'Low'
        else:
            risk_level = 'Normal'

        label = "Parkinson's Detected" if proba_pd >= 0.50 else "Normal (Healthy)"

        # Per-feature comparison vs dataset reference
        comparison = []
        for feat in FEATURE_NAMES:
            val       = feature_dict.get(feat, 0.0)
            pd_mean   = float(self.ref.loc[feat, 'pd_mean'])
            norm_mean = float(self.ref.loc[feat, 'norm_mean'])
            pd_std    = float(self.ref.loc[feat, 'pd_std']) + 1e-10
            norm_std  = float(self.ref.loc[feat, 'norm_std']) + 1e-10

            # z-score distance from each group mean
            z_pd   = abs(val - pd_mean)   / pd_std
            z_norm = abs(val - norm_mean) / norm_std

            closer_to = 'PD' if z_pd < z_norm else 'Normal'

            comparison.append({
                'name':        feat,
                'value':       round(val, 4),
                'pd_mean':     round(pd_mean, 4),
                'norm_mean':   round(norm_mean, 4),
                'closer_to':   closer_to,
                'z_pd':        round(z_pd, 3),
                'z_norm':      round(z_norm, 3),
            })

        return {
            'label':              label,
            'confidence':         round(max(proba_pd, proba_norm), 4),
            'risk_level':         risk_level,
            'proba_pd':           round(proba_pd, 4),
            'proba_norm':         round(proba_norm, 4),
            'feature_comparison': comparison,
        }
