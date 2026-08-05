#!/usr/bin/env python
"""
Convert mosei_senti_data.pkl (MultiBench format) to .npy files.

MultiBench MOSEI pkl structure:
{
    'train': {
        'text':   np.array (N, seq_len, text_dim),
        'audio':  np.array (N, seq_len, audio_dim),
        'vision': np.array (N, seq_len, vis_dim),
        'labels': np.array (N, 1, 1),
        'id':     np.array (N,)
    },
    'valid': {...},
    'test':  {...}
}

Output: mosei_text.npy, mosei_audio.npy, mosei_visual.npy, mosei_labels.npy
"""
import pickle
import numpy as np
import os
import sys
import time

INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/features/mosei_senti_data.pkl"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "data/features/"

print(f"Input:  {INPUT_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Loading pkl file (3.5GB, may take a minute)...")

t0 = time.time()
with open(INPUT_PATH, 'rb') as f:
    data = pickle.load(f)
print(f"Loaded in {time.time()-t0:.1f}s")

print(f"\nType: {type(data)}")
if isinstance(data, dict):
    print(f"Top-level keys: {list(data.keys())}")
    for k in data.keys():
        v = data[k]
        print(f"\n  [{k}] type={type(v)}")
        if isinstance(v, dict):
            print(f"    Sub-keys: {list(v.keys())}")
            for sk in v.keys():
                sv = v[sk]
                if isinstance(sv, np.ndarray) and sv.dtype.kind in ('f', 'i', 'u'):
                    print(f"      {sk}: shape={sv.shape}, dtype={sv.dtype}, "
                          f"range=[{sv.min():.4f}, {sv.max():.4f}]")
                elif isinstance(sv, np.ndarray):
                    print(f"      {sk}: shape={sv.shape}, dtype={sv.dtype} (non-numeric)")
                else:
                    print(f"      {sk}: type={type(sv)}")
        elif isinstance(v, np.ndarray):
            print(f"    shape={v.shape}, dtype={v.dtype}")

# ============================================================
# Convert to .npy
# ============================================================
print(f"\n{'='*60}")
print("Converting to .npy format...")
print(f"{'='*60}")

all_text = []
all_audio = []
all_visual = []
all_labels = []

for split in ['train', 'valid', 'test']:
    if split not in data:
        print(f"  WARNING: '{split}' not found, skipping")
        continue

    split_data = data[split]
    print(f"\n  Processing {split}:")

    # Get each modality
    for modality in ['text', 'audio', 'vision', 'labels']:
        if modality in split_data:
            arr = split_data[modality]
            print(f"    {modality:8s}: shape={arr.shape}, dtype={arr.dtype}")

    text = split_data['text'].astype(np.float32)
    audio = split_data['audio'].astype(np.float32)
    vision = split_data['vision'].astype(np.float32)
    labels = split_data['labels'].astype(np.float32)

    # Replace inf/nan
    audio[np.isinf(audio)] = 0.0
    audio[np.isnan(audio)] = 0.0
    vision[np.isinf(vision)] = 0.0
    vision[np.isnan(vision)] = 0.0
    text[np.isinf(text)] = 0.0
    text[np.isnan(text)] = 0.0

    # Mean pool over sequence dimension if 3D
    if text.ndim == 3:
        text = text.mean(axis=1)
    if audio.ndim == 3:
        audio = audio.mean(axis=1)
    if vision.ndim == 3:
        vision = vision.mean(axis=1)

    # Squeeze labels
    labels = labels.squeeze()

    print(f"    After pooling: text={text.shape}, audio={audio.shape}, "
          f"vision={vision.shape}, labels={labels.shape}")

    all_text.append(text)
    all_audio.append(audio)
    all_visual.append(vision)
    all_labels.append(labels)

# Concatenate all splits
text_arr = np.concatenate(all_text, axis=0)
audio_arr = np.concatenate(all_audio, axis=0)
visual_arr = np.concatenate(all_visual, axis=0)
labels_arr = np.concatenate(all_labels, axis=0)

print(f"\n  Combined: {len(labels_arr)} samples")
print(f"    text:   {text_arr.shape}")
print(f"    audio:  {audio_arr.shape}")
print(f"    vision: {visual_arr.shape}")
print(f"    labels: {labels_arr.shape}")

# ============================================================
# Convert labels to 7-class discrete emotions
# ============================================================
# MultiBench MOSEI labels are continuous sentiment in [-3, 3]
# We need 7-class emotion labels
# Strategy: discretize into 7 bins
if labels_arr.dtype in [np.float32, np.float64]:
    print(f"\n  Label stats: min={labels_arr.min():.4f}, max={labels_arr.max():.4f}, "
          f"mean={labels_arr.mean():.4f}")
    # Discretize [-3, 3] into 7 classes
    bins = [-3, -2, -1, 0, 1, 2, 3]
    labels_7class = np.digitize(labels_arr, bins=bins, right=False)
    labels_7class = np.clip(labels_7class, 0, 6).astype(np.int64)
    print(f"    7-class distribution: {np.bincount(labels_7class)}")
    labels_arr = labels_7class
else:
    labels_arr = labels_arr.astype(np.int64)
    print(f"  Label distribution: {np.bincount(labels_arr)}")

# ============================================================
# Pad/project dimensions to match paper requirements
# ============================================================
# Paper requires: text (768), audio (768), visual (342)

def pad_to_dim(arr, target_dim, name):
    """Pad or truncate feature dimension to target_dim."""
    current = arr.shape[-1]
    if current == target_dim:
        print(f"    {name}: {current} → {target_dim} (no change needed)")
        return arr
    elif current < target_dim:
        pad_width = target_dim - current
        print(f"    {name}: {current} → {target_dim} (padding +{pad_width} zeros)")
        return np.pad(arr, ((0, 0), (0, pad_width)), mode='constant')
    else:
        print(f"    {name}: {current} → {target_dim} (truncating -{current - target_dim})")
        return arr[:, :target_dim]

print(f"\n  Adjusting feature dimensions:")
text_arr = pad_to_dim(text_arr, 768, "text")
audio_arr = pad_to_dim(audio_arr, 768, "audio")
visual_arr = pad_to_dim(visual_arr, 342, "visual")

# ============================================================
# Save
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, "mosei_text.npy"), text_arr)
np.save(os.path.join(OUTPUT_DIR, "mosei_audio.npy"), audio_arr)
np.save(os.path.join(OUTPUT_DIR, "mosei_visual.npy"), visual_arr)
np.save(os.path.join(OUTPUT_DIR, "mosei_labels.npy"), labels_arr)

print(f"\n{'='*60}")
print(f"Done! Saved {len(labels_arr)} samples to {OUTPUT_DIR}:")
print(f"  mosei_text.npy:   {text_arr.shape} ({text_arr.dtype})")
print(f"  mosei_audio.npy:  {audio_arr.shape} ({audio_arr.dtype})")
print(f"  mosei_visual.npy: {visual_arr.shape} ({visual_arr.dtype})")
print(f"  mosei_labels.npy: {labels_arr.shape} ({labels_arr.dtype})")
print(f"  Label distribution: {np.bincount(labels_arr)}")
print(f"{'='*60}")
