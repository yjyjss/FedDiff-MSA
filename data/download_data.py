#!/usr/bin/env python
"""
Download CMU-MOSEI pre-extracted features and convert to .npy format.

Strategy:
  1. Try Zenodo (accessible from China) — download processed_mosei.pkl
  2. Convert from CMU SDK CSD format to numpy arrays
  3. Save as mosei_text.npy, mosei_audio.npy, mosei_visual.npy, mosei_labels.npy

If Zenodo download fails or features don't match paper specs,
use extract_features.py on a GPU server instead.

Usage:
  python download_data.py --output data/features/
  python download_data.py --output data/features/ --source zenodo
  python download_data.py --output data/features/ --source mult
"""

import argparse
import os
import sys
import pickle
import numpy as np
import urllib.request
import urllib.error
from typing import Dict, Optional

# ========================================
# Data source URLs
# ========================================

ZENODO_PROCESSED = "https://zenodo.org/api/records/17686067/files/processed_mosei.pkl/content"

ZENODO_CSD_FILES = {
    "Labels.csd": "https://zenodo.org/api/records/17668236/files/Labels.csd/content",
    "VisualFacet42.csd": "https://zenodo.org/api/records/17668236/files/VisualFacet42.csd/content",
    "TimestampedWordVectors.csd": "https://zenodo.org/api/records/17668236/files/TimestampedWordVectors.csd/content",
    "COVAREP.csd": "https://zenodo.org/api/records/17668236/files/COVAREP.csd/content",
}

# Google Drive file IDs (MultiBench MOSEI)
GDRIVE_FOLDER_ID = "1A_hTmifi824gypelGobgl2M-5Rw9VWHv"


def download_file(url: str, output_path: str, max_size_mb: float = 5000) -> bool:
    """Download a file with progress indication."""
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        
        with urllib.request.urlopen(req, timeout=30) as response:
            total = int(response.headers.get('Content-Length', 0))
            total_mb = total / (1024 * 1024)
            
            if total_mb > max_size_mb:
                print(f"  WARNING: File is {total_mb:.1f}MB (limit: {max_size_mb}MB)")
                proceed = input("  Continue? (y/n): ").strip().lower()
                if proceed != 'y':
                    return False
            
            print(f"  Downloading {total_mb:.1f}MB...")
            
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = (downloaded / total * 100) if total > 0 else 0
                    print(f"\r  Progress: {downloaded/(1024*1024):.1f}MB / {total_mb:.1f}MB ({pct:.1f}%)", end='', flush=True)
            
            print(f"\n  Done: {output_path}")
            return True
            
    except urllib.error.URLError as e:
        print(f"  ERROR: Cannot connect to {url}")
        print(f"  Reason: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def try_download_with_curl(url: str, output_path: str) -> bool:
    """Try downloading using curl as fallback."""
    import subprocess
    try:
        result = subprocess.run(
            ["curl", "-sL", "--connect-timeout", "30", "-o", output_path, url],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  Downloaded via curl: {size_mb:.1f}MB")
            return True
        else:
            print(f"  curl failed: {result.stderr[:200]}")
            return False
    except FileNotFoundError:
        print("  curl not available")
        return False
    except Exception as e:
        print(f"  curl error: {e}")
        return False


def convert_zenodo_pkl_to_npy(pkl_path: str, output_dir: str):
    """
    Convert Zenodo's processed_mosei.pkl to .npy files.
    
    The pkl file is in CMU SDK CSD format:
    {
        "video_id": [
            {
                "Labels": {"intervals": np.array, "features": np.array},
                "VisualFacet42": {"intervals": np.array, "features": np.array},
                "COVAREP": {"intervals": np.array, "features": np.array},
                "TimestampedWordVectors": {"intervals": np.array, "features": np.array},
            }
        ]
    }
    """
    print(f"\nLoading {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  Loaded. Type: {type(data)}, Keys: {list(data.keys())[:5]}...")
    
    # Extract features
    all_text = []
    all_audio = []
    all_visual = []
    all_labels = []
    
    if isinstance(data, dict):
        # CSD format: {video_id: [ {feature_name: {intervals, features}} ]}
        for video_id, segments in data.items():
            if not isinstance(segments, list):
                segments = [segments]
            
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                
                # Extract text features (GloVe word vectors → will be padded to 768)
                text_feat = None
                for key in ["TimestampedWordVectors", "glove", "GloVe", "text"]:
                    if key in seg:
                        feat = seg[key].get("features") if isinstance(seg[key], dict) else None
                        if feat is not None and len(feat) > 0:
                            # Average over time dimension
                            text_feat = np.mean(feat, axis=0) if feat.ndim > 1 else feat
                            break
                
                # Extract audio features (COVAREP)
                audio_feat = None
                for key in ["COVAREP", "covarep", "audio"]:
                    if key in seg:
                        feat = seg[key].get("features") if isinstance(seg[key], dict) else None
                        if feat is not None and len(feat) > 0:
                            audio_feat = np.mean(feat, axis=0) if feat.ndim > 1 else feat
                            break
                
                # Extract visual features (FACET)
                visual_feat = None
                for key in ["VisualFacet42", "FACET", "facet", "visual"]:
                    if key in seg:
                        feat = seg[key].get("features") if isinstance(seg[key], dict) else None
                        if feat is not None and len(feat) > 0:
                            visual_feat = np.mean(feat, axis=0) if feat.ndim > 1 else feat
                            break
                
                # Extract labels
                label = None
                for key in ["Labels", "labels", "label"]:
                    if key in seg:
                        feat = seg[key].get("features") if isinstance(seg[key], dict) else None
                        if feat is not None and len(feat) > 0:
                            # Labels might be continuous [-3,3] → convert to 7-class
                            label_val = np.mean(feat) if feat.ndim > 1 else feat[0]
                            # Discretize: map to 7 classes
                            label = int(np.clip((label_val + 3) / 6 * 6, 0, 6))
                            break
                
                if text_feat is not None and audio_feat is not None and visual_feat is not None:
                    all_text.append(text_feat)
                    all_audio.append(audio_feat)
                    all_visual.append(visual_feat)
                    if label is not None:
                        all_labels.append(label)
                    else:
                        all_labels.append(0)  # Default
    
    elif isinstance(data, (list, tuple)):
        # Might be a list of (text, audio, visual, label) tuples
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                all_text.append(item[0])
                all_audio.append(item[1])
                all_visual.append(item[2])
                all_labels.append(item[3])
    
    if not all_text:
        print("  ERROR: Could not extract features from pkl file.")
        print(f"  Data structure: {type(data)}")
        if isinstance(data, dict):
            first_key = list(data.keys())[0]
            print(f"  First key: {first_key}")
            print(f"  First value type: {type(data[first_key])}")
            if isinstance(data[first_key], list) and len(data[first_key]) > 0:
                print(f"  First element type: {type(data[first_key][0])}")
                if isinstance(data[first_key][0], dict):
                    print(f"  First element keys: {list(data[first_key][0].keys())}")
        return False
    
    # Convert to numpy arrays
    text_arr = np.array(all_text, dtype=np.float32)
    audio_arr = np.array(all_audio, dtype=np.float32)
    visual_arr = np.array(all_visual, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)
    
    # Pad/project to paper-required dimensions
    # Text: GloVe 300 → pad to 768 (or use actual RoBERTa features if available)
    if text_arr.shape[-1] < 768:
        pad_width = 768 - text_arr.shape[-1]
        text_arr = np.pad(text_arr, ((0, 0), (0, pad_width)), mode='constant')
    elif text_arr.shape[-1] > 768:
        text_arr = text_arr[:, :768]
    
    # Audio: COVAREP 74 → pad to 768
    if audio_arr.shape[-1] < 768:
        pad_width = 768 - audio_arr.shape[-1]
        audio_arr = np.pad(audio_arr, ((0, 0), (0, pad_width)), mode='constant')
    elif audio_arr.shape[-1] > 768:
        audio_arr = audio_arr[:, :768]
    
    # Visual: FACET 42 → pad to 342
    if visual_arr.shape[-1] < 342:
        pad_width = 342 - visual_arr.shape[-1]
        visual_arr = np.pad(visual_arr, ((0, 0), (0, pad_width)), mode='constant')
    elif visual_arr.shape[-1] > 342:
        visual_arr = visual_arr[:, :342]
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "mosei_text.npy"), text_arr)
    np.save(os.path.join(output_dir, "mosei_audio.npy"), audio_arr)
    np.save(os.path.join(output_dir, "mosei_visual.npy"), visual_arr)
    np.save(os.path.join(output_dir, "mosei_labels.npy"), labels_arr)
    
    print(f"\n  Saved {len(labels_arr)} samples:")
    print(f"    mosei_text.npy:   {text_arr.shape}")
    print(f"    mosei_audio.npy:  {audio_arr.shape}")
    print(f"    mosei_visual.npy: {visual_arr.shape}")
    print(f"    mosei_labels.npy: {labels_arr.shape}")
    print(f"    Label distribution: {np.bincount(labels_arr)}")
    
    return True


def convert_mult_pkl_to_npy(pkl_path: str, output_dir: str):
    """
    Convert MulT's mosei_senti_data_noalign.pkl to .npy files.
    
    MulT format:
    {
        'train': {'text': np.array (N, seq_len, text_dim),
                   'audio': np.array (N, seq_len, audio_dim),
                   'vision': np.array (N, seq_len, vis_dim),
                   'labels': np.array (N, 1, 1)},
        'valid': {...},
        'test': {...}
    }
    """
    print(f"\nLoading MulT pkl: {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  Splits: {list(data.keys())}")
    
    all_text = []
    all_audio = []
    all_visual = []
    all_labels = []
    
    for split in ['train', 'valid', 'test']:
        if split not in data:
            continue
        
        split_data = data[split]
        print(f"  {split}: text={split_data['text'].shape}, audio={split_data['audio'].shape}, "
              f"vision={split_data['vision'].shape}, labels={split_data['labels'].shape}")
        
        # Average over sequence dimension
        text = split_data['text'].astype(np.float32)
        audio = split_data['audio'].astype(np.float32)
        vision = split_data['vision'].astype(np.float32)
        labels = split_data['labels'].astype(np.float32)
        
        # Replace -inf in audio
        audio[audio == -np.inf] = 0
        
        # Mean pool over sequence length
        if text.ndim == 3:
            text = text.mean(axis=1)
        if audio.ndim == 3:
            audio = audio.mean(axis=1)
        if vision.ndim == 3:
            vision = vision.mean(axis=1)
        
        # Labels: continuous sentiment [-3, 3] → binary or 7-class
        if labels.ndim == 3:
            labels = labels.squeeze()
        if labels.ndim == 2:
            labels = labels.squeeze()
        
        # Convert to 7-class: discretize [-3, 3] into 7 bins
        labels_7class = np.digitize(labels, bins=[-3, -2, -1, 0, 1, 2, 3]) - 1
        labels_7class = np.clip(labels_7class, 0, 6)
        
        all_text.append(text)
        all_audio.append(audio)
        all_visual.append(vision)
        all_labels.append(labels_7class)
    
    text_arr = np.concatenate(all_text, axis=0)
    audio_arr = np.concatenate(all_audio, axis=0)
    visual_arr = np.concatenate(all_visual, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    
    # Pad to paper dimensions
    if text_arr.shape[-1] < 768:
        text_arr = np.pad(text_arr, ((0, 0), (0, 768 - text_arr.shape[-1])))
    elif text_arr.shape[-1] > 768:
        text_arr = text_arr[:, :768]
    
    if audio_arr.shape[-1] < 768:
        audio_arr = np.pad(audio_arr, ((0, 0), (0, 768 - audio_arr.shape[-1])))
    elif audio_arr.shape[-1] > 768:
        audio_arr = audio_arr[:, :768]
    
    if visual_arr.shape[-1] < 342:
        visual_arr = np.pad(visual_arr, ((0, 0), (0, 342 - visual_arr.shape[-1])))
    elif visual_arr.shape[-1] > 342:
        visual_arr = visual_arr[:, :342]
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "mosei_text.npy"), text_arr)
    np.save(os.path.join(output_dir, "mosei_audio.npy"), audio_arr)
    np.save(os.path.join(output_dir, "mosei_visual.npy"), visual_arr)
    np.save(os.path.join(output_dir, "mosei_labels.npy"), labels_arr)
    
    print(f"\n  Saved {len(labels_arr)} samples:")
    print(f"    mosei_text.npy:   {text_arr.shape}")
    print(f"    mosei_audio.npy:  {audio_arr.shape}")
    print(f"    mosei_visual.npy: {visual_arr.shape}")
    print(f"    mosei_labels.npy: {labels_arr.shape}")
    print(f"    Label distribution: {np.bincount(labels_arr)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Download CMU-MOSEI data and convert to .npy")
    parser.add_argument("--output", type=str, default="data/features/", help="Output directory")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "zenodo", "mult", "local"],
                        help="Data source: zenodo, mult, local (skip download)")
    parser.add_argument("--input", type=str, default=None,
                        help="Local pkl file path (for --source local)")
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "csd", "mult"],
                        help="Input pkl format: csd (CMU SDK) or mult (MulT)")
    args = parser.parse_args()
    
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Source: {args.source}")
    
    if args.source == "local":
        if not args.input:
            print("ERROR: --input required for --source local")
            sys.exit(1)
        
        pkl_path = args.input
        fmt = args.format
        
        if fmt == "auto":
            # Detect format from filename
            if "noalign" in pkl_path or "mult" in pkl_path.lower():
                fmt = "mult"
            else:
                fmt = "csd"
        
        print(f"Input: {pkl_path}")
        print(f"Format: {fmt}")
        
        if fmt == "mult":
            convert_mult_pkl_to_npy(pkl_path, output_dir)
        else:
            convert_zenodo_pkl_to_npy(pkl_path, output_dir)
        return
    
    pkl_path = os.path.join(output_dir, "downloaded_mosei.pkl")
    
    if args.source in ["auto", "zenodo"]:
        print(f"\n{'='*60}")
        print("Trying Zenodo (processed_mosei.pkl, 3.5GB)")
        print(f"{'='*60}")
        
        success = download_file(ZENODO_PROCESSED, pkl_path, max_size_mb=5000)
        
        if success:
            convert_zenodo_pkl_to_npy(pkl_path, output_dir)
            return
        
        print("\n  Zenodo download failed. Trying curl fallback...")
        success = try_download_with_curl(ZENODO_PROCESSED, pkl_path)
        
        if success:
            convert_zenodo_pkl_to_npy(pkl_path, output_dir)
            return
    
    if args.source in ["auto", "mult"]:
        print(f"\n{'='*60}")
        print("Trying MulT Dropbox data (requires VPN)")
        print(f"{'='*60}")
        
        mult_url = "https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEyvv2ZXHxa?dl=1"
        mult_path = os.path.join(output_dir, "mult_data.zip")
        
        success = download_file(mult_url, mult_path, max_size_mb=2000)
        
        if success:
            import zipfile
            with zipfile.ZipFile(mult_path, 'r') as z:
                z.extractall(output_dir)
            
            # Find the mosei pkl
            for root, dirs, files in os.walk(output_dir):
                for f in files:
                    if "mosei" in f.lower() and f.endswith('.pkl'):
                        pkl = os.path.join(root, f)
                        convert_mult_pkl_to_npy(pkl, output_dir)
                        return
    
    print(f"\n{'='*60}")
    print("All download attempts failed.")
    print(f"{'='*60}")
    print("\nManual download options:")
    print(f"  1. Zenodo:  {ZENODO_PROCESSED}")
    print(f"     Then: python download_data.py --source local --input <pkl_path> --format csd")
    print(f"  2. MulT:    https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEyvv2ZXHxa?dl=0")
    print(f"     Then: python download_data.py --source local --input <pkl_path> --format mult")
    print(f"  3. Kaggle:  https://www.kaggle.com/datasets/gnurtqh/cmu-mosei")
    print(f"     Then use extract_features.py to extract RoBERTa/Wav2Vec2/OpenFace features")
    print(f"  4. GPU server: Use extract_features.py with raw CMU-MOSEI video data")


if __name__ == "__main__":
    main()
