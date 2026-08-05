#!/usr/bin/env python
"""
FedDiff-MSA Data Preparation Scripts

This package contains two scripts:
1. download_data.py — Download CMU-MOSEI pre-extracted features from accessible mirrors
2. extract_features.py — Extract RoBERTa/Wav2Vec2/OpenFace features from raw CMU-MOSEI (GPU server)

Paper requires these feature files in data/features/:
  mosei_text.npy    (N, 768)   — RoBERTa-base
  mosei_audio.npy   (N, 768)   — Wav2Vec2-base  
  mosei_visual.npy  (N, 342)   — OpenFace 2.0
  mosei_labels.npy  (N,)       — 7-class emotion labels

Data Sources (ranked by accessibility):
  1. Zenodo (accessible from China): https://zenodo.org/records/17686067
     — processed_mosei.pkl (3.5GB, CMU SDK format with COVAREP/Facet/GloVe features)
  2. Google Drive (requires VPN): MultiBench MOSEI folder
     — https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv
  3. Dropbox (requires VPN): MulT preprocessed data
     — https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEyvv2ZXHxa
  4. Kaggle (requires account): https://www.kaggle.com/datasets/gnurtqh/cmu-mosei
     — Raw video files (3.3GB), requires feature extraction
  5. CMU MultimodalSDK: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
     — Official source, downloads from CMU servers

NOTE: The features used in the paper (RoBERTa-base 768-dim, Wav2Vec2-base 768-dim,
OpenFace 2.0 342-dim) are NOT directly available in pre-extracted form from any single
source. They must be extracted from raw video/audio/text using the extract_features.py
script on a GPU server.

However, the MulT/MultiBench pre-extracted features use COVAREP (audio) and FACET (visual)
which have different dimensions. A conversion script is provided to adapt these features
to our framework by padding/projecting to the required dimensions.
"""
