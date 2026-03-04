# Unified Traffic Forecasting Pipeline

This repository contains a unified, scalable, and standardized deep learning pipeline for evaluating four distinct traffic forecasting architectures under perfectly identical testing conditions:
1. **CADGT** (Context-Aware Dynamic Graph Transformer)
2. **CAMT-GATformer** (Context-Aware Multi-Task Graph Attention Transformer)
3. **AMC-DSTGNN** (Adaptive Multi-Context Dynamic Spatial-Temporal Graph Neural Network)
4. **ST-ACENet** (Spatial-Temporal Adaptive Context-Enhanced Network)

By utilizing a single configuration and dataloader, we guarantee **zero data leakage**, completely standardized metrics evaluation, and identical input architectures across all models. The models natively consume exactly 10 features extracted simultaneously from raw traffic speed, local Open-Meteo weather parameters, public US Holidays, and cyclic temporal node encodings. 

## Project Architecture

```text
c:/Users/vigne/Desktop/quadratic voting system/
├── data/
│   ├── METR-LA.h5              # Raw sequence traffic dataset
│   └── adj_METR-LA.pkl         # Static distances / physical highway graph adj 
├── models/                     # Core PyTorch nn.Module architectures
│   ├── amc_dstgnn.py
│   ├── cadgt.py
│   ├── camt_gatformer.py
│   └── st_acenet.py
├── src/                        # The Unified Pipeline
│   ├── config.yaml             # Single file tuning network hyperparams 
│   ├── data_loader.py          # Enforces 10 identical input features for all models
│   ├── test.py                 # Evaluates 60-Minute Horizon Model Checkpoints
│   ├── train.py                # Runs generic epochs with automatic logging
│   └── utils.py                # Standardized MAE/RMSE/R2 masking limits
├── logs/                       # Automated standard-output execution history 
├── saved_models/               # Location of successful compilation `.pth` best weights
└── legacy_archive.zip          # The original unformatted project history
```

## Running the Pipeline

All settings are localized inside `src/config.yaml`. To train any of the 4 networks, specify the abbreviation as the `--model` argument:

### 1. Training Parameters
Ensure your hyper-parameters in `src/config.yaml` are set. The dataloader uses `training.window` (default 12) history to predict `training.horizon` (default 12) timesteps dynamically.

```bash
python src/train.py --model CADGT
```
*(Options: CADGT, CAMT, AMC_DSTGNN, ST_ACENet)*

### 2. Evaluating 60-Minute Targets
To calculate validation metrics explicitly masked for non-zero null sensors on your best saved checkpoint weights:

```bash
python src/test.py --model CADGT
```

### 3. Modifying the Input Stream
Every feature ingested by a model passes through `src/data_loader.py > get_dataloaders()`. The unified standard features list includes: `[Traffic Speed, Temperature, Precipitation, Visibility, Wind Speed, IsHoliday, TOD_sin, TOD_cos, DOW_sin, DOW_cos]`. 
If you add an 11th feature, update the initialization size configurations dynamically within `train.py`.

### 4. Quadratic Voting Ensemble
After all 4 models are trained, combine their predictions using the Quadratic Voting ensemble:

```bash
python src/ensemble_qv.py
```

**How QV works:** Each model receives a "voice credit" budget inversely proportional to its validation MAE. Under QV, casting *v* votes costs *v²* credits, so effective votes = `√(1/MAE)`. Models with lower error earn disproportionately more influence, but no single model can dominate — this is the core democratic property of QV.

The script outputs a side-by-side comparison table of all individual models vs. the ensemble at 5/15/30/60-minute horizons.

## License
Provided for the Traffic Sequence Evaluation metrics review format.
