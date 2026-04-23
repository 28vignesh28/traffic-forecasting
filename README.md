# Unified Traffic Forecasting Pipeline

This project provides a unified training and evaluation pipeline for four traffic forecasting models under identical data processing and metrics:
1. CADGT (Context-Aware Dynamic Graph Transformer)
2. CAMT-GATformer
3. AMC-DSTGNN
4. ST-ACENet

All models use the same dataloader, same feature set, and same evaluation logic so comparisons stay fair.

## Feature Set

Each input timestep includes 10 features per sensor:
1. Traffic speed
2. Temperature
3. Precipitation
4. Visibility
5. Wind speed
6. Holiday indicator
7. Time-of-day sine
8. Time-of-day cosine
9. Day-of-week sine
10. Day-of-week cosine

## Project Structure

```text
traffiic flow forecasting/
├── data/
│   ├── METR-LA.h5
│   ├── holidays_cache.json
│   └── weather_cache_*.csv
├── models/
│   ├── amc_dstgnn.py
│   ├── cadgt.py
│   ├── camt_gatformer.py
│   └── st_acenet.py
├── src/
│   ├── config.yaml
│   ├── data_loader.py
│   ├── train.py
│   ├── test.py
│   ├── app.py
│   ├── simulation_case_study.py
│   └── utils.py
├── logs/
├── saved_models/
├── plot_metrics.py
└── visualize.py
```

## Quick Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Demonstrate in Project Review

### 1. Show Training Command

```bash
python src/train.py --model CADGT
```

Model options:
- CADGT
- CAMT
- AMC_DSTGNN
- ST_ACENet

### 2. Show Evaluation Command

```bash
python src/test.py --model CADGT
```

This prints MAE, MSE, RMSE, and MAPE at 5/15/30/60-minute horizons.

### 3. Show Interactive Dashboard

```bash
streamlit run src/app.py
```

Use the sidebar to select:
- date/time
- sensor ID
- horizon

Then explain predicted speed, congestion level, and context-aware insights.

### 4. Show Comparative Visualizations

```bash
python visualize.py --node 91 --horizon_idx 11
python plot_metrics.py
```

Outputs are saved under visualizations.

### 5. Optional Case Study Output

```bash
python src/simulation_case_study.py
```

This generates focused congestion-behavior plots and summary CSVs for presentation.

## Configuration

Main settings are in src/config.yaml:
- training.window
- training.horizon
- training.batch_size
- training.learning_rate
- model_overrides

## Notes

- This README reflects files currently present in this repository.
- Previous ensemble scripts are not part of the current src directory.

## License

Provided for the Traffic Sequence Evaluation metrics review format.
