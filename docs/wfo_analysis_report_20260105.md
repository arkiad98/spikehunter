# WFO Performance Analysis & Fix Report (2026-01-05)

## 1. Issue Summary
- **Observation:** 
  - Global Model (Menu 3-1): **CAGR 6374%** (Recent period)
  - Walk-Forward Optimization (Menu 3-3): **CAGR -0.27% ~ -4.0%** (Same recent period)
- **Problem:** Significant performance discrepancy between the global "Sniper" model and the rolling WFO validation, despite covering similar recent timelines.

## 2. Root Cause Analysis

### A. The "Double Offset" Bug (Critical Logic Error)
- **Mechanism:** 
  - WFO logic correctly slices data up to the training end date (e.g., `2025-06-30`).
  - However, `train.py` (and `settings.yaml`) had `classification_train_end_offset: 6` enabled.
  - This caused the training module to **remove an additional 6 months** from the already sliced data.
- **Impact:** 
  - The model for the Period 8 test (2025.07 ~ 2025.12) was effectively trained only up to `2024.12.31`, missing the entire `2025.01 ~ 2025.06` bull run.
  - **Correction:** Modified `run_pipeline.py` to force `classification_train_end_offset = 0` during WFO execution.

### B. Model Specialization vs. Generalization
- **Global Model (Sniper):**
  - Settings: `scale_pos_weight: 1.0` (Precision-focused).
  - Context: Trained on the full recent history (including the bull run).
  - Result: Extremely high precision for the current market regime.
- **WFO (Generalist):**
  - Settings: Same `scale_pos_weight: 1.0`.
  - Context: Trained on historical windows (e.g., 2022-2023 Bear Market).
  - Result: The "Precision-focused" setting was too strict for volatile/bearish past regimes. The model failed to identify enough trade signals during training, resulting in underfitted models that performed poorly in testing.

## 3. Applied Solutions

### A. Code Fix
- **File:** `run_pipeline.py`
- **Fix:** In WFO mode, explicitly override the offset parameter:
```python
train_cfg['ml_params']['classification_train_end_offset'] = 0
```
- **Effect:** Ensures WFO models utilize the full training window provided, up to the day before the test period starts.

### B. Parameter Restoration (Balanced Model)
- **User Decision:** Restore historical optimal parameters that provided better balance (Recall vs Precision).
- **Restored Settings (`settings.yaml`):**
  - `scale_pos_weight`: **8.3044** (Shift from Precision 1.0 -> Balance/Recall ~8.3)
  - `n_estimators`: **540**
  - `learning_rate`: **0.01276**
  - `num_leaves`: **94**
  - `max_depth`: **17**
  - `colsample_bytree`: **0.9126**
  - `subsample`: **0.8480**
- **Expected Outcome:** 
  - The higher `scale_pos_weight` (Recall-focused) ensures the model remains robust and identifiable across various WFO periods (including past bear markets), preventing the "silence" observed with the strict setting.

## 4. Conclusion
The system has been updated to combine **correct data logic** (Double-Offset fix) with **robust historical parameters** (Balanced LGBM). This configuration is expected to bridge the gap between backtest reliability and real-world performance robustness.
