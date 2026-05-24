# 因子挖掘负面知识库 (Distilled Lessons)

> 上次更新时间: 2026-05-13 22:39:45

# Distilled Lessons from Failed Factor Mining Attempts

## 1. VWAP Deviation & Range-Normalized Signals Are Over-Mined
A large cluster of failures centers on variations of “VWAP deviation,” “range scaling,” and “z-scored VWAP premium” (e.g., `VWAP_ZRange_Opposing_Tilt_6to8h`, `Signed_Range_Scaled_VWAP_Deviation`, `VWAP_Deviation_Range_Ratio_20`, `VWAP_Deviation_Score_20`).  
**Lesson:** Simple VWAP-based mean-reversion or momentum signals are likely saturated or too noisy at the horizons tested. Further marginal tweaks (range normalization, exponential damping, regime gating) do not rescue them.

## 2. “Overheat / Panic / Exhaustion” Composite Signals Lack Edge
Many factors attempt to capture “overheating,” “panic selling,” or “exhaustion” via combinations of volume spikes, volatility contraction, and VWAP proximity (e.g., `Overheat_Reversal_Momentum`, `PanicSelling_Exhaustion_Indicator`, `Volume_Contraction_After_Spike`, `Momentum_Exhaustion_Volume_Divergence`).  
**Lesson:** These heuristic composites are either too fragile or already priced in. The market microstructure noise dominates the signal, leading to near-zero IC.

## 3. Funding Rate & Open Interest Dynamics Are Weak Predictors Here
Factors built on funding rate extremes, OI changes, and their interactions with VWAP or volume (e.g., `Funding_Rate_Extreme_Reversal`, `OI_Momentum_Divergence`, `Funding_Regime_Gated_VWAP_Deviation`) consistently fail.  
**Lesson:** In this dataset, funding/OI-based signals either lack predictive power or require more sophisticated modeling (e.g., non-linear regime shifts) rather than simple z-scores or linear composites.

## 4. Liquidation-Based Signals Show No Predictive Value
Several factors rely on liquidation volume, imbalance, or stress (e.g., `Liquidation_Amplified_Overheat`, `Liquidation_Imbalance_Reversal`, `LiqStress_VWAP_Premium_Factor`).  
**Lesson:** Raw liquidation metrics, especially when combined linearly with VWAP or volume, do not translate into alpha in this context—likely due to data latency, noise, or market adaptation.

## 5. Data Constraints & Column Availability Limit Factor Design
At least one failure (`Volume_Spike_After_Overheat`) was rejected because it used `$returns_5d`, which is not available.  
**Lesson:** Factor expressions must strictly adhere to the allowed column set (`$close, $funding_rate, $high, $long_liq, $low, $oi_change_pct, $open, $open_interest, $short_liq, $volume, $vwap`). Any factor requiring derived returns or external data must be precomputed or avoided.