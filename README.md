# adaptive-regime-ai

An end-to-end AI-powered market intelligence system built and tested on NIFTY 50 (Indian equity markets). Combines Hidden Markov Models for regime detection, LSTM networks for directional forecasting, and a Random Forest meta-labelling layer into a unified signal pipeline — each model aware of the current market regime.

> **Ticker is configurable.** The system was developed and validated on NIFTY 50 (`^NSEI`) but works with any ticker supported by `yfinance` — US stocks, indices, ETFs. Change the `symbol` variable at the top of each notebook to switch markets.

---

## What this does

Most ML trading systems treat the market as a single stationary process. This system doesn't. It first detects *what kind of market* we are in (bull, bear, or sideways), then routes prediction to a model specifically trained on that regime. The result is a regime-conditioned signal that adapts to changing market structure.

```
Raw NIFTY data
      │
      ▼
┌─────────────────┐
│  HMM Regime     │  ← detects Bull / Bear / Sideways
│  Detector       │
└────────┬────────┘
         │  regime label
         ▼
┌─────────────────┐
│  LSTM Predictor │  ← separate model per regime
│  (per regime)   │
└────────┬────────┘
         │  direction probability
         ▼
┌─────────────────┐
│  MetaAlpha RF   │  ← filters low-confidence signals
│  Layer          │
└────────┬────────┘
         │
         ▼
    BUY / SELL / HOLD
```

---

## Project structure

```
nifty-ai-trading-system/
│
├── notebooks/
│   ├── 01_Market_Regime_Detector.ipynb   # HMM regime detection
│   ├── 02_LSTM_NIFTY_Predictor.ipynb     # Standalone LSTM baseline
│   ├── 03_Hybrid_HMM_LSTM_System.ipynb  # Full regime-conditioned pipeline
│   └── 04_MetaAlpha.ipynb               # Random Forest signal filter
│
├── research/
│   └── HMM_Market_Regime_Detection.pdf  # Supporting research writeup
│   └── LSTM_Market_Prediction.pdf
│
├── requirements.txt
└── README.md
```

---

## Models

### 1. HMM Regime Detector (`01_Market_Regime_Detector.ipynb`)

Uses a Gaussian Hidden Markov Model to identify 3 latent market states from NIFTY 50 daily data.

**Features used:**
- Log returns
- 20-day rolling volatility
- RSI (14-day)
- MA20 − MA50 spread
- Log trading volume

**Identified regimes:**
| Regime | Character | Avg Return | Volatility |
|--------|-----------|------------|------------|
| 0 | Bull | High positive | Low |
| 1 | Bear | Negative | High |
| 2 | Sideways | Near zero | Moderate |

**Algorithm:** Baum-Welch EM for parameter estimation, Viterbi for decoding.

---

### 2. LSTM Direction Predictor (`02_LSTM_NIFTY_Predictor.ipynb`)

Binary classifier predicting next-day market direction using 30-day sliding windows.

**Architecture:**
```
Input (30 days × 5 features)
  → LSTM (64 units, return_sequences=True)
  → Dropout (0.3)
  → LSTM (64 units)
  → Dropout (0.3)
  → Dense (1, sigmoid)
```

**Training:** Adam optimizer, binary cross-entropy loss, 20 epochs, batch size 16.

**Result:** ~52–55% direction accuracy on held-out test set (random baseline = 50%).

---

### 3. Hybrid HMM + LSTM System (`03_Hybrid_HMM_LSTM_System.ipynb`)

The core of this project. Trains a separate LSTM for each regime identified by the HMM. At inference time, the current regime is detected and prediction is routed to the matching LSTM.

**Signal logic:**
```python
if probability > 0.60:  →  STRONG BUY
if probability > 0.55:  →  BUY
if probability < 0.40:  →  SELL
else:                   →  HOLD
```

**Why this matters:** An LSTM trained only on bull-market sequences learns very different patterns from one trained on high-volatility bear sequences. Mixing them degrades performance. Regime conditioning fixes this.

---

### 4. MetaAlpha — Random Forest Signal Filter (`04_MetaAlpha.ipynb`)

Applies Triple Barrier labelling (take-profit / stop-loss / time-exit) to generate meta-labels, then trains a Random Forest to filter only the base signals worth acting on.

**Features:** Returns, volatility, RSI, MACD, MA spread.  
**Output:** A binary filter — for each signal from the LSTM, the RF decides whether to trade or skip.

---

## Quickstart

```bash
git clone https://github.com/yourusername/adaptive-regime-ai
cd nifty-ai-trading-system
pip install -r requirements.txt
```

Open notebooks in order: `01` → `02` → `03` → `04`.  
All data is downloaded automatically via `yfinance`. No CSV files needed.

---

## Requirements

```
yfinance
numpy
pandas
matplotlib
scikit-learn
hmmlearn
tensorflow
ta
```

---

## Key concepts

| Concept | Used in |
|--------|---------|
| Hidden Markov Models | Regime detection |
| Baum-Welch / Viterbi | HMM training & decoding |
| LSTM (Long Short-Term Memory) | Sequence-based direction prediction |
| Regime conditioning | Hybrid system — separate model per state |
| Triple Barrier method | Meta-labelling in RF filter |
| Random Forest | Signal quality classification |

---

## Limitations & honest notes

- Direction accuracy of 52–55% is modest. It becomes meaningful only when combined with position sizing and risk management.
- HMM regime boundaries are probabilistic — transition periods are uncertain.
- No transaction costs modelled. Real-world returns will be lower.
- Walk-forward validation not yet implemented — this is a priority next step.
- This is a research/learning project, not a live trading system.

---

## What's next

- [ ] Walk-forward validation across rolling windows
- [ ] Attention mechanism on top of LSTM
- [ ] Live paper trading via Zerodha Kite API
- [ ] Streamlit dashboard for live signal monitoring

---

## Author

**Piyush Patil** — AI & Data Science / Quantitative Finance  
Built as part of a personal quantitative research portfolio covering Indian and US equity markets.
