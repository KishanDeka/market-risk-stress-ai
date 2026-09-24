# AI-Assisted Market Risk Stress Testing

A compact Python project for measuring portfolio losses under changing equity, interest-rate and volatility conditions. The first version combines conventional risk modelling with machine-learning-based market regime discovery. A later extension will generate joint market stress paths using a conditional deep generative model.

## Business objective

The project answers three questions:

1. How much could a portfolio lose under a specified market scenario?
2. Which products and risk factors drive the loss?
3. Can ML and generative AI help discover additional, coherent stress scenarios?

The example portfolio contains synthetic positions in an equity index, a fixed-rate bond, an equity forward and a European put. Public market observations provide the pricing and scenario inputs.

## Data

Daily data are downloaded from [FRED](https://fred.stlouisfed.org/):

| Series | Description | Use |
| --- | --- | --- |
| `SP500` | S&P 500 index | Equity returns and shocks |
| `VIXCLS` | CBOE Volatility Index | Volatility conditions |
| `DGS3MO` | 3-month Treasury yield | Short-rate input |
| `DGS5` | 5-year Treasury yield | Bond valuation and rate shocks |

The initial study period is 2017–2025. Portfolio holdings and risk limits are synthetic and stored in configuration files.

## Stage 1: risk modelling and ML

The first working version will include:

- Mark-to-market valuation for the bond, equity exposure, forward and European put.
- Full portfolio repricing under equity, rate and volatility shocks.
- Product-level profit-and-loss attribution.
- Historical Value at Risk and Expected Shortfall.
- Comparison of full repricing with delta, gamma, vega and DV01 approximations.
- Gaussian Mixture Model clustering to identify normal, equity-stress and rate-stress regimes.
- Regime-derived scenarios using historically observed joint factor movements.

Simple risk formulas:

```text
P&L = stressed portfolio value - current portfolio value
Loss = -P&L
VaR = selected quantile of the loss distribution
Expected Shortfall = average loss beyond the VaR threshold
```

The ML model supports scenario discovery. Financial pricing and full repricing remain the source of portfolio-loss estimates.

## Stage 2: AI-assisted generative scenarios

A conditional Temporal VAE will later generate joint paths for equity returns, yield changes and volatility changes under different market regimes. Generated paths will be checked for plausibility and passed through the same full-repricing engine.

The generative model will be compared with historical bootstrap, Gaussian and Student-t simulations using:

- Marginal and joint distributions.
- Correlation and temporal dependence.
- Tail co-movements.
- VaR and Expected Shortfall stability.
- Portfolio-loss distributions.

This extension is intended for scenario generation and reverse stress testing, not market-price prediction.

## Project structure

```text
market-risk-stress-ai/
├── configs/
│   ├── portfolio.yaml
│   └── scenarios.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_regime_analysis.ipynb
├── src/
|   ├── market_risk/
│       ├── data.py
│       ├── instruments.py
│       ├── portfolio.py
│       ├── scenarios.py
│       ├── historical_risk.py
│       ├── regime_model.py
│    └── gen_market_scenarios/
├── scripts/
│   ├── download_data.py
│   └── run_stress.py
├── tests/
├── outputs/
├── requirements.txt
└── README.md
```

## Tools

- Python, NumPy, pandas and SciPy
- scikit-learn for regime discovery
- PyTorch for the later Temporal VAE
- Matplotlib and Seaborn for reporting
- PyYAML for configuration
- pytest for validation
- GitHub Actions for automated checks

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py
python scripts/run_stress.py --config configs/scenarios.yaml
pytest
```

## Current roadmap

- [ ] Download and validate market data
- [ ] Explore risk-factor changes in a notebook
- [ ] Implement product pricing and portfolio aggregation
- [ ] Run hypothetical and historical stresses
- [ ] Calculate VaR, Expected Shortfall and sensitivities
- [ ] Add GMM market-regime discovery
- [ ] Add conditional generative scenario modelling
- [ ] Validate generated scenarios and implement reverse stress testing

## Scope

This is an educational prototype using public data and synthetic positions. It does not reproduce a bank's proprietary Economic Capital or regulatory methodology.
