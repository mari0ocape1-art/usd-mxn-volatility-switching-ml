# Foreign Exchange Risk Governance: USD/MXN Early Warning System

This production-grade quantitative architecture deploys a sequential three-phase pipeline designed for predictive volatility modeling, international regulatory compliance, and tail risk management for the USD/MXN exchange rate. The framework unifies traditional econometrics with supervised machine learning algorithms to optimize corporate capital allocation efficiency and automate foreign exchange hedging strategies in enterprise environments.

##  Operational Pipeline Infrastructure

The methodological framework is modularly structured across three highly integrated working notebooks. Click on each section below to expand the technical and statistical details:

<details>
<summary><b>Phase 1: Macroeconomic Data Acquisition & Time-Series Foundation (Click to expand)</b></summary>

### `USD_MXN_Macro_Risk_Analysis.ipynb`
* **Core Focus:** Establish the temporal precedence of global macroeconomic determinants, model conditional mean trajectories, and diagnose latent heteroskedasticity channels.
* **Methodology and Empirical Findings:**
  * **Variance Stabilization:** Mitigated non-stationarity by applying first-order logarithmic differencing to `USD_MXN_log` and `Tasa_Fed_10Y_log`, while maintaining the `VIX` index in levels to prevent statistical bias from over-differencing.
  * **Vector Autoregressive (VAR) Order Selection:** Executed mathematical lag optimization via information criteria, minimizing the Akaike Information Criterion (AIC) at lag 5 for the VIX and lag 2 for the U.S. 10-year Treasury yield.
  * **Granger Causality Testing:** Confirmed that movements in the U.S. sovereign interest rate differential act as an immediate, continuous structural driver starting from the first 24 hours. Conversely, the VIX exhibits an informational lag where its causal predictive power peaks and amplifies after 48 trading hours (2 business days), strictly aligning with its univariate AR(2) structure.
  * **Linear Modeling & ARCH Diagnosis:** Fit **ARIMA** specifications on the series. Sample moment evaluation on the residuals revealed a positive skewness of 0.54 and an absolute kurtosis of 6.00 for the Mexican Peso, providing parametric confirmation of severe fat-tailed behavior (leptokurtosis). Analyzing squared residuals via Engle’s ARCH test yielded a highly significant LM statistic of **110.21**, formally rejecting homoskedasticity and validating the presence of volatility clustering.

</details>

<details>
<summary><b>Phase 2: Conditional Volatility & Basel Compliance Framework (Click to expand)</b></summary>

### `USD_MXN_GARCH_VaR_Modeling.ipynb`
* **Core Focus:** Capture time-varying conditional variance dynamics, construct a parametric Value-at-Risk (VaR) framework in native currency scale (Pesos), and validate accuracy under international banking standards.
* **Methodology and Empirical Findings:**
  * **GARCH(1,1) Volatility Modeling:** Implemented a conditional variance structure calibrated via Maximum Likelihood Estimation (MLE) under a **Standardized Student's t-distribution**, simultaneously estimating the degrees of freedom parameter (ν ≈ 5.99) to analytically absorb tail risk. The system satisfied covariance-stationarity under a strong mean-reverting dynamic (α₁ + β₁ = 0.9183 < 1).
  * **Residual Post-Estimation Audit:** Executed Engle's ARCH test on the standardized errors, yielding a p-value of **0.3940**. Failing to reject the null hypothesis of homoskedasticity mathematically proves that the GARCH framework successfully filtered out all time-varying volatility dynamics.
  * **Parametric Dynamic VaR Metrics:** Projected asymmetric 95% and 99% risk thresholds. Restoring the metrics to their original Peso scale visually demonstrates an efficient behavior: risk boundaries contract tightly during stability regimes to liberate corporate capital and expand proactively upon the imminence of a market shock.
  * **Basel Committee Validation:** Subjected the 99% VaR out-of-sample backtest to Kupiec's Proportion of Overcharges (POF) Inconditional Coverage Test. The resulting p-value of **0.2706** officially placed the framework within the **Basel Green Zone**, achieving maximum regulatory accreditation with zero capital requirement penalties.

</details>

<details>
<summary><b>Phase 3: Regime-Switching Classification via Machine Learning (Click to expand)</b></summary>

### `USD_MXN_Regime_XGBoost.ipynb`
* **Core Focus:** Deploy an ensemble machine learning classifier to build a proactive Early Warning System (EWS) that anticipates discrete structural transitions into financial panic regimes (defined by breaches of the predictive 95% upper VaR threshold).
* **Methodology and Empirical Findings:**
  * **Target Formulation:** Engineered a strict binary classification target mapping historical days where actual log returns breached the 95% predictive upper risk limit (`df_var['USD_MXN_real_log'] > var_superior_95_predictivo`).
  * **Feature Engineering Matrix:** Fed the gradient boosting model with GARCH conditional volatility vectors, structural macro lag features (lags 1–5 for U.S. rates and lags 1–2 for VIX, mapping the leads validated in phase one), and 21-day rolling Spearman rank correlations to capture medium-term non-linear asset coupling.
  * **Cost-Sensitive Learning:** Managed severe class imbalance (panic events represented only 3.88% of historical observations) by calibrating the `scale_pos_weight` hyperparameter in XGBoost, rejecting synthetic oversampling (SMOTE) to preserve empirical financial noise.
  * **Business-Constrained Optimization:** Enforced an operational business constraint requiring a minimum 66% Recall floor for Class 1. The algorithm scanned the probability space and isolated an optimal operational decision threshold of **24.40%**.
  * **Out-of-Sample Evaluation:** The classifier successfully captured **67% of actual panic events** (4 out of 6 breaches in the test set). The resulting 137 False Positives are defended under corporate financial asymmetry: the operational costs of maintaining preventive hedges are infinitesimal compared to the catastrophic capital erosion triggered by an unmitigated False Negative.
  * **Feature Importance Insights:** Weight rankings confirmed the supremacy of immediate monetary policy innovations (`tasa_fed_lag_1`) and short-term Wall Street stress (`vix_lag_1`). The `vix_lag_2` predictor dropped in weight due to tree-based informational redundancy and multi-variable multicollinearity.

</details>

## Corporate Strategy & Business Valuation

1. **Working Capital Efficiency:** The time-varying shrinkage of the VaR boundaries during stable periods prevents cash or financial collateral from remaining idle, significantly improving corporate liquidity rotation metrics.
2. **Liquidity Constraint Mitigation (Margin Calls):** Utilizing a Standardized Student's t-distribution prevents the systematic underestimation of risk inherent to traditional Gaussian models, allowing the trading desk to calculate true capital cushions required during severe currency devaluations.
3. **Proactive Derivative Execution:** The XGBoost Early Warning System provides the corporate treasury with an automated operational window to structure and execute derivative positions (FX Forwards or Options) well ahead of the 48-hour macro-informational lag uncovered in the econometrics pipeline.
