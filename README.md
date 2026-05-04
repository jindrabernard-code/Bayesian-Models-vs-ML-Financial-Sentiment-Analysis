# Bayesian Models vs Machine Learning in Financial Sentiment Analysis

> Academic Research Project — Masaryk University, 2026 | Nov 2025 – Jan 2026
>
> ## Overview
>
> This project compares the predictive performance of classical Bayesian statistical models and modern transformer-based language models (FinBERT) in financial sentiment classification. The core research question: can traditional probabilistic models compete with state-of-the-art AI when analyzing sentiment in financial news?
>
> ## Dataset
>
> - Source: Hugging Face financial text dataset
> - - Total size: 100,000 labeled financial headlines and reports
>   - - Sample used: 15,000 observations for model training and evaluation
>     - - Labels: Positive / Negative / Neutral sentiment
>      
>       - ## Models Implemented
>      
>       - | Model | Type |
>       - |---|---|
>       - | Complement Naive Bayes (CNB) | Bayesian |
> | Bayesian Multinomial Logistic Regression | Bayesian |
> | FinBERT | Transformer (BERT-based) |
>
> ## Technical Workflow
>
> - Text cleaning and preprocessing
> - - Feature engineering with TF-IDF vectorization (unigrams + bigrams)
>   - - Bayesian parameter estimation using MCMC / NUTS sampling
>     - - Fine-tuning of FinBERT transformer model
>       - - Train/test validation and overfitting monitoring
>        
>         - ## Evaluation Metrics
>        
>         - - Accuracy, F1-score, ROC-AUC, Log Loss
>           - - Confusion matrices
>             - - Posterior predictive checks
>               - - MCMC convergence diagnostics (R-hat, ESS, trace plots)
>                
>                 - ## Key Findings
>                
>                 - FinBERT significantly outperformed classical Bayesian classifiers:
>                 - - **77% classification accuracy**
>                   - - **ROC-AUC of 0.91+**
> - Superior probability calibration vs statistical models
>
> - Bayesian models offered greater **interpretability and transparency**, while transformer-based AI delivered substantially stronger predictive accuracy — highlighting the core trade-off between interpretability and performance.
>
> - ## Skills & Technologies
>
> - Python · NLP · Bayesian Statistics · Machine Learning · FinBERT · MCMC · TF-IDF · PyMC · Hugging Face Transformers
