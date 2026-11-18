# Glass Type Classification: Random Forest & Hyperparameter Optimization

Multi-class classification of glass types using a **Random Forest** classifier on the
[Glass Identification dataset](https://www.openml.org/d/41) (214 samples, 9 chemical features, 6 classes).

The project compares **five hyperparameter optimization methods** on the same model, search space, data split, and cross-validation setup.

> Random Forest is not trained with gradient-based optimizers (GD, SGD, RMSprop, Adam, ...), so hyperparameter search methods are used instead.

## Dataset

- Source: [OpenML Glass Identification (ID 41)](https://www.openml.org/d/41), loaded with `fetch_openml(data_id=41)`
- Features: RI, Na, Mg, Al, Si, K, Ca, Ba, Fe
- Target: glass type (6 classes: build wind float, build wind non-float, vehic wind float, containers, tableware, headlamps)
- The dataset is small and imbalanced (the smallest class has only 9 samples), so stratified splitting and Macro F1 are used.

## Project Structure

| Notebook | Method |
|---|---|
| `random-forest-glass-classification-grid-search.ipynb` | Grid Search |
| `random-forest-glass-classification-random-search.ipynb` | Random Search |
| `random-forest-halving-grid-search.ipynb` | Halving Grid Search |
| `random-forest-halving-random-search.ipynb` | Halving Random Search |
| `random-forest-bayesian-optimization.ipynb` | Bayesian Optimization (Optuna) |

Each notebook follows the same pipeline: data loading, data understanding, class distribution, train/test split, hyperparameter search, evaluation, confusion matrix, and feature importance.

## Experimental Setup

- **Train/test split:** 80/20, stratified (`random_state=42`)
- **Cross-validation:** 5-fold Stratified K-Fold
- **Scoring metric:** Macro F1-score
- **Search space:**

| Hyperparameter | Values |
|---|---|
| `n_estimators` | 50, 100, 200 |
| `max_depth` | None, 5, 10, 20 |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | sqrt, log2 |

- **Budget:** Grid and Halving Grid search all 216 combinations; Random Search, Halving Random Search, and Bayesian Optimization use 30 candidates/trials.
- **Evaluation:** Accuracy, Precision (macro), Recall (macro), Macro F1, confusion matrix, feature importance, search time.

## Results

| Method | CV Macro F1 | Test Accuracy | Test Precision | Test Recall | Test Macro F1 | Time (s) |
|---|---|---|---|---|---|---|
| Grid Search | 0.6635 | 0.8372 | 0.7523 | 0.7889 | 0.7565 | 62.9 |
| Random Search | 0.6603 | 0.8372 | 0.8686 | 0.7881 | 0.7865 | 14.4 |
| Halving Grid Search | 0.6531 | 0.7674 | 0.7327 | 0.6389 | 0.6458 | 157.9 |
| Halving Random Search | 0.6390 | 0.7907 | 0.8036 | 0.7659 | 0.7452 | 20.0 |
| Bayesian Optimization | 0.6560 | 0.7674 | 0.7883 | 0.7540 | 0.7341 | 23.9 |

### Key Findings

- **Random Search** gave the best balance: the highest test Macro F1, a cross-validation score close to Grid Search, and about 4x faster.
- **Grid Search** achieved the highest cross-validation score, but at a much higher computational cost.
- **Halving Grid Search** was the slowest and had the lowest test performance. On a dataset this small, the first round already costs almost as much as a full grid search, so halving saves nothing.
- **Bayesian Optimization** was fast and competitive, but did not outperform Random Search on this small, discrete search space.

### Notes

- Halving methods report CV scores on a 120-sample subset (last iteration), so they are not directly comparable to the other methods.
- The test set has only 43 samples, so small differences between methods are within noise.
- Execution times vary between runs and machines and should be treated as approximate.

## Requirements

- Python 3.9+
- numpy, pandas, matplotlib, seaborn, scikit-learn, optuna

## Usage

```bash
pip install numpy pandas matplotlib seaborn scikit-learn optuna
jupyter notebook
```

Open any notebook and run all cells. The dataset is downloaded automatically from OpenML (internet connection required).

## License

This project is for educational purposes.
