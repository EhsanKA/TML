<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-lightning-792ee5.svg" alt="PyTorch Lightning">
  <img src="https://img.shields.io/badge/license-GPL--3.0-green.svg" alt="License">
</p>

# TML — Trustworthy Machine Learning

A Python package for **dropout-based uncertainty quantification** and **dataset pruning** in binary classification tasks. TML implements a two-level training pipeline that uses Monte Carlo Dropout to produce reliable probability scores with associated uncertainty estimates.

## ✨ Key Features

- **Two-Level Training Pipeline** — First prunes unreliable samples, then trains on high-confidence data
- **Monte Carlo Dropout** — Uncertainty estimation through stochastic forward passes at inference time
- **Balanced Sampling** — Automatic handling of class imbalance during training
- **Custom Architectures** — Bring your own PyTorch Lightning models
- **Analysis Tools** — Built-in metrics (ROC-AUC, Brier score, calibration) and visualizations
- **GPU Acceleration** — Seamless CUDA support via PyTorch Lightning

## 📦 Installation

### From Source (Recommended)

```bash
git clone https://github.com/EhsanKA/tml.git
cd tml
conda env create --file environment.yaml
conda activate tml
pip install .
```

### Dependencies

The conda environment includes:

| Package | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| PyTorch Lightning | Training orchestration |
| scikit-learn | Metrics and evaluation |
| pandas | Data manipulation |
| matplotlib / seaborn | Visualization |
| tensorboard | Training logging |

## 🚀 Quick Start

```python
import torch
from tml.pipeline import Pipeline, ModelHandler
from models.mnist import CNNBinaryMNISTClassifier

# Prepare your data (must be torch tensors with binary labels 0/1)
X_train = ...  # Shape: (n_samples, *input_dims)
y_train = ...  # Shape: (n_samples,) with values {0, 1}

# Initialize your model
model = CNNBinaryMNISTClassifier(learning_rate=1e-3, dropout_rate=0.5)
model_handler = ModelHandler(model_instance=model)

# Create and run the pipeline
pipeline = Pipeline(
    model_handler=model_handler,
    data=X_train,
    hard_targets=y_train,
    batch_size=64,
    max_epochs=10,
    lower_threshold=0.3,
    upper_threshold=0.7,
    drop_iterations=10,
    seed=42
)

# Run n_steps iterations of the two-level training
pipeline.run(n_steps=5)

# Access results
probability_scores = pipeline.probability_scores  # Mean predictions
uncertainty_scores = pipeline.uncertainty_scores  # Prediction variance
```

## 🔬 How It Works

TML implements a two-level training strategy designed to improve prediction reliability:

### Level 1: Initial Training & Pruning

1. **Balanced Sampling** — Creates a balanced subset from imbalanced data
2. **Initial Training** — Trains the model on the balanced subset
3. **Prediction** — Generates predictions on the full dataset
4. **Pruning** — Identifies high-confidence predictions:
   - True Positives: samples with label=1 and prediction > `lower_threshold`
   - True Negatives: samples with label=0 and prediction < `upper_threshold`

### Level 2: Refined Training & Uncertainty Estimation

1. **Refined Training** — Re-trains on the pruned (high-confidence) subset
2. **Standard Prediction** — Generates probability scores
3. **MC Dropout** — Multiple forward passes with dropout enabled to estimate uncertainty

```
┌─────────────────────────────────────────────────────────────────┐
│                         TML Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │  Balanced   │───▶│   Train     │───▶│  Predict & Prune    │ │
│  │  Sampling   │    │   Level 1   │    │  (remove uncertain) │ │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘ │
│                                                   │             │
│                                                   ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ Uncertainty │◀───│   Train     │◀───│  High-Confidence    │ │
│  │  (MC Drop)  │    │   Level 2   │    │     Subset          │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Analysis & Visualization

TML provides tools for evaluating model performance and calibration:

```python
from tml.analysis import BinaryClassificationAnalysis

# Create analysis object
analysis = BinaryClassificationAnalysis(
    labels=y_test,
    probability_scores=pipeline.probability_scores,
    uncertainty_scores=pipeline.uncertainty_scores
)

# Metrics
roc_auc = analysis.calculate_roc_auc()
brier = analysis.calculate_brier_score()
ece = analysis.expected_calibration_error(n_bins=10)
optimal_thresh = analysis.find_optimal_threshold()

# Visualizations
analysis.plot_roc_curve()
analysis.plot_reliability_diagram()
analysis.plot_uncertainty_distribution()
analysis.plot_uncertainty_vs_confidence()
```

### Domain-Specific Plotting

For genomics applications (e.g., SNP classification):

```python
from tml.plotting import tml_plots

cutoff = tml_plots(
    final=results_array,      # (n_samples, 2) - [prob_score, uncertainty]
    neg_ind=negative_indices,
    hpos_ind=positive_indices,
    minScore=0.5,
    auc_cf=0.9,
    tpr_cf=0.95,
    out="output_prefix"
)
```

## 🏗️ Project Structure

```
tml/
├── tml/
│   ├── pipeline.py      # Main Pipeline and ModelHandler classes
│   ├── tml_dataset.py   # TMLDataset, BalancedSampler, prune function
│   ├── analysis.py      # BinaryClassificationAnalysis metrics & plots
│   ├── plotting.py      # Domain-specific visualization functions
│   └── utils.py         # Helper functions
├── models/
│   ├── mnist.py         # Example CNN for MNIST binary classification
│   └── model.py         # Generic binary classification models
├── notebooks/
│   ├── MNIST_MLP_0_7.ipynb                      # Usage example
│   ├── MNIST_MLP_0_7_test_model_performance.ipynb
│   └── MNIST_MLP_1_7.ipynb
├── environment.yaml     # Conda environment specification
├── pyproject.toml       # Package configuration
└── README.md
```

## ⚙️ Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_handler` | ModelHandler | required | Wrapper for your PyTorch Lightning model |
| `data` | Tensor | required | Input features |
| `hard_targets` | Tensor | required | Binary labels (0 or 1) |
| `batch_size` | int | 64 | Training batch size |
| `max_epochs` | int | 1 | Epochs per training phase |
| `learning_rate` | float | 1e-3 | Optimizer learning rate |
| `lower_threshold` | float | 0.3 | Pruning threshold for positive class |
| `upper_threshold` | float | 0.7 | Pruning threshold for negative class |
| `drop_iterations` | int | 2 | MC Dropout forward passes |
| `seed` | int | 42 | Random seed for reproducibility |

## 🎯 Custom Models

TML works with any PyTorch Lightning model. Requirements:

1. Must include `nn.Dropout` layers for MC Dropout to work
2. Output should be probability (use `nn.Sigmoid()` for binary classification)
3. Model class should be re-instantiable

```python
import torch.nn as nn
import pytorch_lightning as pl

class MyBinaryClassifier(pl.LightningModule):
    def __init__(self, input_dim, dropout_rate=0.5, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze()
        loss = nn.BCELoss()(y_pred, y.float())
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# Use with TML
model = MyBinaryClassifier(input_dim=100)
handler = ModelHandler(model_instance=model)
```

## 📓 Examples

See the [notebooks](notebooks/) directory for complete examples:

- **MNIST 0 vs 7 Classification** — Binary digit classification with CNN
- **Model Performance Testing** — Evaluation and analysis workflows

## 📄 License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ehsan Karimiara** — [e.karimiara@gmail.com](mailto:e.karimiara@gmail.com)

## 📚 Citation

If you use TML in your research, please cite:

```bibtex
@software{tml2024,
  author = {Karimiara, Ehsan},
  title = {TML: Trustworthy Machine Learning},
  year = {2024},
  url = {https://github.com/EhsanKA/tml}
}
```

---

<p align="center">
  <i>Built with PyTorch Lightning ⚡</i>
</p>
