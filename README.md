# Features-Aware Autoencoder

A collaborative-filtering project for predicting whether a student will answer a
diagnostic question correctly, based on sparse historical response data. The
centerpiece is a **feature-aware (subject-aware) autoencoder** that augments a
standard item autoencoder with side information about each question, improving
prediction accuracy over the baseline models.

Built for UofT CSC311 (Introduction to Machine Learning). The dataset is drawn
from Eedi, an online education platform, and consists of student responses to
diagnostic multiple-choice questions.

## Problem

Given a sparse `student × question` matrix of correct/incorrect responses (most
entries missing), predict the correctness of held-out (student, question) pairs.
This is a matrix-completion / collaborative-filtering task evaluated by prediction
accuracy on validation and test splits.

## The Features-Aware Autoencoder

The core contribution (see `partb.ipynb`) is an item autoencoder that is *aware*
of question features rather than relying on the response matrix alone.

- **Item-based encoding.** Each question is represented by its column of student
  responses concatenated with an observation mask (`[answer_values, answer_mask]`),
  so the model knows which entries are observed vs. missing.
- **Subject feature branch.** Every question carries a multi-hot vector over
  ~388 subjects (parsed from `data/question_meta.csv`). A small subject encoder
  (`num_subjects → 16 → 64`) projects these features and injects them into the
  autoencoder bottleneck through a learned `subject_gate`, letting the model share
  strength across questions that cover the same topics.
- **Bias terms.** Per-question, per-learner, and global bias parameters capture
  overall difficulty and ability offsets.
- **Regularization.** Dropout, gradient clipping, and AdamW weight decay.
- **Two-seed ensemble.** Two independently seeded models are trained and their
  predicted probabilities are averaged for the final prediction, reducing variance.

## Baseline models

The repository also contains the Part A baselines the autoencoder is compared
against:

| File | Model |
| --- | --- |
| `knn.py` | k-Nearest Neighbors imputation (user- and item-based) |
| `item_response.py` | Item Response Theory (1-PL / Rasch model) |
| `matrix_factorization.py` | Matrix factorization (SVD / ALS-style) |
| `neural_network.py` | Vanilla item autoencoder baseline |
| `ensemble.py` | Bagged IRT ensemble (bootstrap resampling) |
| `majority_vote.py` | Majority-vote baseline |
| `utils.py` | Data loading and evaluation helpers |

## Project structure

```
.
├── partb.ipynb          # Features-aware (subject-aware) autoencoder + two-seed ensemble
├── neural_network.py    # Baseline item autoencoder
├── item_response.py     # IRT baseline
├── ensemble.py          # Bagged IRT ensemble
├── knn.py               # kNN baseline
├── matrix_factorization.py
├── majority_vote.py
├── utils.py             # Data loaders / evaluation
├── data/                # Response matrix + question/student/subject metadata
└── output/              # Report (PDF)
```

## Data

The `data/` directory contains the response splits and metadata:

- `train_data.csv`, `valid_data.csv`, `test_data.csv`, `private_test_data.csv` —
  response records as `(user_id, question_id, is_correct)`.
- `train_sparse.npz` — the training responses as a sparse `student × question` matrix.
- `question_meta.csv` — per-question subject lists (source of the subject features).
- `student_meta.csv`, `subject_meta.csv` — additional metadata.

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn torch matplotlib
```

## Usage

Run the features-aware autoencoder and two-seed ensemble:

```bash
jupyter notebook partb.ipynb
```

Run a baseline, e.g. the item autoencoder or IRT:

```bash
python neural_network.py
python item_response.py
```

Each script loads the data from `./data`, trains the model (selecting
hyperparameters on the validation split), and reports validation and test accuracy.

## Report

A write-up of the method and results is in
`output/pdf/csc311_partb_subject_aware_autoencoder_report.pdf`.
