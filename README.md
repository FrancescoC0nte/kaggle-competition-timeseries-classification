# Timeseries Classification of Pain Levels from simulated sensor data

Project for the AN2DL First Challenge — Ottima ANNata

## What this is about

This project focuses on time series classification using deep learning techniques and Recurrent Neural Networks. The goal is to classify sensor time series into three pain levels: **no pain**, **low pain**, and **high pain**.

The dataset contains 661 time series, each consisting of 160 consecutive timestamps described by 38 heterogeneous features. The project addresses several challenges in this setting, including class imbalance, feature selection, temporal windowing, and the sensitivity of model performance to the training/validation split.

The work explores different neural architectures, starting from recurrent models such as GRUs and LSTMs and progressing to models combining convolutional and attention mechanisms. The main objective was to identify an effective architecture while maintaining robust generalization across different data partitions.

The complete description of the methodology, experiments, and results is available in the [project report](Project%20Report.pdf). This README provides a concise overview of the implementation choices and the main findings.

## Repository structure

The repository is organized as follows:

```text
.
├── Code/                  # Model implementations and analysis scripts
├── Project Report.pdf     # Report for the AN2DL First Challenge
└── README.md
```

The source code is available in the [`Code/`](Code/) directory.

## Problem analysis

The dataset consists of 661 time series with 160 timestamps and 38 heterogeneous features. Each series is associated with one of three classes: no pain, low pain, or high pain.

Two preprocessing choices were particularly relevant:

- **Class imbalance.** The three pain classes are not equally represented. We therefore used class weights to penalize errors on underrepresented classes. In a subsequent experiment, we replaced weighted Cross-Entropy with Focal Loss to further address the imbalance.
- **Feature selection.** The feature `joint 30` was constant and equal to 0.5 across all samples, so it was removed because it provided no useful information for classification.

For the more complex architecture, we also applied a logarithmic transformation to features `joint 13` to `joint 29`, whose values were heavily concentrated near zero.

## Methodology

To prevent data leakage, the train/validation split was performed at the time-series level: each complete series was assigned either to the training set or to the validation set. The resulting split contained approximately 90% training data and 10% validation data.

Each time series was then divided into fixed-length, overlapping windows. This allowed the models to learn from different local temporal patterns while ensuring that windows from the same original series did not appear in both training and validation sets.

We initially compared GRU and LSTM models using different window sizes and strides. The best configuration used:

- **Input:** sensor features, excluding the series index and timestamp
- **Recurrent layer:** one bidirectional GRU layer with 256 neurons
- **Output:** fully connected layer with softmax over the three pain classes
- **Windowing:** window length of 40 timestamps and stride of 5
- **Series-level prediction:** majority voting across the predictions of its windows

We also tested L2 regularization and Dropout. The best configuration used no L2 regularization and a dropout rate of 0.3. A Step Scheduler was used to decay the learning rate.

### Bagging

The results of five-fold cross-validation showed that performance could depend substantially on how the training and validation sets were sampled. To improve robustness, we applied Bagging: five GRU models were trained on different data splits, and their predictions were combined through voting.

### Convolutional and attention-based architecture

We also investigated a more expressive architecture combining:

- **Multi-Kernel Conv1D:** extracts local temporal patterns at multiple scales
- **Group Normalization:** merges and normalizes convolutional features
- **Two-layer LSTM:** models temporal dynamics on the enriched representation
- **Attention:** reweights time steps to provide a long-term perspective
- **Output layer:** softmax classification over the three pain classes

To address class imbalance in this experiment, we also tried upsampling minority classes through Jittering, adding Gaussian noise to sensor data to generate synthetic samples.

## Key results

The best-performing model was the BaggedGRU, which achieved an F1 score of **95.70%** on the test set. The single GRU achieved **95.15%**, while the more complex AttentionCNNLSTM model achieved **87.10%**.

| Model | Test F1 score |
|---|---:|
| GRU | 95.15% |
| BaggedGRU | **95.70%** |
| AttentionCNNLSTM | 87.10% |

The BaggedGRU provided a small but consistent improvement over the single GRU, suggesting that ensembling helped reduce the dependence on a specific data partition. In contrast, the more complex architecture did not improve performance in these experiments.

## Possible improvements

The analysis of the pain survey values suggested similarities between the feature distributions of the no-pain and high-pain classes, which may contribute to their misclassification.

We also tested simple feature extraction approaches, such as computing mean feature values per series and analyzing the pain surveys in the frequency domain. However, these approaches did not improve the performance of the models.

Further improvements would therefore likely require more refined feature engineering and a deeper understanding of the dataset, rather than simply increasing architectural complexity.
