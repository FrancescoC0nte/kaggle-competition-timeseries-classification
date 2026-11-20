# Timeseries Classification of Pain Levels from simulated sensor data

Project for the AN2DL First Challenge — Ottima ANNata

## What this is about

This project focuses on time series classification using deep learning techniques and Recurrent Neural Networks. The goal is to classify sensor time series into three pain levels: **no pain**, **low pain**, and **high pain**. We was required to addresse several challenges; in particular, class imbalance and the sensitivity of model performance to the training/validation split.

The dataset consists of 661 time series with 160 timestamps and 38 heterogeneous features. Each series is associated with one of three classes: no pain, low pain, or high pain. During the competition, only half of the test set was made available for evaluation, while the final ranking was determined using the complete test set. 
This allowed us to assess our models on a partial test set throughout the competition and to account for overfitted teams' models.

Our work explores different neural architectures, starting from recurrent models such as GRUs and LSTMs and progressing to models combining convolutional and attention mechanisms. The main objective was to identify an architecture with an outstanding performance, while maintaining robust generalization across different data partitions.

The complete description of the methodology, experiments, and results is available in the [project report](Project%20Report.pdf). This README provides a concise overview of the implementation choices and the main findings.

## Repository structure

The repository is organized as follows:

```text
.
├── Code/                        # Model implementations and analysis scripts
    ├── AttentionCNNLSTM.ipynb
    ├── BaggedGRU.ipynb
    └── GRU.ipynb
├── Project Report.pdf           # Report for the AN2DL First Challenge
└── README.md
```

The source code is available in the [`Code/`](Code/) directory.

## Challenges and Results

**Class imbalance** was the main challenge for our task. The three pain classes are not equally represented. We therefore used class weights to penalize errors on underrepresented classes. In a subsequent experiment, we replaced weighted Cross-Entropy with Focal Loss to further address the imbalance.

We initially compared GRU and LSTM models using different hyperparameters, yet the best model turned out to be a **GRU with one layer and 256 neurons**. 
This highlights the fact that the best-performing model is not necessarily the deeper and more complex one, but rather the one that is more carefully tailored to the complexity of the task.

The **key technique** that allowed us to improve the model's performance and generalization capabilities was **Bagging**. That is, training the same GRU models on different data splits and then combining their predictions through voting. This method is generally effective for learners characterized by a high risk of overfitting and computationally expensive training procedures, as their performance can depend substantially on each particular training-validation split.

Therefore, the best-performing model was the BaggedGRU, which achieved an F1 score of **95.70%** on the test set. The single GRU achieved **95.15%**, while the more complex AttentionCNNLSTM model achieved **87.10%**. Moreover, the final F1 score of the model obtained through bagging on the full test set turned out to be **0.9618**, placing our team **17th out of 193 teams** and confirming the effectiveness of this technique in improving model generalization.
