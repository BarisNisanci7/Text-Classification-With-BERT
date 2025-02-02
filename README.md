# Multi-Class Turkish Text Classification with BERT

## 1. Introduction
This project aims to classify newspaper articles by predicting the author’s identity from a set of 30 possible authors. Each author contributed 50 articles, resulting in a total of 1500 articles. We approach this multi-class text classification problem using a BERT-based model specialized for the Turkish language.

We employ a 5-fold cross-validation strategy to systematically evaluate our model’s performance across the entire dataset and ensure that each article is used for both training and testing. By the end of training, we generate a comprehensive classification report detailing precision, recall, and F1-score for each of the 30 authors, as well as overall averages.

---

## 2. Dataset and Preprocessing

### 2.1 Dataset Description
- **Source**: A folder named `makaleler-yazarlar` containing 30 subdirectories, each corresponding to one author.  
- **Structure**: Each author’s subdirectory has 50 text files (articles).  
- **Total Articles**: 1500 (30 authors × 50 articles each).

### 2.2 Preprocessing Steps
1. **Reading Files**  
   We loop through each author’s folder and read the text files using `utf-8` encoding. In case of a `UnicodeDecodeError`, we attempt a fallback encoding (`ISO-8859-9`).

2. **Basic Text Cleaning**  
   - Replace multiple spaces with a single space.  
   - Strip leading/trailing whitespaces.

3. **Label Encoding**  
   - Convert each author’s name into a numeric label (0 to 29) using `LabelEncoder`.

After these steps, the dataset is organized as `(author_label, article_text)` pairs, ready for tokenization.

---

## 3. Model and Tokenization

### 3.1 BERT Model
- **Model Name**: `dbmdz/bert-base-turkish-cased`  
- **Reason**: It is pretrained on Turkish corpora and preserves letter casing, which is often beneficial for Turkish text nuances.

We load the model using the `BertForSequenceClassification` class from Hugging Face’s Transformers library, which adds a classification head on top of the standard BERT architecture.

### 3.2 Tokenization
We apply the `BertTokenizer` from the same pre-trained model to convert each article into `input_ids` and `attention_mask`.

- **Padding and Truncation**: `padding=True` and `truncation=True` to ensure all sequences align to the same length.  
- **Max Sequence Length**: 256. Any sequence longer than 256 tokens is truncated.

Once tokenized, we have two main tensors for each article:
- `input_ids_all`: Numerical token IDs of shape `(num_articles, 256)`.  
- `attention_mask_all`: Mask indicating which tokens are padding and which are actual input `(num_articles, 256)`.

---

## 4. Training Procedure

### 4.1 5-Fold Cross Validation
We use `StratifiedKFold(n_splits=5)` from scikit-learn, which:
- Splits the data into 5 distinct folds, keeping the class distribution balanced.
- Iterates 5 times; each time, one fold becomes the validation set, and the remaining 4 folds are used for training.

### 4.2 Model Hyperparameters
- **Optimizer**: AdamW (Transformers version)  
- **Learning Rate**: `2e-5`  
- **Batch Size**: 16  
- **Number of Epochs**: 4 (per fold)  
- **Loss Function**: `CrossEntropyLoss()` (PyTorch)  
- **Device**: GPU if available (`cuda`), otherwise CPU (`cpu`)

### 4.3 Training and Validation
For each fold:

1. **Dataset Splitting**  
   Separate `input_ids_all` and `attention_mask_all` into training (80% of fold) and validation (20% of fold) subsets, along with their respective labels.

2. **Fine-Tuning**  
   Load a fresh instance of the Turkish BERT model and fine-tune it on the training set for 4 epochs.

3. **Evaluation**  
   After each epoch, measure validation accuracy and loss. Finally, compute predictions on the entire validation set, recording them for the fold-level classification report.

---

## 5. Results and Performance
After completing all 5 folds, we combine predictions and true labels from each fold, yielding a 5-fold comprehensive evaluation of the entire dataset. The main metric used is the classification report, detailing:

- **Precision**: TP / (TP + FP)  
- **Recall**: TP / (TP + FN)  
- **F1 Score**: Harmonic mean of precision and recall  
- **Support**: Number of samples for each author

A summarized snippet of the results (macro-averaged) often hovers around:
- **Accuracy (Approx.)**: ~0.82  
- **Macro F1-Score**: ~0.81 to 0.82  
- **Weighted F1-Score**: ~0.81 to 0.82  

Individual authors’ F1-scores vary, sometimes ranging from ~0.60 to 1.00, reflecting differences in writing style, text length, or similarity among certain authors’ vocabularies.

---

## 6. File Outputs
- **Fold-wise Reports**: Printed after each fold to analyze performance on the validation subset.  
- **Final Combined Report**: Saved to `bert_5fold_cv_report.txt`, containing:  
  - Precision, Recall, F1 for each of the 30 authors  
  - Overall (macro / weighted) averages  
  - Total support across all folds (1500 articles)

---

## 7. Conclusions and Future Work

### 7.1 Key Observations
1. **Feasibility**: Training a Turkish BERT model with 5-fold CV on 1500 articles is feasible within moderate GPU resources.  
2. **Performance**: Achieving around ~81% macro F1-score for 30-class author classification is quite strong, indicating that author writing styles are distinguishable with contextual embeddings.  
3. **Data Distribution**: Each class is well balanced (50 articles per author), and `StratifiedKFold` leverages this balance effectively.

### 7.2 Limitations & Potential Enhancements
1. **Preprocessing**: Additional text normalization (e.g., removing certain punctuation patterns, applying morphological analysis, or advanced stopword handling) could yield improvements.  
2. **Hyperparameter Tuning**: Experimenting with different batch sizes, learning rates (e.g., `3e-5`, `5e-5`), or using more training epochs might further improve performance.  
3. **Early Stopping**: Incorporating an early stopping strategy based on validation loss could prevent overfitting in some folds.

---

## 8. References and Tools
1. **Hugging Face Transformers**  
   [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)  
   Used for the BERT tokenizer, model instantiation, and AdamW optimizer.

2. **scikit-learn**  
   [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)  
   Used for `StratifiedKFold` cross-validation, `LabelEncoder`, and performance metrics.

3. **PyTorch**  
   [https://pytorch.org/](https://pytorch.org/)  
   Used as the deep learning framework for training and data handling.

---

## 9. Project Structure
1. **Main Script / Notebook**  
   - **Imports & Data Loading**  
   - **Preprocessing**  
   - **Tokenization & Label Encoding**  
   - **5-Fold Cross Validation and Training**  
   - **Evaluation & Report Generation**

2. **Data Folder**  
   - 30 subdirectories, each containing 50 `.txt` files.

3. **Output Files**  
   - `bert_5fold_cv_report.txt`: Final classification report over all 5 folds.

---

## 10. Final Notes
This report outlines the methodology, implementation, and results of our multi-class Turkish text classification system using a BERT-based approach. The project demonstrates how contextual embeddings and cross-validation can yield strong performance for identifying author-specific linguistic patterns, achieving an average F1-score of approximately 0.82 across 30 distinct authors. Further refinements, such as extended preprocessing or hyperparameter tuning, can be explored to push performance even higher.
