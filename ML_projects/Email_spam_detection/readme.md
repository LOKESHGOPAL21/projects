Below is a README file tailored for the email spam classification project based on the provided code and results. It includes an overview, instructions, and key details about the implementation and findings.

---

# Email Spam Classification Project

## Overview
This project implements a spam email classification system using natural language processing (NLP) and machine learning techniques. The goal is to classify emails as "spam" (1) or "not spam" (0) based on their text content. The dataset used contains 5,728 email samples with two columns: `text` (email content) and `spam` (label). Various machine learning models are trained and evaluated to determine the best-performing classifier.

## Dataset
- **File**: `emails.csv`
- **Shape**: 5,728 rows, 2 columns
- **Columns**:
  - `text`: The raw email text (e.g., subject and body).
  - `spam`: Binary label (1 for spam, 0 for not spam).
- **Sample Data**:
  ```
      text                                              spam
  0   Subject: naturally irresistible your corporate...     1
  1   Subject: the stock trading gunslinger  fanny i...     1
  2   Subject: unbelievable new homes made easy  im ...     1
  3   Subject: 4 color printing special  request add...     1
  4   Subject: do not have money , get software cds ...     1
  ```

## Prerequisites
To run this project, ensure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`

You can install them using pip:
```bash
pip install pandas numpy nltk scikit-learn
```

Additionally, download the required NLTK resources by running the following in Python:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

## Methodology
The project follows these key steps:

1. **Data Preprocessing**:
   - Convert text to lowercase and remove punctuation.
   - Replace multiple spaces with a single space and strip whitespace.
   - Remove English stopwords using NLTK's stopwords list.
   - Tokenize text into words.
   - Perform POS (Part-of-Speech) tagging and lemmatize words using WordNetLemmatizer.

2. **Feature Extraction**:
   - Use TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text into numerical features.
   - Limit to the top 10 features for simplicity (e.g., '2000', 'cc', 'com', 'ect', 'enron', 'hou', 'kaminski', 'pm', 'subject', 'vince').

3. **Model Training**:
   - Split data into 80% training and 20% testing sets.
   - Train four machine learning models:
     - Naive Bayes (`MultinomialNB`)
     - Logistic Regression (`LogisticRegression`)
     - Support Vector Machine (`SVC`)
     - Random Forest (`RandomForestClassifier`)
   - Perform hyperparameter tuning for Logistic Regression (`C`) and Random Forest (`n_estimators`) using `GridSearchCV`.

4. **Evaluation**:
   - Evaluate models using accuracy on the test set.

## Results
The accuracy scores for each model on the test set are as follows:
- **Naive Bayes**: 0.8944
- **Logistic Regression**: 0.8953 (best performer)
- **SVM**: 0.8944
- **Random Forest**: 0.8927

**Key Observation**: Logistic Regression achieved the highest accuracy (0.8953) among the models tested, making it the most suitable classifier for this dataset with the given preprocessing and feature extraction.

## How to Run
1. Ensure the dataset file `emails.csv` is in the same directory as the script.
2. Copy the provided code into a Python file (e.g., `spam_classifier.py`).
3. Run the script:
   ```bash
   python spam_classifier.py
   ```
4. The script will output:
   - Dataset summary (head, shape, columns, info).
   - Selected TF-IDF features.
   - Training progress for each model.
   - Accuracy scores for all models.

## Future Improvements
- Increase the number of TF-IDF features beyond 10 for better representation.
- Experiment with additional preprocessing steps (e.g., handling email headers separately).
- Test more advanced models like neural networks or ensemble methods.
- Address potential class imbalance in the dataset if present.

## License
This project is for educational purposes and does not include a formal license.

---

This README provides a clear and concise guide to understanding and replicating the project. Let me know if you'd like to adjust or expand any section!