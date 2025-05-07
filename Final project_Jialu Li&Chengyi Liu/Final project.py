import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score



# Import data
data = pd.read_csv('Combined Data.csv', index_col=0)
data.head()

#data cleaning
data = data.drop_duplicates(subset='statement', keep='first')       #drop duplicates
data = data.dropna(subset=['statement'])                            #drop nas

print('The total number of samples is:\n', data['status'].count())
print(data['status'].value_counts())

# Text preprocessing
def preprocess_text(text):
    text = str(text)
    text = text.lower() 
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

data['clean_statements'] = data['statement'].apply(preprocess_text)
data.head()

# Split dataset to train and test 
X_train, X_test, y_train, y_test = train_test_split(
    data['clean_statements'], 
    data['status'], 
    test_size=0.2, 
    random_state=42,
    stratify=data['status']
)

############################traditional machine learning methods##############

# Vectorize features
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(X_train_tfidf.shape)


###################Setting hyperparameter tuning and training###########################

#Random Forest
rf = RandomForestClassifier(random_state=42)  
params_r= {'n_estimators': [50, 100, 200],                  #The number of trees in the forest.
           'max_depth': [None, 10, 20],                     #The maximum depth of the tree
           'criterion': ['gini', 'entropy', 'log_loss']     #The function to measure the quality of a split
          }

#Logistic Regression
lr_model = LogisticRegression()
params_l = {'C': [0.1, 1, 10],                              #Inverse of regularization strength
            'solver': ['lbfgs', 'liblinear', 'saga'],       #Algorithm to use in the optimization problem
            'max_iter': [100, 500, 1000]                    #Maximum number of iterations taken for the solvers to converge
           }

#SVM
svm = SVC()
params_s = {'C':[0.1, 1, 10],                               #Inverse of regularization strength
            'kernel':['linear', 'poly', 'sigmoid'],         #Kernel algorithm  
            'gamma':['scale', 'auto']                       #Kernel coefficient
           }

#Model evaluate scores
scorers = {'accuracy': 'accuracy',
           'f1_macro': make_scorer(f1_score, average='macro'),
           'f1_weighted': make_scorer(f1_score, average='weighted')
          }



#Random forest training 
random_search_rf = RandomizedSearchCV(
    estimator=rf,
    param_distributions=params_r,
    n_iter=20,  
    cv=5,
    n_jobs=-1,
    scoring=scorers,
    refit='accuracy',
    random_state=42
)
# Fit RandomizedSearchCV to the training data
random_search_rf.fit(X_train_tfidf, y_train)
print("Best rf hyperparameters:", random_search_rf.best_params_)

# Best model performance on the test set
best_rf = random_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test_tfidf)

print("Classification Report for Random Forest:\n", classification_report(y_test, y_pred_rf))


#Logistic regression training
random_search_lr = RandomizedSearchCV(
    estimator=lr_model, 
    param_distributions=params_l,
    n_iter=20,
    cv=5, 
    n_jobs=-1, 
    scoring=scorers, 
    refit='accuracy',
    random_state=42
)

# Fit RandomizedSearchCV to the training data
random_search_lr.fit(X_train_tfidf, y_train)
print("Best lr hyperparameters:", random_search_lr.best_params_)

# Best model performance on the test set
best_lr = random_search_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_tfidf)

print("Classification Report for Logistic Regression:\n",classification_report(y_test, y_pred_lr))



#SVM training 
random_search_svm = RandomizedSearchCV(
    estimator=svm, 
    param_distributions=params_s,
    n_iter=20,
    cv=5, 
    n_jobs=-1, 
    scoring=scorers, 
    refit='accuracy',
    random_state=42
)

# Fit RandomizedSearchCV to the training data
random_search_svm.fit(X_train_tfidf, y_train)
print("Best svm hyperparameters:", random_search_svm.best_params_)

# Best model performance on the test set
best_svm = random_search_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test_tfidf)

print("Classification Report for SVM:\n",classification_report(y_test, y_pred_svm))


############################deep learning method lstm############################

# Encode the labels
label = {label: idx for idx, label in enumerate(data['status'].unique())}
data['encoded_status'] = data['status'].map(label)
print("Label Mapping:", label)


# Tokenize features
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data['clean_statements'])

X1 = tokenizer.texts_to_sequences(data['clean_statements'])
X1 = pad_sequences(X1, maxlen=max_len)

y1 = to_categorical(data['encoded_status'])

# Split data
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1
)


# LSTM model
lstm = Sequential()
lstm.add(Embedding(max_words, 128, input_length=max_len))
lstm.add(LSTM(64, return_sequences=True))
lstm.add(Dropout(0.5))
lstm.add(LSTM(32))
lstm.add(Dropout(0.5))
lstm.add(Dense(y1.shape[1], activation='softmax'))

lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the LSTM model
lstm.fit(X1_train, y1_train, epochs=10, batch_size=64, validation_split=0.1)
y1_pred = lstm.predict(X1_test)

y_pred_labels = np.argmax(y1_pred, axis=1)
y_true_labels = np.argmax(y1_test, axis=1)

print("Classification report for LSTM:\n", classification_report(y_true_labels, y_pred_labels))
