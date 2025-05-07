import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import re
import joblib

#import data
data = pd.read_csv('Combined Data.csv', index_col=0)
data.head()

#data cleaning
data = data.drop_duplicates(subset='statement', keep='first')  #drop duplicates
data = data.dropna(subset=['statement']) #drop nas

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


# ############################traditional machine learning methods with the best set of parameters##############

# Vectorize features
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(X_train_tfidf.shape)


#best random forest
rf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None)
rf.fit(X_train_tfidf, y_train)

y_pred_rf = rf.predict(X_test_tfidf)
print("Classification Report for Random Forest:\n", classification_report(y_test, y_pred_rf))


#best logistic regression
lr_model = LogisticRegression(max_iter=500, solver='liblinear', C=10)
lr_model.fit(X_train_tfidf, y_train)

y_pred_lr = lr_model.predict(X_test_tfidf)
print("Classification Report for Logistic Regression:\n",classification_report(y_test, y_pred_lr))


#best SVM
svm = SVC(C=1.0, kernel='linear', gamma= 'scale')
svm.fit(X_train_tfidf, y_train)

y_pred_svm = svm.predict(X_test_tfidf)
print("Classification Report for SVM:\n", classification_report(y_test, y_pred_svm))


#############################################deep learning method lstm########################################

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

joblib.dump(lr_model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
