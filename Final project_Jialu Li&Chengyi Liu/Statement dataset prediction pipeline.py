import joblib
import pandas as pd

#define a prediction function

def predict_sentiment_dataset(df, text_column, model, vectorizer):

    # Vectorize text data
    X_input = vectorizer.transform(df[text_column].astype(str))

    # Predict sentiments
    predictions = model.predict(X_input)

    # Return dataframe with predictions
    df_result = df.copy()
    df_result['prediction'] = predictions
    return df_result

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


#########usage example###########################

#replace df with the dataset collected
df = pd.read_csv('sentiment_analysis.csv')

#replace df with the dataset name, and replace 'text' with the dataset statement column name
results_df = predict_sentiment_dataset(df, 'text', model, vectorizer)
results_df.to_csv("predicted_sentiments.csv", index=False)
print(results_df)
