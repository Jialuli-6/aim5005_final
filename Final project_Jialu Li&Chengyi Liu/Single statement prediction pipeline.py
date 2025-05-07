import joblib

#define a prediction function
def predict_sentiment_statement(texts, model, vectorizer):

    if isinstance(texts, str):
        texts = [texts]

    X_input = vectorizer.transform(texts)

    # Predict
    preds = model.predict(X_input)


    # Print results
    for text, pred in zip(texts, preds):
        print(f"Statement: '{text}' → Predicted Sentiment: {pred}")


model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


#########usage example###########################

#replace the intended statement in the ""
predict_sentiment_statement("im soo bored, I don't like this music video", model, vectorizer)