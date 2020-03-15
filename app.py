
# import libraries
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
import pickle

from model import NLPModel

app = Flask(__name__)
api = Api(app)

# Create a new model object
model = NLPModel()

# Load trained classifier
clf_path = 'models/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

# load trained vectorizer
vec_path = 'models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

# create argument parser
parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    def get(self):
        # use parser and find user's query
        args = parser.parse_args()
        user_query = args['query']

        # vectorize the user's query and make a prediction
        uq_vectorized = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)

        # Output 'Negative' or 'Positive' along with the score
        if prediction == 0:
            pred_text = 'Negative'
        else:
            pred_text = 'Positive'

        # Round the predict proba value and assign it to the confidence variable
        confidence = round(pred_proba[0], 3)

        # Create JSON object
        output = {'prediction': pred_text, 'confidence': confidence}

        return output

# Route URL to resource
api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=True)