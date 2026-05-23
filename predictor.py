import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics

class Predictor:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model = load_model(os.path.join(model_dir, "model.h5"), compile=False)
        self.model_transformer = load_model(os.path.join(model_dir, "model_transformer.h5"), compile=False)
        
        # The unsupervised scaler was saved as sc.pickle in the code block
        self.sc = pd.read_pickle(os.path.join(model_dir, "sc.pickle"))
        self.sc_y = pd.read_pickle(os.path.join(model_dir, "sc_y.pickle"))
        self.pipe = pd.read_pickle(os.path.join(model_dir, "pipe.pickle"))
        self.ls_features = pd.read_pickle(os.path.join(model_dir, "features.pickle"))
        
        self.tokenizer_title = pd.read_pickle(os.path.join(model_dir, "tokenizer_title.pickle"))
        self.tokenizer_content = pd.read_pickle(os.path.join(model_dir, "tokenizer_content.pickle"))
        
        self.title_params = pd.read_pickle(os.path.join(model_dir, "title_params.pickle"))
        self.content_params = pd.read_pickle(os.path.join(model_dir, "content_params.pickle"))
        
        self.cluster = pd.read_pickle(os.path.join(model_dir, "cluster.pickle"))

    def vectorize_text(self, text_list, tokenizer, max_sequence_length):
        X = tokenizer.texts_to_sequences(text_list)
        X = pad_sequences(X, maxlen=max_sequence_length)
        return X

    def predict(self, title_cleaned, content_cleaned, features_dict):
        title_max_words, title_max_sequence_length = self.title_params
        content_max_words, content_max_sequence_length = self.content_params
        
        X_title = self.vectorize_text([title_cleaned], self.tokenizer_title, title_max_sequence_length)
        X_content = self.vectorize_text([content_cleaned], self.tokenizer_content, content_max_sequence_length)
        
        features_list = []
        for f in self.ls_features:
            features_list.append(features_dict.get(f, 0.0))
            
        X_features_df = pd.DataFrame([features_list], columns=self.ls_features)
        X_features = self.pipe.transform(X_features_df)
        
        pred_scaled = self.model.predict([X_title, X_content, X_features])
        shares_pred = self.sc_y.inverse_transform(pred_scaled)[0][0]
        
        Xt = self.model_transformer.predict([X_title, X_content, X_features])
        Xs = pd.DataFrame(data=self.sc.transform(Xt), columns=[f"x_{i}" for i in range(Xt.shape[1])])
        cl = self.cluster.predict(Xs)[0]
        
        return shares_pred, str(cl)
