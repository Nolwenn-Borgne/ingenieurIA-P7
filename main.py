# -*- coding: utf-8 -*-

# 1. Library imports
#import uvicorn ##ASGI
from fastapi import FastAPI
from Tweets import Tweet
from keras.models import load_model 
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import re
import json
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# 2. Create the app object
app = FastAPI()

# 3. Chargement du modèle et du tokenizer
def load_the_model():
    model = load_model('lstm_model.h5')

    return model

my_model = load_the_model() 

def load_tokenizer():
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    
    return tokenizer

tk = load_tokenizer()

# 4. Nettoyage du tweet
def clean_tweet(tweet):
    #ponctuations
    ponctuations = list(string.punctuation)

    #lemmatisation
    lem = WordNetLemmatizer()

    #charger les stopwords
    mots_vides = stopwords.words('english')

    #harmonisation de la casse
    temp = tweet.lower()
    #retirer les contractions en anglais
    temp = re.sub("'", "", temp)
    #retrait des @ (mentions)
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    #retrait des # (hashtags)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    #retrait des liens web (http et https)
    temp = re.sub(r'http\S+', '', temp)
    #retrait des ponctuations
    temp = "".join([char for char in list(temp) if not (char in ponctuations)])
    #retrait des nombres   
    temp = re.sub("[0-9]","", temp)
    #tokenisation
    temp = word_tokenize(temp)
    #lemmatisation des termes
    temp = [lem.lemmatize(mot) for mot in temp]
    #retrait des stopwords
    temp = [mot for mot in temp if not mot in mots_vides]
    #retirer les tokens de moins de 3 caractères
    temp = [mot for mot in temp if len(mot) >= 3]    
    #reformer la chaîne
    temp = " ".join(mot for mot in temp)

    return temp

# 5. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 6. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome ': f'{name}'}

# 7. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted sentiment with the confidence
@app.post('/predict')
def predict(data:Tweet):
    data = data.dict()
    tweet=data['text']

   # nettoyage
    tweet_clean = clean_tweet(tweet)

    # #transformation en séquence
    my_seq = tk.texts_to_sequences([tweet_clean])

    max_length = 21
    marge_length = 5
    #puis en pad
    my_pad = pad_sequences(my_seq,maxlen=max_length + marge_length,padding='post')

    # prédiction
    prediction = my_model.predict(my_pad)
    if prediction[0][0] >= 0.5:
        prediction = "Positif"
    else:
         prediction = "Négatif"

    return {
        'prediction': prediction
    }


# 8. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
#if __name__ == '__main__':
#    uvicorn.run(app, host='0.0.0.0', port=8000)
#uvicorn main:app --reload
