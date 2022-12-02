import streamlit as st
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

# chargement et  mise en cache du modèle
@st.experimental_singleton
def load_the_model():
    model = load_model('lstm_model.h5')

    return model

my_model = load_the_model() 

# chargement du tokenizer et mise en cache
@st.experimental_singleton
def load_tokenizer():
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    
    return tokenizer

tk = load_tokenizer()


# nettoyage du tweet
# on applique le même nettoyage que pour les données utilisées pour entraîner le modèle
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

# fonction qui renvoie la prédiction
def predict(tweet):
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

    return prediction


st.title("Prédiction de sentiment")
st.markdown("Nous utilisons un tweet en entrée pour prédire son sentiment positif ou négatif")

# récupération du tweet
st.subheader("Entrez un tweet")
st.write('Le tweet doit être en anglais.')
tweet = st.text_input('',0,280)

# affichage de la prédiction
st.subheader("Sentiment")
if st.button("Prédire le sentiment"):
    sentiment = predict(tweet)
    if sentiment[0][0] >= 0.5:
        st.success('Le sentiment du tweet est positif')
    else:
        st.warning('Le sentiment du tweet est négatif')

