
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')
def cleaning(df,colomns):
    stop_words = set(stopwords.words('french'))
    stemmer = SnowballStemmer('french')
    df.dropna()
    # Suppression de tout ce qui n'est pas alphabet français
    df[colomns] = df[colomns].str.replace('[^a-zA-ZÀ-ÿ]', ' ', regex=True)
    # Suppression des espaces multiples en début et fin de lignes
    df[colomns] = df[colomns].str.replace('^\s+\s+$', ' ', regex=True)
    # Suppression des espaces multiples dans le texte
    df[colomns] = df[colomns].str.replace('\s+', ' ', regex=True)
    # Suppression des caractères html
    df[colomns] = df[colomns].str.replace('&\w+', '', regex=True)
    # Transformation de majuscule en minuscule
    df[colomns] = df[colomns].str.lower()
    # Normalisation des caractères accentués
    #df['infos'] = df['infos'].apply(lambda x: unidecode(x))
    # Suppression des stopwords en français et lemmatisation
    df[colomns] = df[colomns].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x) if word.lower() not in stop_words and len(word) > 2]))
    return df