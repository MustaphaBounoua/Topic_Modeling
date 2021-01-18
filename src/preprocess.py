import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import re

STOP_WORDS = set(nltk.corpus.stopwords.words('english'))


def get_lemma(word):
    """[summary]

    Args:
        word ([str]): [a word]

    Returns:
        [ lema word after lemmatization]: [Lemmatization is the process of grouping together 
        the different inflected forms of a word so they can be analysed as a single item. 
        Lemmatization is similar to stemming but it brings context to the words. So it links words with similar meaning to one word]
    """
    lemmatizer = WordNetLemmatizer() 
    return lemmatizer.lemmatize(word)



def remove_punctuation(text):
    """[summary]

    Args:
        text ([str]): [Sentence, paragraph or a word  in one str]

    Returns:
        [arr(str)]: [list of words without punctuation]
    """
    new_words = []
    for word in text.split():
        new_word = re.sub(r'[^\w\s]', '', (word))
        if new_word != '':
            new_words.append(new_word)
    return new_words



def clean_sentence(text): 
    """[summary]

    Args:
        text ([type]): [description]

    Returns:
        [type]: [description]
    """
    text = re.sub('@[^\s]+', '', text) ## remove @mentions
    text = re.sub('((www.[^\s]+)|(https?://[^\s]+))', '', text) ##remove urls
    text = re.sub(r'\d+', '',  text) ## remove digits
    text = text.lower() ## lower the text
    tokens = remove_punctuation(text) ## remove punctuation
    ## lemma of words with lenght > 2
    tokens = [get_lemma(token) for token in tokens if ( token not in STOP_WORDS and len(token) > 2)]
    return tokens



def prepare_for_lda(df):
    """[Get a clean array for lda]

    Args:
        df ([DataFrame]): [Data Frame]

    Returns:
        [Arr]: [Array of clean text]
    """
    data = df["text"].to_numpy()
    data = [clean_sentence(sentence) for sentence in data ]
    #return np.concatenate( data, axis=0 )
    return data