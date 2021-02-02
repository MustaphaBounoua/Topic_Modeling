from abc import ABC, abstractmethod 
from .preprocess import clean_sentence
from gensim import corpora, models, similarities
import pyLDAvis.gensim
import pandas as pd
import numpy as np


NB_TOPICS = 5


class TopicModeler(ABC):
    
    
    def __init__(self,data,nb_topics):
        
        self.data = data
        self.nb_topics = nb_topics
        super().__init__()
    
    @abstractmethod
    def clean_data(self):
        pass
    @abstractmethod
    def create_corpus_from_text(self):
        pass
    @abstractmethod
    def vectorize_text(self):
        pass
    
    @abstractmethod
    def train_clustering_model(self):
        pass
   
    @abstractmethod
    def generate_topics(self):
        pass
    
    


class LDATopicModeler(TopicModeler):
    
    def __init__(self,data,nb_topics=NB_TOPICS):
        super().__init__(data,nb_topics)
        self.clean_data()
    
    
    def clean_data(cls):
        text = cls.data["text"].to_numpy()
        cls.text = [clean_sentence(sentence) for sentence in text ]
    
    
    
    def create_corpus_from_text(cls):
        
        cls.dictionary = corpora.Dictionary(cls.text)
        cls.corpus = [cls.dictionary.doc2bow(word) for word in cls.text]
    
    
    
    def vectorize_text(cls):
        tfidf = models.TfidfModel(cls.corpus)
        cls.corpus_tfidf = tfidf[cls.corpus]
    
    
    def train_clustering_model(cls):
        cls.lda = models.LdaModel(cls.corpus_tfidf, id2word = cls.dictionary, num_topics = cls.nb_topics)
    
    def generate_topics(cls,nb_words):
        cls.corpus_lda = cls.lda[cls.corpus_tfidf]
        return cls.lda.show_topics(cls.nb_topics, nb_words)
    

    def fit(cls):
        cls.create_corpus_from_text()
        cls.vectorize_text()
        cls.train_clustering_model()
    
    
    
    def display_gensim_topics_plot(cls):
       lda_display = pyLDAvis.gensim.prepare(cls.lda, cls.corpus, cls.dictionary, sort_topics=False)
       pyLDAvis.display(lda_display)