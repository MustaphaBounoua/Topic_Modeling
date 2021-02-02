import streamlit as st
import numpy as np
import pandas as pd
import pyLDAvis.gensim
from src.model import LDATopicModeler
import base64
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt

FILE = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/dataset.csv"
@st.cache
def get_data():
    return pd.read_csv(FILE,delimiter='\n',header=None ).rename(columns={0: "text"})



st.cache()
def check_input_method(data_input_mthd):
    """
    function check user input method if uploading or pasting
    Parameters
    ----------
    data_input_mthd: str -> the default displayed text for decision making
    """

    if data_input_mthd =='Copy-Paste text':
        user_input = st.sidebar.text_area("Your text goes here", "")
        df =  pd.DataFrame({"text": str(user_input).split('.')})
    else:
        df = get_data()
    
    return df
    





def vis_word_cloud(lda_model,n_topics):
    """
    funtion visualize topics dominat words using word cloud and export as pdf
    in: n_topics : int -> number of topics
    in: lda_model : trained model 
    
    """

    cloud = WordCloud(stopwords=STOPWORDS,
                background_color='white',
                width=2500,
                height=1800,
                max_words=10,
                colormap='tab10',
                prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    if n_topics % 2 == 0 and n_topics<=6:
        fig, axes = plt.subplots(2,n_topics//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 != 0 and n_topics<=6:
        fig, axes = plt.subplots(2,(n_topics+1)//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 == 0 and n_topics>6:
        fig, axes = plt.subplots(3,n_topics//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
    elif n_topics % 2 != 0 and n_topics>6:
        fig, axes = plt.subplots(3,(n_topics+1)//2,figsize=(16,14), dpi=160, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        if i == n_topics:
            break
        else:
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
            plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    fig.suptitle('Topics Words Cloud', fontsize=22)
        

    return fig
   







data_choice = st.sidebar.radio("Select Data Input Method",('Use a ready dataset','Copy-Paste text'))



st.title('Topic Modeling')

st.markdown("Welcome to this Topic Modeling App. If you want to see more project and cool apps. Please visit my portofolio [page] (http://mustaphabounoua.ml/). ")

st.markdown(" Here’s the full  [code] (https://github.com/MustaphaBounoua/Topic_Modeling) for this app if you would like to get a look.")

st.header("DataSet")

if data_choice == 'Use a ready dataset':
    st.markdown("We will use a public [dataset](http://data.insideairbnb.com/united-states/ny/new-york-city/2019-09-12/visualisations/listings.csv) which is a collection of articles title. We will try to model the different topics to which these articles may be grouped to ")
else :
    st.markdown("We will use the text you copy pasted. We will try to model the different topics to which these articles may be grouped to ")

df = check_input_method(data_choice)

st.dataframe(df)


st.header("Data Cleaning")


model = LDATopicModeler(df)

st.markdown("We will remove clean data : Punctuation, Digits, Stop words. And Then apply lemmatization")



@st.cache
def get_text():
    return model.text

text = get_text() 

def to_sentence(word_list):
    return ' '.join(word for word in word_list)
    
def to_paragraph(sentence_list):
    return '.'.join(sentence for sentence in sentence_list)

display_text = pd.DataFrame([to_sentence(word_list) for word_list in text ])


st.dataframe(display_text)



st.header("Model training")

st.markdown("LDA Model training")



nb_topic = st.sidebar.slider("nb topic",min_value=2,max_value=10,value= 5)
model.nb_topics = nb_topic



model.fit()


st.markdown("Topic modeling results")

nb_words = st.slider("nb words",min_value=5,max_value=20)

topics =  model.generate_topics(10)


topics






st.header("Visualization")


fig = vis_word_cloud(model.lda,nb_topic)

st.pyplot(fig)


st.header("pyldavis gensim")

st.markdown("Using  gensim.pyldavis we can get a more interactive plot")
st.code("dictionary = gensim.corpora.Dictionary.load(data/dictionary.gensim)")
st.code("pyLDAvis.display(pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False))")