### Requirements ###
## Japanese text normalizer
# !pip install neologdn
## Japanese tokenizer "GiNZA"
# !pip install "https://github.com/megagonlabs/ginza/releases/download/latest/ginza-latest.tar.gz"
## Japanese stop words
# !wget http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt
## Japanese fonts for WordCloud
# !wget -qO-  https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip | bsdtar -xvf- 
####################

import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
from matplotlib import pyplot as plt
import neologdn
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Stopwords
stopwords_ja = [i.replace('\n','') for i in open('./Japanese.txt','r').readlines()]
stopwords_en = stopwords.words("english")

# Spacy Models (Japanese / English)
nlp_ja = spacy.load('ja_ginza')
nlp_en = spacy.load('en_core_web_sm')


def ja_preprocess(text):
    lowered = text.lower()
    normalized = neologdn.normalize(lowered)
    tokenized = []
    for i in nlp_ja(normalized):
        word = i.lemma_
        pos = i.pos_
        if pos in ["NOUN", "ADJ", "VERB", "ADV", "PROPN"] and len(word) > 1 and i.text not in stopwords_ja:
            tokenized.append(word)
    preprocessed = " ".join(tokenized)
    return preprocessed


def en_preprocess(text):
    lowered = text.lower()
    tokenized = []
    for i in nlp_en(lowered):
        word = i.lemma_
        pos = i.pos_
        if pos in ["NOUN", "ADJ", "VERB", "ADV", "PROPN"] and len(word) > 1 and i.text not in stopwords_en:
            tokenized.append(word)
    return ' '.join(tokenized)


def bow_features(text, _max_features=10, _max_ngrams=2):
    bow = CountVectorizer(max_features=_max_features, ngram_range=(1,_max_ngrams))
    vec = bow.fit_transform(text).toarray()
    bow_df = pd.DataFrame(vec, columns=["BOW_" + n for n in bow.get_feature_names()])
    return bow_df


def tfidf_features(text, _max_features=10, _max_ngrams=2):
    tfidf = TfidfVectorizer(max_features=_max_features, use_idf=True, ngram_range=(1,_max_ngrams))
    vec = tfidf.fit_transform(text).toarray()
    tfidf_df = pd.DataFrame(vec, columns=["TFIDF_" + n for n in tfidf.get_feature_names()])
    return tfidf_df


def ngram_count(text, ngram=1, common_num=30):
    words =  text.sum().split()
    dic = nltk.FreqDist(nltk.ngrams(words, ngram)).most_common(common_num)
    ngram_df = pd.DataFrame(dic, columns=['ngram','count'])
    ngram_df.index = [' '.join(i) for i in ngram_df.ngram]
    ngram_df.drop('ngram',axis=1, inplace=True)
    return ngram_df


def wordCloud(text, _max_words=300, _font_path=None, _output_file='wordCloud.png'):
    wordcloud = WordCloud(background_color='white', width=800, height=600, min_font_size=10, max_words=_max_words, \
                          collocations=False, min_word_length=2, font_path=_font_path)
    wordcloud.generate(text.sum())
    wordcloud.to_file(_output_file)
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud)
    plt.show()
 

if __name__ == '__main__':
    # sample dataset : The Kyoto Free Translation Task (KFTT)
    # http://www.phontron.com/kftt/index.html
    base_dir = '../input/the-kyoto-free-translation-task-kftt/'
    ja = [i.rstrip('\n') for i in open(base_dir + 'kyoto-train.ja', 'r').readlines()]
    en = [i.rstrip('\n') for i in open(base_dir + 'kyoto-train.en', 'r').readlines()]
    df = pd.DataFrame({"ja":ja, "en":en}).iloc[0:3000,:]
    
    # Japanese PreProcessing
    df['ja_preprocessed'] = df['ja'].apply(lambda x: ja_preprocess(x))
    # English PreProcessing
    df['en_preprocessed'] = df['en'].apply(lambda x: en_preprocess(x))
    
    # English tfidf / bow features
    en_bow = bow_features(df['en_preprocessed'], _max_features=10, _max_ngrams=1)
    en_tfidf = tfidf_features(df['en_preprocessed'], _max_features=10, _max_ngrams=1)
    
    # English Ngram Count
    en_ngram = ngram_count(df['en_preprocessed'], ngram=1, common_num=30)
    # Ngram Count Bar Plot
    en_ngram_graph = hv.Bars(en_ngram[::-1]) \
        .opts(opts.Bars(title="Ngram Count", color="red", xlabel="Unigrams", ylabel="Count", width=400, height=600, show_grid=True, invert_axes=True))
    hv.save(en_ngram_graph, 'en_graph.html')

    # Japanese WordCloud
    wordCloud(df['ja_preprocessed'], _font_path='NotoSansCJKjp-Regular.otf', _output_file='wordCloud_ja.png')
    # English WordCloud
    wordCloud(df['en_preprocessed'], _output_file='wordCloud_en.png')
