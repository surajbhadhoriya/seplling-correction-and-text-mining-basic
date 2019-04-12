# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:38:10 2019

@author: SURAJ BHADHORIYA
"""
#load libraries
import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer

#load dataset
with open('C:/Users/SURAJ BHADHORIYA/Desktop/New folder/nltk/moby.txt','r') as f:
    moby_raw=f.read()
#word tokenizer    
moby_token=nltk.word_tokenize(moby_raw)  
text1=nltk.Text(moby_token)
moby_series=pd.Series(moby_token)
#length of words
def len_moby_token():
    length=len(moby_token)
    return length
leng=len_moby_token()
print("moby has {:,} token".format(leng))
#find unique words
def unique_wrd():
    length=len(set(moby_token))
    return length
unique=unique_wrd()
print("moby has {:,}  unique token".format(unique))
#lemmatizing
def lemmatizer_moby_token():
    lemmatizer=WordNetLemmatizer()
    length=len(set(lemmatizer.lemmatize(w,'v') for w in text1))
    return length
lem=lemmatizer_moby_token()
print("moby has {:,} token".format(lem))

#lexical diversity of document
def lexical_diversity(token):
    lex=(len(set(token))/len(token))
    return lex
lexical=lexical_diversity(moby_token)
print("moby has {:,}  lexical diversity".format(lexical))

moby_fre= nltk.FreqDist(moby_token)
#frequency distribution for particular word whale
def wrd_moby_token():
   whale=moby_fre["whale"]+moby_fre["whale"]
   x=100*(whale/leng)
   return x
per=wrd_moby_token()
print("moby has whale % {:,} ".format(per))
#find most common words
def most_common_moby_token():
    return moby_fre.most_common(20)
most_common=most_common_moby_token()
print(most_common)

moby_frame=pd.DataFrame(moby_fre.most_common(),columns=["token","frequency"])
#make data frame and find the words which has frequency greater than 150
def le_moby_token():
    frame=moby_frame[(moby_frame.frequency>150 ) & (moby_frame.token.str.len()>5)]
    return sorted(frame.token)
frame1=le_moby_token()
print(frame1)
#find logest word
def longest_moby_token():
    length=max(moby_frame.token.str.len())
    longest=moby_frame.token.str.extractall("(?P<long>.{{{}}})".format(length))
    return (longest.long.iloc[0],length)
print(longest_moby_token())

moby_word = moby_frame[moby_frame.token.str.isalpha()]
print(moby_word)
common=moby_word[moby_word.frequency>2000]
print(common)
print(list(zip(common.frequency,common.token)))
#avrage of words in document
def avg_moby_token():
    sent=nltk.sent_tokenize(moby_raw)
    count=(len(nltk.word_tokenize(sent1)) for sent1 in sent)
    return sum(count)/float(len(sent))
avg1=avg_moby_token()
print(avg1)
#part of speech
def pos_moby():
    tags=nltk.pos_tag(moby_word.token)
    fre=nltk.FreqDist([tag for (word,tag) in tags])
    return fre.most_common(5)
pos=pos_moby()
print("pos {}".format(pos))

#spelling correction
from nltk.corpus import words
from nltk.metrics.distance import (
        edit_distance,
        jaccard_distance,)
from nltk.util import ngrams
correct_spellings=words.words()
spellings_series=pd.Series(correct_spellings)

def jaccard(entries, gram_number):
    outcomes=[]
    for entry in entries:
        spellings=spellings_series[spellings_series.str.startswith(entry[0])]
        distances=((jaccard_distance(set(ngrams(entry,gram_number)),set(ngrams(word,gram_number))),word)
                   for word in spellings)
        closest=min(distances)
        outcomes.append(closest[1])
        return outcomes
def result(entries=['cormulent','incendence','validrate']):
    return jaccard(entries,3)
print(result())

def result1(entries=['cormulent','incedence','validare']):
    return jaccard(entries,4)
print(result1())

def result2(entries=['cormulent','incedence','validare']):
    outcomes=[]
    for entry in entries:
        distances=((edit_distance(entry,word),word)
                  for word in correct_spellings)
        closest=min(distances)
        outcomes.append(closest[1])
        return outcomes
print(result2())   
  
