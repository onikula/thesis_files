import nltk
import os
import statistics

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, sent_tokenize, pos_tag, RegexpParser

from nltk.tree import Tree

import argparse
from collections import Counter
from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))
from string import punctuation
from nrclex import NRCLex
import re

from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.parse import stanford
from nltk.parse import CoreNLPParser
os.environ['STANFORD_PARSER'] = "jar/stanford-parser.jar" 
os.environ['STANFORD_MODELS'] = "jar/stanford-parser-4.2.0-models.jar"

from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tree import Tree


from readability import Readability
from lexical_diversity import lex_div as ld
import syntok.segmenter as segmenter

import numpy as np 
import pandas as pd 
from wordfreq import word_frequency


parset = CoreNLPParser('http://localhost:9000')

parser = argparse.ArgumentParser()

parser.add_argument('filename', help = "Path to the file that will  be POS tagged",)

args = parser.parse_args()

text_file = args.filename

with open(text_file, 'r') as fp:
    file = fp.read()

texts = file.split('\n\n')

tokens_text = []
ttr_text = []
num_sentences = []
words_sentences = []
f_scores = []
g_scores = []
s_scores = []
punctuations = []
quotations = []
caps = []
caps2 = []
Named_News = []
Named_ENT = []
Abbr = []
stoppwords = []
POS_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH','VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
POS_list_dict = { t: [] for t in POS_list}
med_depth = []
med_vp_depth = []
med_np_depth = []
emotion_list = ['anger', 'fear', 'disgust','sadness', 'negative', 'anticipation', 'trust', 'surprise', 'joy', 'positive']
emotion_list_dict = {e: [] for e in emotion_list}
negatives = ['anger', 'fear', 'disgust','sadness', 'negative']
positives = ['anticipation', 'trust', 'surprise', 'joy', 'positive']
top_emotion = []
w_frequency = []
w_frequency_lc = []
anger = []
fear = []
disgust = []
sadness = []
negative = []
anticipation = []
trust = []
surprise = []
joy = []
positive = []
neg_intensity = []
pos_intensity = []


word_count = []
avg_w_len = []


def main(text):
    text = text
    sentences = sent_tokenize(text)
    tokenized = '\n\n'.join(
    '\n'.join(' '.join(token.value for token in sentence)
       for sentence in paragraph)
    for paragraph in segmenter.analyze(text))
    tokens = word_tokenize(text)
    tokens_text.append(len(tokens))
    words2 = tokens
    for token in words2:
        if token in punctuation:
            words2.remove(token)
    word_count.append(len(words2))
    lemmas = ld.flemmatize(text)
    #print("Number of tokens article:" + str(len(tokens)))
    #print("TTR of article:" + str(ld.ttr(lemmas)))
    ttr_text.append(ld.ttr(lemmas))
    #print("Number of tokens article:" + str(len(tokens)))
    #print("TTR of article:" + str(ld.ttr(lemmas)))

    # The parser needs to be initialised in order for the code to work.
    # Navigate to the file location where the Core NLP parser is and run this command
    # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &
    def syntax_depth():
        depths = []
        VP_depths = []
        NP_depths = []
        for sentence in sentences:
            bad_chars = ["%"]
            for i in bad_chars:
                sentence = sentence.replace(i, "")
            #print(sentence)
            parses = parset.parse(sentence.split())
            p = list(parses)
            #t = Tree.fromstring(p[0])
            #print(p[0].height())
            # for i in p:
            #     if p[i] == "S"
            #p[0].pretty_print()
            root = p[0]
            VP_subdepths = []
            NP_subdepths = []
            for phrase in root.subtrees(lambda t: t.label() == "S"):
                #print(f"{phrase.label()}, {phrase.height()}")
                for fraasi in phrase.subtrees(lambda t: t.label() == "VP"):
                    VP_subdepths.append(fraasi.height())
                for fraasi in phrase.subtrees(lambda t: t.label() == "NP"):
                    NP_subdepths.append(fraasi.height())
                if len(VP_subdepths) > 0:
                    VP_depths.append(max(VP_subdepths))
                if len(NP_subdepths) > 0:
                    NP_depths.append(max(NP_subdepths))    
            depths.append(p[0].height())
        if len(depths) < 1:
            med_depth.append("NA")
        else:
            medians = statistics.median(depths)
            med_depth.append(medians)
        if len(VP_depths) < 1:
            med_vp_depth.append("NA")
        else:
            medianvp = statistics.median(VP_depths)
            med_vp_depth.append(medianvp)
        if len(NP_depths) < 1:
            med_np_depth.append("NA")
        else:
            mediannp = statistics.median(NP_depths)
            med_np_depth.append(mediannp)
        #print("Median syntax tree depth: " + str(med_depth))

    def words_per_sentence():
        summa = 0
        #print("Sentences in article: " + str(len(sentences)))
        num_sentences.append(len(sentences))
        for sentence in sentences:
            words = []
            words = word_tokenize(sentence)
            for word in words:
                if word in punctuation:
                    words.remove(word)
            summa += len(words)
        #print(f"Average amount of words per sentence: {(summa/len(sentences)):.2f}")
        if len(sentences) > 0:
            words_sentences.append(summa/len(sentences))

    def word_length():
        text = words2
        length = len(text)
        summa = 0
        for i in text:
            summa += len(i)
        avg_w_len.append(summa/length)        
        
    def readable():  
        r = Readability(tokenized)
        if len(words2) < 100:
            f_scores.append("NA")
            g_scores.append("NA")
        else:
            fk = r.flesch_kincaid()
            f_scores.append(str(fk.score))
            gf = r.gunning_fog()
            g_scores.append(gf.score)
        if len(sentences) >= 30:
            #r.smog(all_sentences=True)
            s = r.smog()
            #print(s.score)
            #print(s.grade_level)
            #print("Smog score article :" + str(s))
            s_scores.append(s.score)
        else:
            s_scores.append("NA")


    def nrc_emotions():
        # text_object = NRCLex(text)
        e_counts = 0
        e_counts = Counter()
        neg_list = []
        pos_list = []
        for i in range(len(words2)):
            text_object = NRCLex(words2[i])
            #print(text_object.affect_frequencies)
            for ee in emotion_list:
                for key, value in text_object.raw_emotion_scores.items():
                    if key == ee:
                        e_counts[ee] += value
        object_2 = NRCLex(text)
        #print(object_2.affect_frequencies)
        pos = 0
        neg = 0
        for key, value in object_2.affect_frequencies.items():
            if key in negatives:
                neg += value
            elif key in positives:
                pos += value
            neg_list.append(neg)
            pos_list.append(pos)
        #print(neg)
        #print(pos)
        for ee in emotion_list:
            emotion_list_dict[ee].append(e_counts[ee])
        neg_intensity.append(neg)
        pos_intensity.append(pos)

    def word_tokens():
        tokens = word_tokenize(text)
        count_p = 0
        count_q = 0
        count_news = 0
        count_NER = 0
        count_abbr = 0
        tags = []
        result = []
        for cap in tokens:
            m = re.match(r'\b[A-Z]+\b', cap)
            if m:
                result.append(cap)
        num_caps1 = len(result)
        news_entities = ['ABC','AFP','AP','CBS','CNBC','CNN','ESPN','FOX','GZ','ITAR','NBC','NYT','TASS',
        'REUTERS','Reuters','POLITICO','Politico','WaPo','Washington','WWE','York']
        organisations = ['AFL','AMC','BLM','CDC','CEI','CGT','CIA','CIO','D.H.S.','DHS','D-N.Y.','EPA','EU','FBI','GOP',
        'ISIS','NAFTA','NATO','NCAA','NFL','NH','NIH','NHLBI','L.A.','L.B.J.','OPEC','OSHA','PAC','PIMCO','RAND',
        'R-S.D.','RNC','RUSI','RWE', 'SC','SCOTUS','SEC','SSE','SOS','SPLC','TRP','UCLA','UK','U.N.','UN','U.S.','US','USA','VDARE']
        other = ['ACA','AIDS','COVID']
        abbreviations = ['AWOL','BCE','CEO','ETF','GDP','HIV','II','PM','PTSD','TV','USB','VIP','WWII']
        for abb in result:
            if len(abb) == 1:
                # Removes words such as "I" and "A", so the number of capitalisations isn't inflated.
                result.remove(abb)
            elif abb in news_entities or organisations or other or abbreviations:
                result.remove(abb)
                # This removes acronyms, abbreviations, and news entities
        num_caps2 = len(result)
        for instance in tokens:
            if instance in news_entities:
                count_news +=1
            elif instance in organisations:
                count_NER +=1
            elif instance in other:
                count_NER +=1
            elif instance in abbreviations:
                count_abbr +=1
        #print("Abbreviations: " + str(count_abbr))
        for token in tokens:
            if token in punctuation:
                count_p += 1
        for i in text:
            if i in '“”\"':
                count_q +=1
        quotes = count_q/2
        #print("Number of punctuations:" + str(count_p))
        punctuations.append(count_p)
        quotations.append(quotes)
        #print("Number of all Caps article:" + str(len(result)))
        caps.append(num_caps1)
        caps2.append(num_caps2)
        Named_News.append(count_news)
        Named_ENT.append(count_NER)
        Abbr.append(count_abbr)
        tag = nltk.pos_tag(words2)
        tags.append(tag)
        counts = 0
        counts = Counter()
        for j in tags:
            for pair in j:
                counts[pair[1]] += 1
        for tt in POS_list_dict:
            if tt not in counts:
                counts[tt] = (0)
        for tt in POS_list_dict:
            for key in counts.keys():
                if key == tt:
                    POS_list_dict[tt].append(counts[key])
            

    def stop_words():
        count_sw = 0
        for word in words2:
            if word in en_stops:
                count_sw +=1 
        stoppwords.append(count_sw)
        

    def fluency():
        freqs = []
        total = 0
        for word in words2:
            freq = word_frequency(word, 'en')
            freqs.append(freq)
            total += freq
        freq_avg = total/len(words2)
        # average frequency of all words in article
        for i in freqs:
            if i == 0:
                freqs.remove(i)
        lc = sorted(freqs)
        for j in lc:
            if j == 0:
                lc.remove(j)
        lc2 = lc[:3]
        freq_avg_lc = (lc2[0] + lc2[1] + lc2[2])/3
        # average frequency of least common 3 words in article
        w_frequency.append(freq_avg)
        w_frequency_lc.append(freq_avg_lc)
        #print(len(w_frequency))
        #print(len(w_frequency_lc))
    
    words_per_sentence()
    word_length()
    readable()
    nrc_emotions()
    word_tokens()
    stop_words()
    syntax_depth()
    fluency()
    print()


j = 1    
for i in texts:
    main(i)
    j += 1

df = pd.DataFrame(
            {
                "Word_Count": word_count,
                "Num_tokens": tokens_text,
                "TTR": ttr_text,
                "Num_sentences": num_sentences,
                "Avg_w_length": avg_w_len,
                "WPS": words_sentences,
                "FK_scores": f_scores,
                "GF_scores": g_scores,
                "Smog_scores": s_scores,
                "Num_punctuations": punctuations,
                "Num_quotations": quotations,
                "Num_caps1": caps,
                "Num_caps2":caps2,
                "Named_News": Named_News,
                "Named_ENT" : Named_ENT,
                "Abbreviations" : Abbr,
                "Num_stopwords": stoppwords,
                "Med_depth": med_depth,
                "Med_np_depth": med_np_depth,
                "Med_vp_depth": med_vp_depth,
                "Word_frequency": w_frequency,
                "LC_frequency": w_frequency_lc,
                "avg_negstr" : neg_intensity,
                "avg_posstr" : pos_intensity,
            }
        )

for key, value in POS_list_dict.items():
    df[key] = value

for key, value in emotion_list_dict.items():
    df[key] = value


print(df.T)
df.to_csv(r'name_file.csv')
# define the name of the file you want the results in

