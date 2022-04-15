import nltk
import os
import statistics

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize, sent_tokenize, pos_tag, RegexpParser
# import sklearn
# from sklearn import preprocessing

# import stanza
# stanza.download('en') 
# nlp = stanza.Pipeline('en')
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

from sentistrength import PySentiStr

import numpy as np 
import pandas as pd 
from wordfreq import word_frequency


# import java.io.StringReader
# import edu.stanford.nlp.trees.Tree
# import edu.stanford.nlp.objectbank.TokenizerFactory
# import edu.stanford.nlp.process.CoreLabelTokenFactory
# import edu.stanford.nlp.ling.CoreLabel
# import edu.stanford.nlp.process.PTBTokenizer
# import edu.stanford.nlp.parser.lexparser.LexicalizedParser

#lp = LexicalizedParser.loadModel("englishPCFG.ser.gz")
#lp.setOptionFlags(new String[]{"-outputFormat", "penn,typedDependenciesCollapsed", "-retainTmpSubcategories"})

#parse = stanford.StanfordParser(model_path="jar/englishPCFG.ser.gz")
parset = CoreNLPParser('http://localhost:9000')

parser = argparse.ArgumentParser()

parser.add_argument('filename', help = "Path to the file that will  be POS tagged",)

args = parser.parse_args()

text_file = args.filename

with open(text_file, 'r') as fp:
    text = fp.read()

characters = len(text)
articles = text.split('\n\n')
#article = articles.split('\n\n')

tokens_body = []
ttr_body = []
num_sentences = []
words_sentences = []
f_scores = []
g_scores = []
s_scores = []
punctuations = []
quotations = []
caps = []
stoppwords = []
POS_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
POS_list_dict = { t: [] for t in POS_list}
med_depth = []
med_vp_depth = []
med_np_depth = []
emotion_list = ['anger', 'fear', 'disgust','sadness', 'negative', 'anticipation', 'trust', 'surprise', 'joy', 'positive']
emotion_list_dict = {e: [] for e in emotion_list}
top_emotion = []
w_frequency = []
w_frequency_lc = []

title_lengths = []
tokens_title = []
ttr_title = []
# num_sentences_t = []
# words_sentences_t = []
# f_scores_t = []
# g_scores_t = []
# s_scores_t = []
# punctuations_t = []
# quotations_t = []
# caps_t = []
# stoppwords_t = []
# POS_list_dict_t = { s: [] for s in POS_list}
# med_depth_t = []
# med_vp_depth_t = []
# med_np_depth_t = []
# emotion_list_dict_t = {f: [] for f in emotion_list}
# top_emotion_t = []
# w_frequency_t = []
# w_frequency_lc_t = []


def main(article):
    article = article
    lines = article.split('\n')
    title = lines[0]
    del lines[0]
    body = ".".join(lines)
    title_words = title.split(' ')
    title_lengths.append(len(title_words))
    #print("Title: " + title)
    #print("Title length: " + str(len(title_words)))
    sentences = sent_tokenize(body)
    tokens = word_tokenize(body)
    lemmas = ld.flemmatize(body)
    #print("Number of tokens article:" + str(len(tokens)))
    tokens_body.append(len(tokens))
    #print("TTR of article:" + str(ld.ttr(lemmas)))
    ttr_body.append(ld.ttr(lemmas))
    sentences_t = sent_tokenize(title)
    #tokens_t = word_tokenize(title)
    #lemmas_t = ld.flemmatize(title)
    #print("Number of tokens article:" + str(len(tokens)))
    tokens_title.append(len(title))
    #print("TTR of article:" + str(ld.ttr(lemmas)))
    ttr_title.append(ld.ttr(title))

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
            #if error raise HTTPError(http_error_msg, response=self)
            #print(parses)
            p = list(parses)
            #t = Tree.fromstring(p[0])
            #print(p[0].height())
            # for i in p:
            #     if p[i] == "S"
            print("***********")
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
                #print(max(VP_subdepths))
                    #print(f"{fraasi.label()}, {fraasi.height()}")
            # for sent in root.subtrees(lambda s: s.label() == 'S'):
            #     for phrase in sent.subtrees(lambda p: p.label() in ('VP', 'NP')):
            #         print(f"S height: {sent.height()}, {phrase.label()} height: {phrase.height()}")
                    #print(f"Sentence: {' '.join(sent.leaves())}")
                    #print(f"Phrase: {' '.join(phrase.leaves())}")
                    #print()        
            # for phrase in root.subtrees(lambda t: t.label() in ("VP", "NP")):
            #     print(f"{phrase.label()}, {phrase.height()}")
            #print(t[1])
            #print(p)
            depths.append(p[0].height())
        medians = statistics.median(depths)
        med_depth.append(medians)
        medianvp = statistics.median(VP_depths)
        med_vp_depth.append(medianvp)
        mediannp = statistics.median(NP_depths)
        med_np_depth.append(mediannp)
        #print("Median syntax tree depth: " + str(med_depth))

    # def syntax_depth_t():
    #     depths = []
    #     VP_depths = []
    #     NP_depths = []
    #     for sentence in sentences_t:
    #         bad_chars = ["%"]
    #         for i in bad_chars:
    #             sentence = sentence.replace(i, "")
    #         parses = parset.parse(sentence.split())
    #         p = list(parses)
    #         print("***********")
    #         root = p[0]
    #         VP_subdepths = []
    #         NP_subdepths = []
    #         for phrase in root.subtrees(lambda t: t.label() == "S"):
    #             for fraasi in phrase.subtrees(lambda t: t.label() == "VP"):
    #                 VP_subdepths.append(fraasi.height())
    #             for fraasi in phrase.subtrees(lambda t: t.label() == "NP"):
    #                 NP_subdepths.append(fraasi.height())
    #             if len(VP_subdepths) > 0:
    #                 VP_depths.append(max(VP_subdepths))    
    #             if len(NP_subdepths) > 0:
    #                 NP_depths.append(max(NP_subdepths)) 
    #         depths.append(p[0].height())
    #     medians = statistics.median(depths)
    #     med_depth_t.append(medians)
    #     medianvp = statistics.median(VP_depths)
    #     med_vp_depth_t.append(medianvp)
    #     mediannp = statistics.median(NP_depths)
    #     med_np_depth_t.append(mediannp)

    def words_per_sentence():
        summa = 0
        summa_t = 0
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
        words_sentences.append(summa/len(sentences))
        #words_sentences_t.append()
        
    def readable():  
        r = Readability(article)
        fk = r.flesch_kincaid()
        gf = r.gunning_fog()
        #print("Flesch-Kincaid score article:" + str(fk))
        f_scores.append(str(fk.score))
        #print("Gunning Fog score article:" + str(gf))
        g_scores.append(gf.score)
        if len(sentences) >= 30:
            s = r.smog()
            #print("Smog score article :" + str(s))
            s_scores.append(s.score)
        else:
            #print("Not enough sentences to compute Smog score.")
            s_scores.append("NaN")


    def nrc_emotions():
        text_object = NRCLex(article)
        print("+++++++++++")
        #print(text_object.affect_dict) # e.g. church': ['anticipation', 'joy', 'positive', 'trust'], 'law': ['trust'], 'effort': ['positive'], 'force': ['anger', 'fear', 'negative'],
        #print(text_object.raw_emotion_scores) # anticipation: 23, joy: 34 etc.
        print(text_object.top_emotions) # [('positive', 0.21008403361344538)]
        print(text_object.affect_frequencies) # 'fear': 0.12184873949579832, 'anger': 0.058823529411764705, 'anticip': 0.0, 'trust': 0.1806722689075630
        #("{%.2f}".format(text_object.affect_frequencies)) 
        # obtain the frequencies of each emotion within our tweets/ proportion out of 1
        print("**************")
        for ee in emotion_list:
            for key, value in text_object.raw_emotion_scores.items():
                if key == ee:
                    emotion_list_dict[ee].append(value)
            if ee not in text_object.raw_emotion_scores.keys():
                emotion_list_dict[ee].append(0)
        #top_emotion.append(text_object.top_emotions[0:key])
        #print((emotion_list_dict))

        # works best for unicode text, work on this later

    def word_tokens():
        count_p = 0
        count_q = 0
        #count_p = Counter()
        #count_q = Counter()
        tags = []
        result = []
        b = article
        result = re.findall(r'\b[A-Z]+\b', b)
        for token in tokens:
            if token in punctuation:
                count_p += 1
        for i in b:
            if i in '“”\"':
                count_q +=1
        quotes = count_q/2
        #print("Number of punctuations:" + str(count_p))
        punctuations.append(count_p)
        # This somehow needs to be fixed by getting the unicode version of the quotation marks
        #print("Number of quotations article:" + str(quotes))
        quotations.append(quotes)
        #print("Number of all Caps article:" + str(len(result)))
        # account for abbreviations e.g. CIA or proper nouns e.g. COVID
        caps.append(len(result))
        tag = nltk.pos_tag(tokens)
        tags.append(tag)
        counts = 0
        counts = Counter()
        for j in tags:
            for pair in j:
                counts[pair[1]] += 1
        for tt in POS_list_dict:
            if tt not in counts:
                counts[tt] = (0)
        #jarjestys = sorted(counts)
        #print(counts)
        #print(jarjestys)
        #print(len(counts))
        for tt in POS_list_dict:
            for key in counts.keys():
                if key == tt:
                    POS_list_dict[tt].append(counts[key])
            

    def stop_words():
        count_sw = 0
        words = article.split(' ')
        #for word in articles[k]:
        for word in words:
            if word in en_stops:
                count_sw +=1 
                #maybe refine to amount of each stopword, not just cumulative together
        #print("Stopwords in article: " + str(count_sw))
        stoppwords.append(count_sw)

    def fluency():
        freqs = []
        total = 0
        words = article.split(' ')
        for word in words:
            freq = word_frequency(word, 'en')
            freqs.append(freq)
            total += freq
        freq_avg = total/len(words)
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
        print("#####")
        print(w_frequency)
        print(w_frequency_lc)
    
    words_per_sentence()
    readable()
    nrc_emotions()
    word_tokens()
    stop_words()
    syntax_depth()
    #syntax_depth_t()
    fluency()
    print()


j = 1    
for i in articles:
    #print("Article: " + str(j))
    main(i)
    j += 1

# print(title_lengths)
# print(tokens_article)
# print(ttr_article)
# print(num_sentences)
# print(words_sentences)
# print(f_scores)
# print(g_scores)
# print(s_scores)
# print(punctuations)
# print(quotations)
# print(caps)
# print(stoppwords)
# print(POS_list_dict)

df = pd.DataFrame(
            {
                "Title_Length": title_lengths,
                "Num_tokens": tokens_body,
                "TTR": ttr_body,
                "Num_sentences": num_sentences,
                "WPS": words_sentences,
                "FK_scores": f_scores,
                "GF_scores": g_scores,
                "Smog_scores": s_scores,
                "Num_punctuations": punctuations,
                "Num_quotations": quotations,
                "Num_caps": caps,
                "Num_stopwords": stoppwords,
                "Med_depth": med_depth,
                "Med_np_depth": med_np_depth,
                "Med_vp_depth": med_vp_depth,
                "Word_frequency": w_frequency, 
                "LC_frequency": w_frequency_lc,
            }
        )

for key, value in POS_list_dict.items():
    df[key] = value

for key, value in emotion_list_dict.items():
    df[key] = value

#df["top_emotion"] = top_emotion


# Complexity features: words per sentencs, syntax tree depth, NP syntax depth, Vp syntax depth, Gunning Fog, SMOG, Flesh, TTR
# Psychological features: LIWC --> NRC Emotions, SentiStrength =?

# nt: normalised to token counts
# nc: normalised to character counts
# numsennorm = preprocessing.normalize(num_sentences)
# numpuncnorm = preprocessing.normalize(punctuations)
# numquotnorm = preprocessing.normalize(quotations)
# numcapsnorm = preprocessing.normalize(caps)
# numstopnorm = preprocessing.normalize(stoppwords)

# df_norm = pd.DataFrame(
#             {
#                 "Title_Length": title_lengths,
#                 "Num_tokens": tokens_article,
#                 "TTR": ttr_article,
#                 "Num_sentences_nt": numsennorm,
#                 "WPS": words_sentences,
#                 "FK_scores": f_scores,
#                 "GF_scores": g_scores,
#                 "Smog_scores": s_scores,
#                 "Num_punctuations_nc": numpuncnorm,
#                 "Num_quotations_nc": numquotnorm,
#                 "Num_caps_nt": numcapsnorm,
#                 "Num_stopwords_nt": numstopnorm,
#                 # "Med_depth": med_depth,
#                 # "Med_np_depth": med_np_depth,
#                 # "Med_vp_depth": med_vp_depth,

                 
#             }
#         )

# for key, value in POS_list_dict.items():
#     df[key] = value

#data <- read.table('fake.csv', sep=',', header=T)
#df.style.format("{:.4%}")


print(df.T)
#print("Tokens: "+ str(sum(tokens_article)))
#print("Chars: " + str(characters))
df.to_csv(r'fake.csv')
