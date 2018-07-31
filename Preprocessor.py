# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:52:01 2017

@author: Jannek
"""

import os
import nltk
import pickle
import pandas
import Utils
from TextFeatures import TextFeatures

stopword_list = nltk.corpus.stopwords.words('finnish')
eos_marker_list = ['!','"','\'','.',':','?']

#https://www.datascience.com/resources/notebooks/word-embeddings-in-python
#sent_w_pos = [nltk.pos_tag(d) for d in sentences]
#sents = [[tup[0]+tup[1] for tup in d] for d in sent_w_pos]

def remove_text_inside_brackets(text, brackets="{}()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)        

def main(Params):
                
    target_data = pickle.load(open(Params['INPUT-targets'],'rb'))
    targets={}
    target_items = set()
    for dimension in target_data:
        targets[dimension] = target_data[dimension][Params['TargetSource']]['item_bias']
        target_items |= set(targets[dimension].keys())

    #root, dir_names, file_names = os.walk(Params['INPUT-folder'])
    file=Params['INPUT-metainfo']
    metainfo = pandas.read_csv(file, engine='python', delimiter='\t', index_col='ID')  # keep original
    metainfo_labels = list(metainfo)

    documents = {}
    file=Params['INPUT-texts']
    drive, path = os.path.splitdrive(file)
    path, filename = os.path.split(path)
    with open(file, 'r', encoding='utf-8') as f:
        STATE=0
        # 0 = initial or document ended
        # 1 = started new document, next must be ID
        # 2 = reading document
        for linenum, line in enumerate(f):
            line = line.split(sep='\t')
            if len(line)>1:
                if line[1]=='<START>':
                    assert int(line[0])==1 and (STATE==0 or STATE==2),'Wrong state!'
                    STATE = 1
                elif line[1]=='<END>':
                    assert int(line[0])==1 and STATE == 2, 'Wrong state!'
                    if STATE==2 and len(sentence)>0:
                        sentences.append(sentence)
                        paragraphs.append(sentences)
                    documents[ID] = paragraphs
                    print('... text \'%s\' had %i paragraphs with %i tokens' % (ID, len(paragraphs), n_token))
                    assert len(paragraphs) > 2 and 200 < n_token < 3000, 'PARSING FAILED!'
                    STATE = 0
                elif STATE == 1:
                    assert int(line[0])==2 and line[1][0:2]=='ID','Wrong ID line!'
                    ID = line[1].strip()
                    paragraphs = []
                    sentence = []
                    sentences = []
                    n_token = 0
                    STATE=2
                # print(line)
                elif STATE==2:
                    if int(line[0]) == 1 and line[1] == 'PARAGRAPH_SEPARATOR':
                        paragraphs.append(sentences)
                        sentences = []
                        sentence=[]
                    else:
                        sentence.append([line[1], line[2], line[3], line[5], line[7]])
                        n_token+=1
            else:
                if STATE==2:
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []



    #data = shuffle(data)
    count=0
    for k in set(documents.keys()).difference(target_items):
        documents.pop(k)
        metainfo = metainfo.drop([k])
        count+=1

    assert len(set(documents.keys()).difference(set(metainfo.index)))==0,'metainfo and document IDs do not match!'

    print('\n----- Data summary: %i text available (dropped %i)\n' % (len(documents),count))

    # FIX ANALYZER ERRORS
    count1=0
    count2=0
    count3=0
    count4=0
    count5=0
    count6 = 0
    for ID, doc in documents.items():
        for k_par,par in enumerate(doc):
            for k_sent,sent in enumerate(par):

                for k, word in enumerate(sent):
                    key = word[0].lower()
                    if key in Params['ExternalData'].fixed_word_list:
                        if Params['ExternalData'].fixed_word_list[key][0] != word[1]:
                            if Params['ExternalData'].fixed_word_list[key][1] is not None:
                                word = word[0:1] + [Params['ExternalData'].fixed_word_list[key][0],Params['ExternalData'].fixed_word_list[key][1]] + word[3:]
                            else:
                                word = word[0:1] + [Params['ExternalData'].fixed_word_list[key][0]] + word[2:]

                            sent[k] = word
                            count5 += 1

                # remove trailing % markers
                remove=[]
                for k,word in enumerate(sent):
                    if k>0:
                        if word[0] == '%' and isnumeric:
                            sent[k - 1][0]=sent[k-1][0]+'%'
                            sent[k - 1][1] = sent[k - 1][1] + '%'
                            remove.append(k)
                    try:
                        float(word[0])
                        isnumeric = 1
                    except:
                        isnumeric = 0
                        pass
                remove.reverse()
                for k in remove:
                    sent=sent[0:k]+sent[(k+1):]
                    count1+=1

                # separate " markers
                k=0
                while k<len(sent):
                    word=sent[k]
                    if len(word[0])>2:
                        if word[0][0]=='"' and word[0][1:].isalpha():
                            # word is: [word,lemma,POS,POS_type, line[7]
                            sent = sent[0:k] + [['"','"','PUNCT','_', 'punct']] + [[word[0][1:],word[1][1:],word[2],word[3],word[4]]] + sent[(k+1):]
                            count2 += 1
                    k+=1

                for k, word in enumerate(sent):
                    if '\xad' in word[0]:
                        word[0]=word[0].replace('\xad','')
                        word[1] = word[1].replace('\xad', '')
                        count3 += 1
                        sent[k]=word

                if sent[-1][2] != 'PUNCT' and sent[-1][0][-1] in ['.',':','!','?',';']:
                    sent = sent[0:k] + [[sent[-1][0][0:-1],sent[-1][1][0:-1],sent[-1][2],sent[-1][3],sent[-1][4]]] + [[sent[-1][0][-1],sent[-1][1][-1],'PUNCT','_', 'punct']]
                    count4 += 1

                if len(sent)>2:
                    arr = []
                    new_sent=[]
                    for k, word in enumerate(sent):
                        arr.append(word)
                        if len(arr)==3:
                            if arr[0][2]=='NUM' and arr[2][2]=='NUM' and arr[1][0]=='/':
                                new_sent.append(['1','1','NUM','_','NumType=Card','nummod'])
                                count6 += 1
                                arr = []
                            else:
                                new_sent.append(arr[0])
                                if k==len(sent)-1:
                                    new_sent.append(arr[1])
                                    new_sent.append(arr[2])
                                arr = arr[1:3]
                    sent = new_sent

                assert len(sent)>0,'Empty sentence, ID=%s!' % ID

                documents[ID][k_par][k_sent]=sent

    print('--- fixed %i percentage marks, %i parenthesis, %i soft hyphens, %i missing punctuation, %i bad lemma words, %i long fractions' % (count1,count2,count3,count4,count5,count6))

    # 2 NUM
    # 3 NumType = Card
    # 4 compound
    fileout = Params['OUTPUT'] + 'final_tokenized_data.txt'
    fout = open(fileout,'w',encoding='utf-8')
    for dimension in target_data:
        fout.write(dimension+',')
    fout.write('ID,normal,normal_lemmed,POS,raw,raw_POS,normal_nostop,lemmed_nostop\n')

    metadata = [metainfo_labels,{}]
    tokenized_documents={}
    for ID,doc in documents.items():

        text = {'RAW':[],'NORMAL':[],'LEMMA':[],'POS':[],'RAW_POS':[],'RAW_LEMMA':[],'NORMAL_NOSTOP':[],'NORMAL_SENTENCES':[],'POS_SENTENCES':[],'LEMMA_SENTENCES':[],'NORMAL_NOSTOP_SENTENCES':[],'LEMMA_NOSTOP_SENTENCES':[]}
        for dimension in target_data:
            fout.write('%f,' % targets[dimension][ID])
        fout.write(ID + ',')
        for k1,par in enumerate(doc):
            for sent in par:
                normal_sent = []
                lemma_sent = []
                pos_sent=[]
                for k2,word in enumerate(sent):

                    text['RAW'].append(word[0])
                    text['RAW_LEMMA'].append(word[1])
                    text['RAW_POS'].append(word[2])
                    if word[2]=='NUM' and word[4]=='compound':
                        pass
                    elif word[2] == 'NUM' and word[3] == 'NumType=Card':
                        text['NORMAL'].append('CARD_NUMBER')
                        text['LEMMA'].append('CARD_NUMBER')
                        text['POS'].append(word[2])
                        normal_sent.append('CARD_NUMBER')
                        lemma_sent.append('CARD_NUMBER')
                        pos_sent.append(word[2])
                    else:
                        text['NORMAL'].append(word[0].lower())
                        text['LEMMA'].append(word[1].lower())
                        text['POS'].append(word[2])
                        pos_sent.append(word[2])
                        normal_sent.append(word[0].lower())
                        lemma_sent.append(word[1].lower())

                text['NORMAL_SENTENCES'].append(normal_sent)
                text['LEMMA_SENTENCES'].append(lemma_sent)
                text['POS_SENTENCES'].append(pos_sent)

                assert len(normal_sent)==len(lemma_sent),'incorrect sentence length'

                text['NORMAL_NOSTOP_SENTENCES'].append([x for x in normal_sent if x not in Params['ExternalData'].stop_words])
                text['LEMMA_NOSTOP_SENTENCES'].append([x for x in lemma_sent if x not in Params['ExternalData'].stop_words])

            if k1<len(doc)-1:
                arr = ('NORMAL_SENTENCES','LEMMA_SENTENCES','NORMAL_NOSTOP_SENTENCES','LEMMA_NOSTOP_SENTENCES',)
                for key in (x for x in text if x not in arr):
                    text[key].append('END_OF_PARAGRAPH')
                for x in arr:
                    text[x].append(['END_OF_PARAGRAPH'])

        text['NORMAL_NOSTOP'] = [x for x in text['NORMAL'] if x not in Params['ExternalData'].stop_words]
        text['LEMMA_NOSTOP'] = [x for x in text['LEMMA'] if x not in Params['ExternalData'].stop_words]

        fout.write('"'+' '.join(text['NORMAL'])+'","'+' '.join(text['LEMMA'])+'","'+' '.join(text['POS'])+'","'+' '.join(text['RAW'])+'","'+' '.join(text['RAW_POS'])+'","'+' '.join(text['NORMAL_NOSTOP'])+'","'+' '.join(text['LEMMA_NOSTOP'])+'"\n')
        tokenized_documents[ID] = text

        metadata[1][ID] = list(metainfo.loc[ID])

    fout.close()

    kernels = Utils.get_StringKernel(tokenized_documents,['NORMAL','LEMMA'])

    return tokenized_documents,documents,metadata,targets,kernels

#data = main()