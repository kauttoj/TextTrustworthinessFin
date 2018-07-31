# for both storing and applying external dictionaries

import numpy as np
import os
import pickle
import re

class TextFeatures():

    def __init__(self,ROOT_PATH=None,data=None,RECOMPUTE_ALL=0,print_info=False):
        self.stop_words = None
        self.positive_words = None
        self.negative_words = None
        self.word_embedding = None
        self.technical_terms=None
        self.subjectivity_terms = None
        self.embedding_dim = -1
        self.finnish_words=None

        self.cached_text_lengths = None

        self.RECOMPUTE_ALL=RECOMPUTE_ALL

        self.full_vocabulary = None
        self.data = data
        self.print_info = print_info
        self.ROOT_PATH = ROOT_PATH

        #self.letters = r'abcdefghijklmnopqrstuvxyzåäö'
        #self.letters = re.compile(self.letters + self.letters.upper() + '.-,:!?"')

    def load_external_lists(self):

        #self.numeric_identifiers = ['g/kg', 'mg', 'milligramma', 'gramma', 'kg', 'g', 'kilogramma', 'kilo', 'desi',
        #                            'desilitra', 'dl', '%', 'prosentti', 'nmol/l', 'IU', 'mikrogramma', 'mmHg']

        filename = self.ROOT_PATH + r'negative.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = set([x.strip().lower() for x in text])
            self.negative_words = text

        filename = self.ROOT_PATH + r'positive.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = set([x.strip().lower() for x in text])
            self.positive_words = text

        filename = self.ROOT_PATH + r'polyglot_negative.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = set([x.strip().lower() for x in text])
            for w in text:
                if w not in self.negative_words:
                    self.negative_words.add(w)

        filename = self.ROOT_PATH + r'polyglot_positive.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = set([x.strip().lower() for x in text])
            for w in text:
                if w not in self.positive_words:
                    self.positive_words.add(w)

        filename = self.ROOT_PATH + r'stopwords-fi.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = set([x.strip().lower() for x in text])
            self.stop_words = text

        filename = self.ROOT_PATH + r'technical_terms.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = set([x.strip().lower() for x in text])
            self.technical_terms = text

        # MPQA (Multi-Perspective Question Answering)
        filename = self.ROOT_PATH + r'subjectivity_fin.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = [(x.strip()).split('\t') for x in text[1:]]
            self.subjectivity_terms={}
            for line in text:
                try:
                    self.subjectivity_terms[line[0].lower()]=int(line[1])
                except:
                    pass

        # modern finnish word list (kotus)
        filename = self.ROOT_PATH + r'kotus_sanalista_v1.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = set([x.strip().lower() for x in text])
            self.finnish_words=text

        # good POS ngrams
        filename = self.ROOT_PATH + r'common_POS_ngrams.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = [(x.strip()).split('\t') for x in text]
            flag=False
            rate_limit = 0.01 # 1 percent
            common_POS_tags=[]
            for line in text:
                if flag and 'X' not in line[0]:
                    line[0] = line[0].replace('CCONJ', 'CONJ')
                    rate = float(line[2])
                    if rate>=rate_limit:
                        common_POS_tags.append(line[0])
                    else:
                        break
                elif line[0]=='POS_ngram' and line[1]=='total_count' and line[2]=='rate':
                    flag = True
        assert flag
        self.common_POS_tags=set(common_POS_tags)

        # custom lemma list
        filename = self.ROOT_PATH + r'fixed_word_list.txt'
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.readlines()
            text = [(x.strip()).split('\t') for x in text[1:]]
            self.fixed_word_list={}
            for line in text:
                if len(line)==3:
                    self.fixed_word_list[line[0]]=(line[2],None,)
                elif len(line)==4:
                    self.fixed_word_list[line[0]] =(line[2],line[3],)

    def build_full_vocab(self,data_in):
        self.full_vocabulary = set()
        for key,data in data_in.items():
            for token in data['RAW']:
                self.full_vocabulary.add(token)
                self.full_vocabulary.add(token.lower())
            for token in data['NORMAL']:
                self.full_vocabulary.add(token)
                self.full_vocabulary.add(token.lower())
            for token in data['LEMMA']:
                self.full_vocabulary.add(token)
                self.full_vocabulary.add(token.lower())
                tok = token.replace('#','-')
                self.full_vocabulary.add(tok)
                self.full_vocabulary.add(tok.lower())
                tok = token.replace('#','')
                self.full_vocabulary.add(tok)
                self.full_vocabulary.add(tok.lower())

    def load_external_embeddings(self,data):

        EMBEDDING_DIM = -1

        filename11 = self.ROOT_PATH + r'fin-word2vec.bin'
        filename21 = self.ROOT_PATH + r'fin-word2vec_SELECTED.pickle'

        filename12 = self.ROOT_PATH + r'cc.fi.300.vec'
        filename22 = self.ROOT_PATH + r'cc.fi.300_SELECTED.pickle'

        if not os.path.isfile(filename21) or not os.path.isfile(filename22) or self.RECOMPUTE_ALL:

            # get all words in corpus
            self.build_full_vocab(data)

            import gensim

            embeddings_index={}

            print('..reading embedding model 1')
            model = gensim.models.KeyedVectors.load_word2vec_format(filename11,binary=True)  # you can continue training with the loaded model!
            k=0
            for word in self.full_vocabulary:
                if word in model:
                    k+=1
                    coefs = np.asarray(model[word], dtype='float32')
                    embeddings_index[word] = coefs
            embedding_dim = len(coefs)
            if self.print_info:
                print('Storing embedding dictionary with selected %i words' % k)
            pickle.dump(embeddings_index, open(filename21, 'wb'))

            self.embedding_dim = embedding_dim
            self.word_embedding = embeddings_index

            embeddings_index = {}

            print('..reading embedding model 2')
            model = gensim.models.KeyedVectors.load_word2vec_format(filename12, binary=False)  # you can continue training with the loaded model!
            k = 0
            for word in self.full_vocabulary:
                if word in model:
                    k += 1
                    coefs = np.asarray(model[word], dtype='float32')
                    embeddings_index[word] = coefs
            embedding_dim = len(coefs)
            if self.print_info:
                print('Storing embedding dictionary with selected %i words' % k)
            pickle.dump(embeddings_index, open(filename22, 'wb'))

            self.embedding_dim_fasttext = embedding_dim
            self.word_embedding_fasttext = embeddings_index

            # create list with all unknown words, most likely with bad lemmas
            wordlist={}
            for key, dat in data.items():
                for k,token in enumerate(dat['RAW_LEMMA']):
                    tok1 = token.lower()
                    if tok1 not in ('card_number','end_of_paragraph') and tok1.find('_tag')==-1 and data[key]['RAW_POS'][k]!='NUM':
                        tok2 = tok1.replace('#','-')
                        tok3 = tok1.replace('#','')
                        if tok1 in self.word_embedding or tok1 in self.word_embedding_fasttext:
                            pass
                        elif tok2 in self.word_embedding or tok2 in self.word_embedding_fasttext:
                            pass
                        elif tok3 in self.word_embedding or tok3 in self.word_embedding_fasttext:
                            pass
                        else:
                            wordlist[data[key]['RAW'][k].lower()]='%s\t%s\t%s\n' % (data[key]['RAW'][k].lower(),tok1,tok1)

            filename = self.ROOT_PATH + r'unknown_word_list.txt'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('RAW_word\tLEMMA_word\tnew_LEMMA_word\n')
                for key in wordlist.keys():
                    f.write(wordlist[key])

        else:

            if self.print_info:
                print('Loading pre-stored embedding dictionary')
            embeddings_index = pickle.load(open(filename21, 'rb'))
            embedding_dim = len(embeddings_index[list(embeddings_index.keys())[0]])

            self.embedding_dim = embedding_dim
            self.word_embedding = embeddings_index

            if self.print_info:
                print('Loading pre-stored embedding dictionary')
            embeddings_index = pickle.load(open(filename22, 'rb'))
            embedding_dim = len(embeddings_index[list(embeddings_index.keys())[0]])

            self.embedding_dim_fasttext = embedding_dim
            self.word_embedding_fasttext = embeddings_index




        return data
    # def get_metainfo(self,x_meta_in,data_x,labels=None):
    #     x_meta = np.array(x_meta_in,dtype=float)
    #     x_meta_scaled = x_meta.copy()
    #     x_length = np.zeros((len(data_x['RAW']),1))
    #     for k in range(0, len(data_x['RAW'])):
    #         RAW = [x for x in data_x['RAW'][k] if x!='END_OF_PARAGRAPH']
    #         raw_text = " ".join(data_x['RAW'][k])
    #         x_length[k] = len(raw_text)
    #     X = np.concatenate((x_meta, x_meta_scaled / x_length), axis=1)
    #     if labels!=None:
    #         X_labels = labels + [(x + '_ratio') for x in labels]
    #         return X,X_labels
    #     else:
    #         return X
    def get_ngrams(self,list_of_tokens, min_n, max_n):
        """
        Generates ngrams(word sequences of fixed length) from an input token sequence.
        tokens is a list of words.
        min_n is the minimum length of an ngram to return.
        max_n is the maximum length of an ngram to return.
        returns a list of ngrams (words separated by a space)
        """
        all_ngrams = list()
        for tokens in list_of_tokens:
            n_tokens = len(tokens)
            for i in range(n_tokens):
                for j in range(i + min_n, min(n_tokens, i + max_n) + 1):
                    all_ngrams.append(" ".join(tokens[i:j]))
        return all_ngrams


    def get_length_features(self,data_x):

        N_features = 28

        X_labels = ['NONE' for k in range(0,N_features)]

        X_labels[0] = 'total_char_count'
        X_labels[1] = 'word_count'
        X_labels[2] = 'total_sentence_count'
        X_labels[3] = 'comma_char_count'
        X_labels[4] = '!_char_count'
        X_labels[5] = '?_char_count'
        X_labels[6] = 'positive_word_count'
        X_labels[7] = 'negative_word_count'
        X_labels[8] = 'stop_word_count'
        X_labels[9] = 'technical_word_count'
        X_labels[10] = 'subjective_word_count'
        X_labels[11] = 'dictionary_word_count'
        X_labels[12] = 'symbol_count'
        X_labels[13] = 'digit_count'
        X_labels[14] = 'emb_dict_word_count'
        X_labels[15] = 'VERB_count'
        X_labels[16] = 'PROPN_count'
        X_labels[17] = 'SYM_count'
        X_labels[18] = 'PUNCT_count'
        X_labels[19] = 'NOUN_count'
        X_labels[20] = 'ADJ_count'
        X_labels[21] = 'AUX_count'
        X_labels[22] = 'CONJ_count'
        X_labels[23] = 'ADP_count'
        X_labels[24] = 'SCONJ_count'
        X_labels[25] = 'PRON_count'
        X_labels[26] = 'NUM_count'
        X_labels[27] = 'common_POS_ngram_count'

        X_labels += [y + '_ratio' for y in X_labels[1:]]

        self.cached_text_lengths={}

        #if isinstance(data_x,dict):
        # will return dict of lists

        X={}
        for key in data_x.keys():

            LEMMA = [x.replace('#', '') for x in data_x[key]['LEMMA'] if x != 'END_OF_PARAGRAPH']
            RAW = [x for x in data_x[key]['RAW'] if x != 'END_OF_PARAGRAPH']
            NORMAL = [x for x in data_x[key]['NORMAL'] if x != 'END_OF_PARAGRAPH']

            assert len(LEMMA) == len(NORMAL)

            pos_text = " ".join(data_x[key]['POS'])
            pos_ngrams = self.get_ngrams(data_x[key]['POS_SENTENCES'],2,3)

            N_sent = len([x for x in data_x[key]['NORMAL_SENTENCES'] if len(x)>1])

            raw_text = " ".join(RAW)
            raw_text=raw_text.replace(' , ',', ')
            raw_text = raw_text.replace(' . ', '. ')
            raw_text = raw_text.replace(' ? ', '? ')
            raw_text = raw_text.replace(' ! ', '! ')
            raw_text = raw_text.replace(' : ', ': ')
            raw_text = raw_text.replace(' " ', '" ')

            X[key]=[np.nan for k in range(0,N_features)]

            # character count (everything!)
            X[key][0] = len(raw_text)

            # word count (not including punctuations, single letter)
            X[key][1] = sum(np.array([len(x)>1 for x in NORMAL]))

            # sentence count
            X[key][2] = N_sent

            # comma count
            X[key][3] = raw_text.count(',')

            # ! count
            X[key][4] = raw_text.count('!')

            # ? count
            X[key][5] = raw_text.count('?')

            # positive word count
            X[key][6] = sum(x[0] in self.positive_words or x[1] in self.positive_words for x in zip(NORMAL,LEMMA))

            # negative word count
            X[key][7] = sum(x[0] in self.negative_words or x[1] in self.negative_words for x in zip(NORMAL,LEMMA))

            # stop word count
            X[key][8] = sum(x in self.stop_words for x in RAW)

            # how many technical terms
            X[key][9] = sum(x[0] in self.technical_terms or x[1] in self.technical_terms for x in zip(NORMAL,LEMMA))

            # subjectivity terms
            s=0
            for k in range(0,len(NORMAL)):
                if NORMAL[k] in self.subjectivity_terms:
                    s += self.subjectivity_terms[NORMAL[k]]
                    continue
                if LEMMA[k] in self.subjectivity_terms:
                    s += self.subjectivity_terms[LEMMA[k]]
            X[key][10] = s

            # how many words found in dictionary
            X[key][11] = sum(x[0] in self.finnish_words or x[1] in self.finnish_words for x in zip(NORMAL,LEMMA))

            # how many non-alphabet characters (e.g., symbols)
            X[key][12] = sum(not(x.isalpha()) for x in raw_text)

            # how many digits
            X[key][13] = sum(x.isnumeric() for x in raw_text)

            # how many tokens have embedding
            X[key][14] = sum(x in self.word_embedding or x in self.word_embedding_fasttext for x in RAW)

            # count all POS tags
            X[key][15] = pos_text.count('VERB')
            X[key][16] = pos_text.count('PROPN')
            X[key][17] = pos_text.count('SYM')
            X[key][18] = pos_text.count('PUNCT')
            X[key][19] = pos_text.count('NOUN')
            X[key][20] = pos_text.count('ADJ')
            X[key][21] = pos_text.count('AUX')
            X[key][22] = pos_text.count('CONJ')
            X[key][23] = pos_text.count('ADP')
            X[key][24] = pos_text.count('SCONJ')
            X[key][25] = pos_text.count('PRON')
            X[key][26] = pos_text.count('NUM')

            # count
            X[key][27] = sum(x in self.common_POS_tags for x in pos_ngrams)

            # words in all caps
            # X[k ,6] = sum(np.array([x.isupper() for x in ]))
            # how many foreign words
            # X[k, 10] = 0
            # scale each column by text length (all characters)
            self.cached_text_lengths[key]=X[key][0]

            X_scaled = X[key][1:].copy()
            for col in range(0,len(X_scaled)):
                X_scaled[col] = X_scaled[col] / X[key][0]

            X[key] += X_scaled

            assert len(X[key])==len(X_labels)==1+2*(N_features-1)

        return X_labels,X

        # else:
        #     # will return np.array
        #
        #     X = np.zeros((len(data_x['RAW']),N_features))
        #
        #     for k in range(0,len(data_x['RAW'])):
        #
        #         #print('..... text %i' % k)
        #
        #         LEMMA = [x.replace('#', '') for x in data_x['LEMMA'][k] if x!='END_OF_PARAGRAPH']
        #         RAW = [x for x in data_x['RAW'][k] if x not in ('END_OF_PARAGRAPH',)]
        #         pos_text = " ".join(data_x['POS'][k])
        #         N_sent = len(data_x['NORMAL_SENTENCES'][k])
        #
        #         raw_text = " ".join(data_x['RAW'][k])
        #         raw_text=raw_text.replace(' , ',', ')
        #         raw_text = raw_text.replace(' . ', '. ')
        #         raw_text = raw_text.replace(' ? ', '? ')
        #         raw_text = raw_text.replace(' ! ', '! ')
        #         raw_text = raw_text.replace(' : ', ': ')
        #         raw_text = raw_text.replace(' " ', '" ')
        #
        #         # character count
        #         X[k ,0] = len(raw_text)
        #
        #         # comma count
        #         X[k ,1] = raw_text.count(',')
        #
        #         # text character count (not including spaces)
        #         X[k ,2] = sum(np.array([len(x) for x in RAW]))
        #
        #         # ! count
        #         X[k ,3] = raw_text.count('!')
        #
        #         # ? count
        #         X[k, 4] = raw_text.count('?')
        #
        #         # positive word count
        #         X[k ,5] = sum(np.array([x in self.positive_words for x in RAW])) + sum(np.array([x in self.positive_words for x in LEMMA]))
        #
        #         # negative word count
        #         X[k ,6] = sum(np.array([x in self.negative_words for x in RAW])) + sum(np.array([x in self.negative_words for x in LEMMA]))
        #
        #         # stop word count
        #         X[k ,7] = sum(np.array([x in self.stop_words for x in RAW]))
        #
        #         # how many technical terms
        #         X[k ,8] = sum(np.array([x in self.technical_terms for x in RAW])) + sum(np.array([x in self.technical_terms for x in LEMMA]))
        #
        #         # subjectivity terms
        #         X[k ,9] = sum(np.array([self.subjectivity_terms[x] for x in RAW if x in self.subjectivity_terms])) \
        #                   + sum(np.array([self.subjectivity_terms[x] for x in LEMMA if x in self.subjectivity_terms]))
        #
        #         # how many words found in dictionary
        #         X[k, 10] = sum(np.array([x in self.finnish_words for x in RAW])) \
        #                    + sum(np.array([x in self.finnish_words for x in LEMMA]))
        #
        #         # how many non-alphabet characters (e.g., symbols)
        #         X[k, 11] = sum(not(x.isalpha()) for x in raw_text)
        #
        #         # how many digits
        #         X[k, 12] = sum(x.isnumeric() for x in raw_text)
        #
        #         # how many words have embedding
        #         X[k, 13] = sum(x.lower() in self.word_embedding or x.lower() in self.word_embedding_fasttext for x in RAW)
        #
        #         X[k, 14] = pos_text.count('VERB')
        #         X[k, 15] = pos_text.count('PROPN')
        #         X[k, 16] = pos_text.count('SYM')
        #         X[k, 17] = pos_text.count('PUNCT')
        #         X[k, 18] = pos_text.count('NOUN')
        #         X[k, 19] = pos_text.count('ADJ')
        #         X[k, 20] = pos_text.count('AUX')
        #         X[k, 21] = pos_text.count('CONJ')
        #         X[k, 22] = pos_text.count('ADP')
        #         X[k, 23] = pos_text.count('SCONJ')
        #         X[k, 24] = pos_text.count('PRON')
        #
        #         # words in all caps
        #         #X[k ,6] = sum(np.array([x.isupper() for x in ]))
        #         # how many foreign words
        #         #X[k, 10] = 0
        #
        #     # scale each column by text length (all characters)
        #     X_scaled = X[:,1:].copy()
        #     for col in range(0 ,X_scaled.shape[1]):
        #         X_scaled[: ,col] = X_scaled[: ,col ]/X[: ,0]
        #         X_labels+=[X_labels[col]+'_ratio']
        #
        #     X = np.concatenate((X ,X_scaled) ,axis=1)



    def add_ratios(self,data_x):

        assert self.cached_text_lengths is not None,'Cached text lengths is empty, need to run get_length_features first for all data!'
        assert len(data_x[1]) == len(self.cached_text_lengths),'Cached text length data must have same number of keys as input data!'

        for key in data_x[1].keys():

            X_scaled = data_x[1][key].copy()
            for col in range(0,len(X_scaled)):
                X_scaled[col] = float(X_scaled[col]) / float(self.cached_text_lengths[key])
            data_x[1][key] += X_scaled

        data_x[0] = data_x[0] + [y+'_ratio' for y in data_x[0]]

        return data_x