# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_selection import f_regression,chi2,SelectKBest,mutual_info_regression
from collections import defaultdict
from sklearn.preprocessing import MaxAbsScaler,StandardScaler
from sklearn.metrics.pairwise import polynomial_kernel,linear_kernel,cosine_similarity,rbf_kernel
import itertools
import time
try:
    from fisher import pvalue
except ImportError:
    pass
else:
    pass
#from sklearn.base import BaseEstimator, TransformerMixin

def mutual_info_regression_partial(X,Y,N_max=500):
    N=X.shape[1]
    vals = []
    k = 0
    while 1:
        k_end = min(N, k + N_max)
        val = mutual_info_regression(X[:, k:k_end],Y,random_state=1)
        vals.append(val)
        k += N_max
        if k >= N:
            break
    assert len(vals)==N
    vals = np.concatenate(tuple(vals), axis=0)
    return vals

def get_scaler(FEATURE_SCALER):
    if FEATURE_SCALER == 'MaxAbsScaler':
        return MaxAbsScaler()
    elif FEATURE_SCALER == 'StandardScaler':
        return StandardScaler()
    else:
        raise (Exception('Unknown scaler name: %s' % FEATURE_SCALER))

def get_kernel(KERNELTYPE):
    if KERNELTYPE=='linear':
        return linear_kernel
    elif KERNELTYPE=='radial':
        return rbf_kernel
    elif KERNELTYPE=='cosine':
        return cosine_similarity
    else:
        raise(Exception('Uknown kernel type!'))

def get_stringkernel(kerneldata,text_types,target_text_IDs,ngram_length):

    kernel = 0.0
    for text_type in text_types:
        source_text_IDs = kerneldata[text_type][ngram_length][0]
        rows = [source_text_IDs.index(x) for x in target_text_IDs[0]]
        if len(target_text_IDs)==2:
            cols = [source_text_IDs.index(x) for x in target_text_IDs[1]]
        else:
            cols=rows
        ker = kerneldata[text_type][ngram_length][1][np.ix_(rows,cols)]
        kernel+=ker

    return kernel/len(text_types)

def replace_hash(texts):
    new_texts = []
    for text in texts:
        text = [x.replace('#','') for x in text]
        new_texts.append(text)
    return new_texts

def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

class FeatureExtractor(object):

    def __init__(self,embedded_transformer=None,external_features=None,FEATURE_SCALER = 'StandardScaler'):
        """
        Initialize variables and check essay set type
        """
        self.transformers = None
        self.feature_type = None
        self.selected_columns = None
        self.POS_ngram = (2,3,)  # hard-coded, no unigrams
        self.text_types = None
        self.analysis_type = None
        self.featureselection = None
        self.embedded_transformer = embedded_transformer
        self.external_features = external_features
        self.word_embedding = None
        self.maxlen_words=None
        self.final_features=-1
        self.maxlen =None
        self.sequence_model_type = None
        self.embedding_type = None
        self.feature_scale_multiplier = None
        self.ngramMaxLength=None
        self.FEATURE_SCALER = FEATURE_SCALER
        self.sequence_vocabulary=None
        self.final_data_scaler = None # final scaling transform of data matrix

    # apply transforms to data, used for test data, no learning takes place! No target data is used
    # order of operations is crucial! Otherwise results are nonsense
    def transform(self,data_x,x_meta=None,x_custom=None,post_transformer=None,text_IDs=None,stringkernels=None):

        #print('Transforming features (testing)')
        XX = []
        if 'CUSTOM' in self.feature_type:
            x = np.array(x_custom, dtype=float)
            XX.append(x)

        if 'TAGS' in self.feature_type:
            x = np.array(x_meta, dtype=float)
            XX.append(x)

        if len(XX)>0:
            XX = np.concatenate(tuple(XX), axis=1)

        if self.analysis_type == 'SEQUENCE':
            from keras.preprocessing import sequence

            # apply fiature scaling
            if len(XX) > 0:
                XX = self.final_data_scaler.transform(XX)

            if self.sequence_model_type == 'FASTTEXT':

                X = data_x[self.text_types[0]]

                # replace unknown words with RARE_WORD
                for k1 in range(0,len(X)):
                    for k2 in range(0,len(X[k1])):
                        token = X[k1][k2]
                        if token not in self.sequence_vocabulary:
                            X[k1][k2]='RARE_WORD'

                # convert words to counts
                X_mat = self.transformer.transform(X)

                X=[]
                for row in range(X_mat.shape[0]):
                    # how many tokens of a kind
                    tokens=[-1 for _ in range(np.sum(X_mat[row,:]))]
                    # nonzero elements
                    ind = np.argwhere(X_mat[row,:]>0)
                    k=0
                    for _,col in ind:
                        for _ in range(0,X_mat[row,col]):
                            tokens[k]=col+1
                            k+=1
                    assert tokens[-1]>-1,'Negative indices found! BUG!'
                    X.append(tokens)

                # print('Pad sequences (samples x time)')
                if len(XX)>0:
                    X_test = [sequence.pad_sequences(X, maxlen=self.maxlen),XX]
                else:
                    X_test = [sequence.pad_sequences(X, maxlen=self.maxlen)]
                # x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

            else:

                X = {'SENTENCES': [], 'FLAT': []}
                for k1, text in enumerate(data_x[self.text_types[0] + '_SENTENCES']):
                    X['SENTENCES'].append([])
                    for k2, sent in enumerate(text):
                        if k2 < self.maxlen_doc:
                            X['SENTENCES'][-1].append([])
                            for k3, word in enumerate(sent):
                                if k3<self.maxlen_sent:
                                    # three cases: (1) in list and dictionary (2) in dictionary (3) nowhere
                                    if word not in self.sequence_vocabulary:
                                        word = 'RARE_WORD'
                                    word_index = self.sequence_vocabulary[word]
                                    X['SENTENCES'][-1][-1].append(word_index)
                    X['FLAT'].append(list(itertools.chain.from_iterable(X['SENTENCES'][-1])))

                X['FLAT'] = sequence.pad_sequences(X['FLAT'], maxlen=self.maxlen_words)

                if len(XX)>0:
                    X_test = [X,XX]
                else:
                    X_test = [X,]

        elif self.analysis_type[0:3] == 'BOW':

            # apply all transformers sequentically (same as in training)
            X_test = []
            for transformer in self.transformers:
                # apply raw data transform
                x = transformer[0].transform(data_x[transformer[1]]).todense()#.astype(np.float32)
                # apply scaling transform, identity for TFIDF
                X_test.append(x)

            # apply all selections sequentically (same as in training)
            is_selected = False
            if len(X_test)>0 and self.featureselection != None and self.featureselection[1] != 'global':
                is_selected=True
                if self.featureselection[1] == 'single':
                    for k,x in enumerate(X_test):
                        X_test[k] = np.take(x, indices=self.selected_columns[k], axis=1)
                elif self.featureselection[1] == 'all':
                    X_test = np.concatenate(tuple(X_test), axis=1)
                    X_test = [np.take(X_test, indices=self.selected_columns, axis=1)]
                else:
                    raise(Exception('Unknown featureselection, must be single or all!'))

            # add embedding features
            if 'EMBEDDING' in self.feature_type:
                if self.embedding_type=='LEMMA':
                    x = self.embedded_transformer.transform(replace_hash(data_x[self.embedding_type]))#.astype(np.float32)
                else:
                    x = self.embedded_transformer.transform(data_x[self.embedding_type])  # .astype(np.float32)
                X_test.append(x)

            # add external data, if any
            if len(XX) > 0:
                X_test.append(XX)

            X_test = np.concatenate(tuple(X_test),axis=1)

            if self.featureselection != None and self.featureselection[1] == 'global':
                assert is_selected==False,'Trying selection twice!'
                X_test = np.take(X_test, indices=self.selected_columns, axis=1)

            # apply fiature scaling
            X_test = self.final_data_scaler.transform(X_test)

            if self.analysis_type == 'BOW_StringKernel':
                assert len(set(text_IDs[0]).intersection(text_IDs[1]))==0,'test and train data are overlapping!'
                X_stringkernel = get_stringkernel(stringkernels, self.text_types, text_IDs, self.ngramMaxLength)
                X_test = self.stringkernel_ratio * X_stringkernel + (1.0 - self.stringkernel_ratio) * self.kernelfunction(X=X_test, Y=self.kerneldata_Y)

            assert self.final_features == X_test.shape[1],'Final feature size not equal!'

        if post_transformer is not None:
            X_test = post_transformer(X_test)

        return X_test

    # get best features
    def get_best(self,x,pass2_features):
        ind = np.argsort(x)
        ind = np.flipud(ind)
        assert x[ind[0]] == max(x), 'sort failed!'
        return ind[0:pass2_features]

    # method to choose columns
    def column_selector(self,X,Y,type,pass2_features):
        if type == 'regression':
            val = f_regression(X,Y)
            val = val[0]/np.max(val[0]) # these are f-values!
            return self.get_best(val,pass2_features)
        elif type == 'fisher':
            return self.fisher_selector(Y,X,pass2_features)
        elif type == 'chi2':
            return self.chi2_selector(Y,X,pass2_features)
        elif type == 'mutualinfo':
            val = mutual_info_regression_partial(X,Y)
            val = val/np.max(val)
            return self.get_best(val, pass2_features)
        else:
            raise(Exception('Unknown method'))

    def chi2_selector(self,set_score,dict_mat,max_feats_pass2):
        med_score = np.median(set_score)
        new_score = set_score
        new_score[set_score < med_score] = 0
        new_score[set_score >= med_score] = 1

        ch2 = SelectKBest(chi2, k=max_feats_pass2)
        ch2.fit(dict_mat,new_score)
        good_cols = ch2.get_support(indices=True)
        return good_cols

    def fisher_selector(self,set_score,dict_mat,max_feats_pass2):
        med_score = np.median(set_score)
        new_score = set_score
        new_score[set_score < med_score] = 0
        new_score[set_score >= med_score] = 1

        new_score_1 = new_score == 1
        new_score_0 = new_score == 0

        fish_vals = np.empty(dict_mat.shape[1])
        fish_vals[:] = np.nan

        for col_num in range(0, dict_mat.shape[1]):

            # loop_vec = np.squeeze(np.asarray(dict_mat[:, col_num]))
            # good_loop_vec = loop_vec[new_score == 1]
            # bad_loop_vec = loop_vec[new_score == 0]
            # good_loop_present = len(good_loop_vec[good_loop_vec > 0])
            # good_loop_missing = len(good_loop_vec[good_loop_vec == 0])
            # bad_loop_present = len(bad_loop_vec[bad_loop_vec > 0])
            # bad_loop_missing = len(bad_loop_vec[bad_loop_vec == 0])

            loop_vec = dict_mat[:, col_num]
            good_loop_vec = loop_vec[new_score_1]
            bad_loop_vec = loop_vec[new_score_0]
            good_loop_present = np.sum(good_loop_vec != 0)
            good_loop_missing = np.sum(good_loop_vec == 0)
            bad_loop_present = np.sum(bad_loop_vec != 0)
            bad_loop_missing = np.sum(bad_loop_vec == 0)

            fish_vals[col_num] = pvalue(good_loop_present, bad_loop_present, good_loop_missing, bad_loop_missing).two_tail

        cutoff = 1
        if (len(fish_vals) > max_feats_pass2):
            cutoff = sorted(fish_vals)[max_feats_pass2]
        good_cols = np.asarray([num for num in range(0, dict_mat.shape[1]) if fish_vals[num] <= cutoff])
        return good_cols

    # tf-idf weighted transformer for document embedding
    class TfidfEmbeddingVectorizer(object):
        def __init__(self,word2vec,dim):
            self.word2vec = word2vec
            self.word2weight = None
            self.dim = dim

        def fit(self, X,y=None):
            tfidf = TfidfVectorizer(analyzer=lambda x: x)
            tfidf.fit(X)
            # if a word was never seen - it must be at least as infrequent
            # as any of the known words - so the default idf is the max of
            # known idf's
            max_idf = max(tfidf.idf_)
            self.word2weight = defaultdict(lambda: max_idf,[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
            return self

        def transform(self, X,y=None):
            return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

    def main(self,data_x,data_y,Params,x_meta=None,x_custom=None,print_info=False,text_IDs=None,stringkernels = None):

        if Params['Algorithm'][0]=='SEQUENCE':
            self.analysis_type = 'SEQUENCE'
        else:
            self.analysis_type = 'BOW'

        self.transformers=[]
        self.text_types = Params['TextTypes']
        self.feature_type = Params['FeatureMethod']
        if self.feature_type is None:
            self.feature_type=[]

        # custom text features
        if 'CUSTOM' in self.feature_type:
            assert x_meta != None, 'Customdata not set!'
            if print_info:
                start_time = time.time()
                print('... adding (custom) count measures', end='')
            x = np.array(x_custom[1], dtype=float)
            x_label = x_custom[0]
            X_custom = x
            X_custom_columns = x_label
            if print_info:
                end_time = time.time()
                print(' ... done (%1.1fs)' % (end_time - start_time))

        # tag features
        if 'TAGS' in self.feature_type:
            assert x_meta != None, 'Metadata not set!'
            if print_info:
                start_time = time.time()
                print('... adding metainfo', end='')
            x = np.array(x_meta[1], dtype=float)
            x_label = x_meta[0]
            X_tags = x
            X_tags_columns = x_label
            if print_info:
                end_time = time.time()
                print(' ... done (%1.1fs)' % (end_time - start_time))

        if self.analysis_type=='SEQUENCE':

            from keras.preprocessing import sequence
            # convert text to index sequences, returns
            # data = text x sentence x word
            XX = []
            XX_columns=[]
            if 'CUSTOM' in self.feature_type:
                XX.append(X_custom)
                XX_columns.append(X_custom_columns)

            if 'TAGS' in self.feature_type:
                XX.append(X_tags)
                XX_columns.append(X_tags_columns)

            if len(XX)>0:
                XX = np.concatenate(tuple(XX), axis=1)
                self.final_data_scaler = get_scaler(self.FEATURE_SCALER)
                XX = self.final_data_scaler.fit_transform(XX)

            X = data_x[self.text_types[0]]
            max_sequence = max([len(x) for x in X])

            if Params['Algorithm'][1]['algorithm'] == 'FASTTEXT':

                self.sequence_model_type = 'FASTTEXT'

                X = [x[0:np.minimum(max_sequence, len(x))] for x in X]

                # get all words that appeared at least in two articles
                transformer=CountVectorizer(tokenizer=lambda x: x,preprocessor=lambda x:x,max_df=1.0,min_df=2,max_features = 50000,ngram_range=(1,1))
                transformer.fit(X)

                for k1 in range(0,len(X)):
                    for k2 in range(0,len(X[k1])):
                        token = X[k1][k2]
                        if token not in transformer.vocabulary_:
                            X[k1][k2]='RARE_WORD'

                self.transformer=CountVectorizer(tokenizer=lambda x: x,preprocessor=lambda x:x,max_df=1.0,min_df=2,max_features = 100000,ngram_range=(1,Params['Algorithm'][1]['ngram']))
                X_mat = self.transformer.fit_transform(X)

                self.sequence_vocabulary = {key : (val+1) for key,val in self.transformer.vocabulary_.items()} # additional tokens for empty and unknown word

                assert 'PADDED_WORD' not in self.transformer.vocabulary_

                self.sequence_vocabulary['PADDED_WORD']=0

                ind2word = ['' for x in range(0, len(self.sequence_vocabulary))]
                for word in self.sequence_vocabulary.keys():
                    ind2word[self.sequence_vocabulary[word]] = word

                maxlen_words=0
                X=[]
                for row in range(X_mat.shape[0]):
                    tokens=[-1 for _ in range(np.sum(X_mat[row,:]))]
                    ind = np.argwhere(X_mat[row,:]>0)
                    k=0
                    for _,col in ind:
                        for _ in range(0,X_mat[row,col]):
                            tokens[k]=col+1
                            k+=1
                    maxlen_words = np.maximum(maxlen_words,len(tokens))
                    X.append(tokens)

                #maxlen_words = np.minimum(Params['Algorithm'][1]['max_sequence'], maxlen_words)
                self.maxlen=maxlen_words

                # print('Pad sequences (samples x time)')
                #X = [sequence.pad_sequences(X, maxlen=self.maxlen)]
                # x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

                # print('Pad sequences (samples x time)')
                if len(XX)>0:
                    X = [sequence.pad_sequences(X, maxlen=self.maxlen),XX]
                    X_columns = [self.sequence_vocabulary,XX_columns]
                else:
                    X = [sequence.pad_sequences(X, maxlen=self.maxlen)]
                    X_columns = [self.sequence_vocabulary]

            else:

                max_sequence = np.minimum(max_sequence, Params['Algorithm'][1]['max_seq_length'])
                X = [x[0:np.minimum(max_sequence, len(x))] for x in X]

                # get all words that appeared at least in two articles
                transformer=CountVectorizer(tokenizer=lambda x: x,preprocessor=lambda x:x,max_df=1.0,min_df=2,max_features = 20000,ngram_range=(1,1))
                transformer.fit(X)
                self.sequence_vocabulary = {key : (val+2) for key,val in transformer.vocabulary_.items()} # additional tokens for empty and unknown word

                assert 'UNKNOWN_WORD' not in self.sequence_vocabulary
                assert 'PADDED_WORD' not in self.sequence_vocabulary

                self.sequence_vocabulary['PADDED_WORD']=0
                self.sequence_vocabulary['RARE_WORD']=1

                # compute mean and mean norm of all word vectors
                sumvec=0.0
                sumnorm=0.0
                for k,word in enumerate(self.external_features.word_embedding):
                    vec = self.external_features.word_embedding[word]
                    sumvec+=vec
                    sumnorm += np.linalg.norm(vec)
                    if k>10000:
                        break
                vec_mean = 0*sumvec/(k+1)
                vec_norm = sumnorm/(k+1)
                EMBEDDING_DIM = len(vec_mean)

                def get_random_vec():
                    # generate random vector with same mean and norm as embeddings on avarage
                    a = 2*np.random.rand(EMBEDDING_DIM) - 1
                    #a = a + vec_mean
                    a = (a / np.linalg.norm(a)) * vec_norm
                    return a

                word_embedding = {}
                word_embedding['RARE_WORD'] = get_random_vec()
                word_embedding['PADDED_WORD'] = 0

                X = {'FLAT':[],'SENTENCES':[]} # index matrix with splitted sentences

                maxlen_doc=0
                maxlen_sent=0
                maxlen_words=0
                unknown_words=set()
                total_words = [0,0]

                # convert tokens to indices, keep sentences
                for k1,text in enumerate(data_x[self.text_types[0] + '_SENTENCES']):
                    X['SENTENCES'].append([])
                    words = 0
                    maxlen_doc = np.maximum(maxlen_doc,len(text))
                    for k2,sent in enumerate(text):
                        X['SENTENCES'][-1].append([])
                        words+=len(sent)
                        maxlen_words = np.maximum(maxlen_words,words)
                        maxlen_sent = np.maximum(maxlen_sent, len(sent))
                        if len(sent)==maxlen_sent:
                            maxlen_sent_example = sent
                        for k3,word in enumerate(sent):
                            lemma_word = data_x['LEMMA_SENTENCES'][k1][k2][k3]
                            lemma_word = lemma_word.replace('#','')

                            total_words[0]+=1
                            # three cases: (1) in list and dictionary (2) in dictionary (3) nowhere
                            vec = None
                            if word in self.external_features.word_embedding:
                                # word has embeddings
                                vec = self.external_features.word_embedding[word]
                            elif lemma_word in self.external_features.word_embedding:
                                # lemma has embedding, use that instead
                                vec = self.external_features.word_embedding[lemma_word]
                                word_embedding[word] = vec
                            if word in self.sequence_vocabulary:
                                # word must have embedding, even a random one
                                if vec is None:
                                    vec = get_random_vec()  # null vector
                                    word_embedding[word]=vec
                            else:
                                if vec is None: # word not in vocabulary and no embedding, mark as unknown
                                    word = 'RARE_WORD'
                                    total_words[1]+=1
                                    unknown_words.add(word)
                                else: # word not in vocabulary but has embedding,
                                    self.sequence_vocabulary[word]=len(self.sequence_vocabulary)

                            word_index = Params['sequence_vocabulary'][word]
                            X['SENTENCES'][-1][-1].append(word_index)

                    X['FLAT'].append(list(itertools.chain.from_iterable(X['SENTENCES'][-1])))

                X['FLAT'] = sequence.pad_sequences(X['FLAT'], maxlen=maxlen_words)

                assert (total_words[1]/total_words[0])<0.10,'over 10% of words (tokens) are unknown!'

                vals = sorted([self.sequence_vocabulary[key] for key in self.sequence_vocabulary])

                assert np.max(vals)+1==len(vals)

                self.maxlen_words=maxlen_words
                self.maxlen_doc=maxlen_doc
                self.maxlen_sent=maxlen_sent
                self.max_unique_words = len(self.sequence_vocabulary)

                # vals = sorted([self.transformer.vocabulary_[key] for key in self.transformer.vocabulary_.keys()])
                W = np.zeros((self.max_unique_words, EMBEDDING_DIM),dtype=np.float32)
                W.fill(np.nan)

                ind2word = ['' for x in range(0,len(self.sequence_vocabulary))]
                for word in self.sequence_vocabulary.keys():
                    W[self.sequence_vocabulary[word]]=word_embedding[word]
                    ind2word[self.sequence_vocabulary[word]]=word

                #for k,word in Params['sequence_vocabulary']

                #
                if 0:
                    data_x_check=[]
                    for k1 in range(0,len(X['FLAT'])):
                        data_x_check.append([])
                        for k2 in range(0,len(X['FLAT'][k1])):
                            data_x_check[k1].append(ind2word[X['FLAT'][k1][k2]])

                Params['W_embedding_matrix'] = W
                Params['max_document_sentences'] = maxlen_doc
                Params['max_sentence_words'] = maxlen_sent
                Params['max_words_in_doc'] = maxlen_words
                Params['max_unique_words'] = self.max_unique_words
                self.word_embedding = word_embedding

                if len(XX) > 0:
                    X = [X, XX]
                    X_columns = 'sequence data (up to %i words) + metadata (% items)' % (maxlen_words,XX.shape[1])
                else:
                    X = [X, ]
                    X_columns = 'sequence data (up to %i words)' % maxlen_words

        elif self.analysis_type=='BOW':

            # feature selection type, only for BOW algorithms (not including fasttext)
            self.featureselection = Params['FeatureSelection']

            if not isinstance(self.text_types, list) and not isinstance(self.text_types, tuple):
                self.text_types = [self.text_types]

            if print_info:
                print('\nBuilding and transforming features (training phase)')

            X = []
            X_columns = []
            for feature in self.feature_type:
                for text_type in self.text_types:
                    if feature == 'TFIDF':
                        if text_type=='POS':
                            ngram_range = self.POS_ngram
                        else:
                            ngram_range = Params['TFIDF_ngram']
                        if print_info:
                            start_time = time.time()
                            print('... adding TF-IDF (%s, ngram=%s)' % (text_type,str(ngram_range)),end='')
                        self.transformers.append((TfidfVectorizer(tokenizer=lambda x: x,preprocessor=lambda x: x,max_df=1.0,min_df=2,use_idf=True,max_features=Params['pass1_features'],ngram_range=ngram_range),text_type))
                        x = self.transformers[-1][0].fit_transform(data_x[text_type]).todense()
                        X.append(x)
                        x=self.transformers[-1][0].get_feature_names()
                        x = ['term=' + y + ',type=%s+TFIDF' % text_type for y in x]
                        X_columns.append(x)
                        if print_info:
                            end_time = time.time()
                            print(' ... done (%1.1fs)' % (end_time - start_time))
                    elif feature == 'BOW':
                        if text_type=='POS':
                            ngram_range = self.POS_ngram
                        else:
                            ngram_range = Params['BOW_ngram']
                        if print_info:
                            start_time = time.time()
                            print('... adding BOW (%s, ngram=%s)' % (text_type,str(ngram_range)),end='')
                        self.transformers.append((CountVectorizer(tokenizer=lambda x: x,preprocessor=lambda x: x,max_df=1.0,min_df=2, max_features=Params['pass1_features'], ngram_range=ngram_range,dtype=np.float32),text_type))
                        x = self.transformers[-1][0].fit_transform(data_x[text_type]).todense()
                        X.append(x)
                        x=self.transformers[-1][0].get_feature_names()
                        x=['term='+y+',type=%s+BOW'%text_type for y in x]
                        X_columns.append(x)
                        if print_info:
                            end_time = time.time()
                            print(' ... done (%1.1fs)' % (end_time - start_time))
                    else:
                        pass

            # do feature selection for individual BOW features or all of them
            is_selected = False
            if len(X)>0 and self.featureselection != None and self.featureselection[1]!='global':
                is_selected=True
                if print_info:
                    start_time = time.time()
                    print('... doing feature selection (type=%s)' % str(self.featureselection),end='')
                self.selected_columns=[]
                if self.featureselection[1]=='single':
                    for k,x in enumerate(X):
                        self.selected_columns.append(self.column_selector(x,data_y.copy(),Params['FeatureSelection'][0],Params['FeatureSelection'][2]))
                        X[k] = np.take(x,indices=self.selected_columns[-1],axis=1)
                        X_columns[k] = [X_columns[k][kk] for kk in self.selected_columns[-1]]
                elif self.featureselection[1]=='all':
                    X = np.concatenate(tuple(X), axis=1)
                    self.selected_columns = self.column_selector(X,data_y.copy(),Params['FeatureSelection'][0],Params['FeatureSelection'][2])
                    X = [np.take(X, indices=self.selected_columns, axis=1)]
                    X_columns = list(itertools.chain.from_iterable(X_columns))
                    X_columns = [list([X_columns[kk] for kk in self.selected_columns])]
                else:
                    raise(Exception('featureselection property must be single or all!'))
                if print_info:
                    end_time = time.time()
                    print(' ... done (%1.1fs)' % (end_time - start_time))

            # tf-ifd weighted document embedding
            if 'EMBEDDING' in self.feature_type:
                if print_info:
                    start_time = time.time()
                    print('... adding embedded document vectors (dim %i) with tf-idf scaling' % self.external_features.embedding_dim,end='')
                self.embedding_type = Params['EMBEDDING_type']
                if self.embedded_transformer == None:
                    self.embedded_transformer = self.TfidfEmbeddingVectorizer(
                        self.external_features.word_embedding,
                        self.external_features.embedding_dim)
                    if self.embedding_type == 'LEMMA':
                        self.embedded_transformer.fit(replace_hash(data_x[self.embedding_type]))
                    else:
                        self.embedded_transformer.fit(data_x[self.embedding_type])
                if self.embedding_type=='LEMMA':
                    x=self.embedded_transformer.transform(replace_hash(data_x[self.embedding_type]))#.astype(np.float32)
                else:
                    x = self.embedded_transformer.transform(data_x[self.embedding_type])  # .astype(np.float32)
                X.append(x)
                X_columns.append(['emb%i_%3.0f'%(self.external_features.embedding_dim,kk) for kk in range(1,self.external_features.embedding_dim+1)])
                if print_info:
                    end_time = time.time()
                    print(' ... done (%1.1fs)' % (end_time - start_time))

            if 'CUSTOM' in self.feature_type:
                X.append(X_custom)
                X_columns.append(X_custom_columns)

            if 'TAGS' in self.feature_type:
                X.append(X_tags)
                X_columns.append(X_tags_columns)

            X = np.concatenate(tuple(X),axis=1)

            X_columns = list(itertools.chain.from_iterable(X_columns))

            # do global feature selection
            if self.featureselection != None and self.featureselection[1]=='global':
                assert is_selected == False,'Trying selection twice!'
                if print_info:
                    start_time = time.time()
                    print('... doing feature selection (type=%s)' % str(self.featureselection),end='')
                self.selected_columns = self.column_selector(X,data_y.copy(),Params['FeatureSelection'][0],Params['FeatureSelection'][2])
                X = np.take(X, indices=self.selected_columns, axis=1)
                X_columns = list([X_columns[kk] for kk in self.selected_columns])

                if print_info:
                    end_time = time.time()
                    print(' ... done (%1.1fs)' % (end_time - start_time))

            assert X.shape[1] == len(X_columns),'X and X_labels have different size! BUG!'

            self.final_data_scaler = get_scaler(self.FEATURE_SCALER)

            if self.FEATURE_SCALER is not 'StandardScaler':
                temp_scaler = get_scaler('StandardScaler')
                temp_scaler.fit(X)
                self.feature_scale_multiplier = temp_scaler.scale_
                X = self.final_data_scaler.fit_transform(X)
                self.feature_scale_multiplier = self.feature_scale_multiplier /self.final_data_scaler.scale_
            else:
                self.feature_scale_multiplier = np.ones(X.shape[1])
                X = self.final_data_scaler.fit_transform(X)

            if Params['Algorithm'][0] == 'StringKernel':
                self.analysis_type = 'BOW_StringKernel'
                self.ngramMaxLength = Params['Algorithm'][1]['ngram']
                X_stringkernel = get_stringkernel(stringkernels, self.text_types, text_IDs, self.ngramMaxLength)
                X_columns = ['String kernels for %s' % " ".join(self.text_types)]
                self.kerneldata_Y = X
                self.kernelfunction = get_kernel(Params['Algorithm'][1]['kerneltype'])
                self.stringkernel_ratio = Params['Algorithm'][1]['stringkernel_ratio']
                X = (self.stringkernel_ratio * X_stringkernel) + (1.0 - self.stringkernel_ratio) * self.kernelfunction(X=X, Y=None)

            self.final_features = X.shape[1]

        else:
            raise(Exception('Unknown analysis type (should be sequence or classical)'))

        return X,X_columns,Params

        #from gensim.models.doc2vec import TaggedDocument,Doc2Vec

