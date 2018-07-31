import numpy as np
import itertools
import random
import os
import sys
from scipy.stats import spearmanr

# This module contains various helper functions to compute, manipulate, plot and write results
# New functions are added as necessary, some might remain unused

def most_common(lst):
    lst=list(lst)
    item = max(set(lst), key=lst.count)
    return (item,lst.count(item))

def convert_to_label(one_hot):
    if len(one_hot.shape)>1 and one_hot.shape[1]>1:
        y=[]
        for row in range(0,one_hot.shape[0]):
            k=np.argmax(one_hot[row,:])
            y.append(k+1)
        return np.array(y)
    else:
        return one_hot
def random_sample_pick(PREPROCESSED_X,PREPROCESSED_Y,culling_rate):
    import time
    x = list(PREPROCESSED_X.keys())
    np.random.seed(seed=int(time.time()))
    keep_texts = [x[y] for y in np.random.choice(len(PREPROCESSED_X), int((1.0-culling_rate) * len(PREPROCESSED_X)), replace=False)]
    PREPROCESSED_X = {x:PREPROCESSED_X[x] for x in keep_texts}
    PREPROCESSED_Y = {dim:{x:PREPROCESSED_Y[dim][x] for x in keep_texts} for dim in PREPROCESSED_Y.keys()}
    return PREPROCESSED_X,PREPROCESSED_Y

# do some processing for parameters (extract ngram and text type information)
def process_params(Params):

    Params['BOW_ngram'] = None
    Params['TFIDF_ngram'] = None

    if Params['Algorithm'][0] in ('ElasticNet_multitarget', 'RandomForest_multitarget', 'MLP_multitarget','Ridge_multitarget'):
        Params['Algorithm'] = list(Params['Algorithm'])
        Params['Algorithm'][0] = Params['Algorithm'][0][0:-12]
        Params['is_multitarget'] = True
    else:
        Params['is_multitarget'] = False

    if Params['Algorithm'][0]=='SEQUENCE':
        pass#Params['Scaler']='MaxAbsScaler' # only applied to custom features
    else:
        Params['FeatureMethod'] = list(Params['FeatureMethod']) # make sure its a list and not tuple
        for k in range(0,len(Params['FeatureMethod'])):
            if Params['FeatureMethod'][k][0:3]=='BOW':
                assert Params['FeatureMethod'][k][3] == '_', 'Incorrect BOW ngram setting, should contain _'
                Params['BOW_ngram'] = (1,int(Params['FeatureMethod'][k][4]))
                Params['FeatureMethod'][k] = 'BOW'
            elif Params['FeatureMethod'][k][0:5]=='TFIDF':
                assert Params['FeatureMethod'][k][5] == '_', 'Incorrect TFIDF ngram setting, should contain _'
                Params['TFIDF_ngram'] = (1,int(Params['FeatureMethod'][k][6]))
                Params['FeatureMethod'][k] = 'TFIDF'
            elif Params['FeatureMethod'][k][0:9] == 'EMBEDDING':
                assert Params['FeatureMethod'][k][9] == '_', 'Incorrect EMBEDDING setting, should contain _'
                Params['EMBEDDING_type'] = Params['FeatureMethod'][k][10:]
                assert Params['EMBEDDING_type'] in ('NORMAL','LEMMA'),'Embedding type must be LEMMA or NORMAL'
                Params['FeatureMethod'][k] = 'EMBEDDING'


    return Params

def print_parameters(params):
    keys = list(params.keys())
    s = 'Using parameters: '
    for key in keys:
        s += key + '='
        if not(isinstance(params[key],tuple) or isinstance(params[key],list)):
            temp=[params[key]]
        else:
            temp=params[key]
        for elem,_ in enumerate(temp):
            val = temp[elem]
            if isinstance(val,str):
                s += '\'%s\', ' % (val)
            elif isinstance(val,float) and np.round(val)==val:
                s+='\'%i\', ' % (int(val))
            else:
                s += '\'%s\', ' % (str(val))
    return s[0:-2]

# create all combinations of given parameters, returned list is randomized
# keyword Algorithm gets special handling (it a list with dictionary as the second element)
# NOTE: elements in final_params are references (change one, change all)
def generate_paramset(params,seed=666,is_algo=0):
    # assumptions: each set has either one list of parameter candidates or a dictionary of multiple lists
    allkeys = list(params.keys()) # different options
    allparams={}
    for key in allkeys:
        if key == 'Algorithm' and is_algo==0:
            temp = generate_paramset(params[key][1],seed=666,is_algo=1)
            allparams[key] = [(params[key][0],x) for x in temp]
        else:
            allparams[key]=[]
            for k in range(len(params[key])):
                if is_algo:
                    allparams[key] = [(x,) for x in params[key]]
                else:
                    arr = []
                    last_item = len(params[key][k])
                    for elem in range(last_item):
                        if elem < last_item-1:
                            arr.append(params[key][k][elem])
                        else:
                            if isinstance(params[key][k][elem],tuple):
                                assert elem==last_item-1,'item is not last one!'
                                for cand in params[key][k][elem]:
                                    allparams[key].append(arr+[cand])
                            elif isinstance(params[key][k][elem],dict):
                                assert elem == last_item - 1, 'item is not last one!'
                                keys1, values1 = zip(*params[key][k][elem].items())
                                allcomb = [dict(zip(keys1, v)) for v in itertools.product(*values1)]
                                for cand in allcomb:
                                    allparams[key].append(arr + [cand])
                            else:
                                allparams[key].append(arr + [params[key][k][elem]])
    keys1, values1 = zip(*allparams.items())
    final_params = [dict(zip(keys1, v)) for v in itertools.product(*values1)]
    random.Random(seed).shuffle(final_params)
    return final_params

# convert all 1-element numerical lists to scalars
def simplify_params(params):

    def simplify(dic,allow_str=False):
        if isinstance(dic, list) or isinstance(dic, tuple):
            if len(dic) == 1:
                if allow_str:
                    dic = dic[0]
                elif dic[0] is None:
                    dic = dic[0]
                else:
                    try:
                        float(dic[0])
                        dic = dic[0]
                    except:
                        pass
        return dic

    for k in range(len(params)):
        for key in params[k].keys():
            if key == 'Algorithm': # needs special threatment, one step deeper
                for key1 in params[k][key][1].keys():
                    params[k][key][1][key1] = simplify(params[k][key][1][key1],allow_str=True)
            elif key == 'FeatureScaler': # this is always singleton
                params[k][key] = params[k][key][0]
            else:
                params[k][key]=simplify(params[k][key])
    return params

def print_results(data,dimension='all',fold='total'):
    print(str_results(data,dimension,fold),end='')

def tvalue(a):
    # std deviation
    ind = np.where(np.abs(a) > 0)[0]
    if len(ind)>=len(a)-1:
        a=np.take(a,ind)
        se = np.std(a)/np.sqrt(len(a))
        tval = np.mean(a)/se
    else:
        tval=np.nan

    return tval

def str_single_fold(data):
    return ('\ntrain: R = %f, R2 = %f, MSE = %f (null %f, ratio %f), f1 = %f (null %f)' % (
                data['train']['R'],
                data['train']['R2'],
                data['train']['MSE'],
                data['train']['MSE_null'],
                data['train']['MSE'] / data['train']['MSE_null'],
                data['train']['F1'],
                data['train']['F1_null']) + \
            '\ntest: R = %f, R2 = %f, MSE = %f (null %f, ratio %f), f1 = %f (null %f)\n' % (
                data['test']['R'],
                data['test']['R2'],
                data['test']['MSE'],
                data['test']['MSE_null'],
                data['test']['MSE'] / data['test']['MSE_null'],
                data['test']['F1'],
                data['test']['F1_null']))

def str_results(data,dimension='all',fold='total'):
    if dimension=='all' and fold=='total':
        res = ''
        for dim in data.keys():
            res+= ('\n----> FOLD %s, dimension \'%s\' <---- ' % (fold,dim) + \
              '\ntrain: R = %f, R2 = %f, MSE = %f (null %f, ratio %f), f1 = %f (null %f)' % (
                  data[dim]['train'].loc[fold, 'R'],
                  data[dim]['train'].loc[fold, 'R2'],
                  data[dim]['train'].loc[fold, 'MSE'],
                  data[dim]['train'].loc[fold, 'MSE_null'],
                  data[dim]['train'].loc[fold, 'MSE'] / data[dim]['train'].loc[fold, 'MSE_null'],
                  data[dim]['train'].loc[fold, 'F1'],
                  data[dim]['train'].loc[fold, 'F1_null']) + \
              '\ntest: R = %f, R2 = %f, MSE = %f (null %f, ratio %f), f1 = %f (null %f)\n' % (
                  data[dim]['test'].loc[fold, 'R'],
                  data[dim]['test'].loc[fold, 'R2'],
                  data[dim]['test'].loc[fold, 'MSE'],
                  data[dim]['test'].loc[fold, 'MSE_null'],
                  data[dim]['test'].loc[fold, 'MSE']/ data[dim]['test'].loc[fold, 'MSE_null'],
                  data[dim]['test'].loc[fold, 'F1'],
                  data[dim]['test'].loc[fold, 'F1_null']))
        return res
    else:
        dim=dimension
        return ('\n----> FOLD %i, dimension \'%s\' <----: ' % (fold,dim) + \
          '\ntrain: R = %f, R2 = %f, MSE = %f (null %f, ratio %f), f1 = %f (null %f)' % (
              data[dim]['train'].loc[fold, 'R'],
              data[dim]['train'].loc[fold, 'R2'],
              data[dim]['train'].loc[fold, 'MSE'],
              data[dim]['train'].loc[fold, 'MSE_null'],
              data[dim]['train'].loc[fold, 'MSE'] / data[dim]['train'].loc[fold, 'MSE_null'],
              data[dim]['train'].loc[fold, 'F1'],
              data[dim]['train'].loc[fold, 'F1_null']) + \
          '\ntest: R = %f, R2 = %f, MSE = %f (null %f, ratio %f), f1 = %f (null %f)\n' % (
              data[dim]['test'].loc[fold, 'R'],
              data[dim]['test'].loc[fold, 'R2'],
              data[dim]['test'].loc[fold, 'MSE'],
              data[dim]['test'].loc[fold, 'MSE_null'],
              data[dim]['test'].loc[fold, 'MSE'] / data[dim]['test'].loc[fold, 'MSE_null'],
              data[dim]['test'].loc[fold, 'F1'],
              data[dim]['test'].loc[fold, 'F1_null']))

def get_best_params(old_results,new_entry,k_max=30):
    new_results = {}
    if len(old_results)==0:
        was_updated = True
        for key in new_entry.keys():
            new_results[key]=[(new_entry[key]['test'].copy(),new_entry[key]['params'],),]
    else:
        was_updated = False
        for key in new_entry.keys():
            if len(old_results[key])>k_max:
                print('!!!!! WARNING: results vector has more elements than k_max, cropping !!!!!')
                old_results[key] = old_results[key][0:k_max]
            new_results[key]=old_results[key]
            old_vals = []
            rank=-1
            for k in range(0,len(old_results[key])):
                old_vals.append(old_results[key][k][0].loc['total','MSE'])
                if rank==-1 and new_entry[key]['test'].loc['total','MSE']<=old_vals[-1]:
                    rank=k
                    break
            if rank>-1:
                was_updated = True
                new_results[key]=[x for x in old_results[key][0:rank]] + [(new_entry[key]['test'].copy(),new_entry[key]['params'],),] + [x for x in old_results[key][rank:np.minimum(k_max-1,len(old_results[key]))]]
            elif len(old_results[key])<k_max:
                was_updated = True
                new_results[key]=old_results[key] + [(new_entry[key]['test'].copy(),new_entry[key]['params'])]
    return new_results,was_updated

def save_params(old_results,savepath_root,total_sets=np.nan):
    for key in old_results.keys():
        savepath = savepath_root + os.sep + key
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        with open(savepath + os.sep + '%s_results_params.txt' % key.upper(),'w',encoding='utf-8') as fout:
            fout.write('\nDIMENSION = %s, TOP %i parameters (with %i tested sets):\n' % (key,len(old_results[key]),total_sets))
            for k in range(0,len(old_results[key])):
                fout.write('______ RANK = %i ______\n' % (k+1))
                fout.write('R = %f, R2 = %f, MSE = %f (null %f, ratio %f), f1 = %f (null %f)\n' % (
                    old_results[key][k][0].loc['total', 'R'],
                    old_results[key][k][0].loc['total', 'R2'],
                    old_results[key][k][0].loc['total', 'MSE'],
                    old_results[key][k][0].loc['total', 'MSE_null'],
                    old_results[key][k][0].loc['total', 'MSE']/old_results[key][k][0].loc['total', 'MSE_null'],
                    old_results[key][k][0].loc['total', 'F1'],
                    old_results[key][k][0].loc['total', 'F1_null']))
                fout.write('all parameters: '+old_results[key][k][1]+'\n')
                if k<3 and total_sets>50: # only plot top-3 after 50 sets
                    plot_data(old_results[key][k][0],savepath + os.sep + '%s_results_plot_RANK_%i' % (key,k+1),title=key + ', rank %i, ' % (k+1))

# estimate required memory for a keras model
def get_model_memory_usage(batch_size, model):
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    #print('Your model needs %i gigabytes of memory'%gbytes)
    return gbytes

# plot and write results as files
def plot_data(result_dataframe=None,filename=None,title='',Y_pred=None,Y_real=None,Y_null=None,X_ticks=None,folds=None):
    import matplotlib.pyplot as plt

    if result_dataframe is not None:
        Y_pred = result_dataframe.loc['total', 'PREDICTED_Y']
        Y_real = result_dataframe.loc['total', 'REAL_Y']
        Y_null = result_dataframe.loc['total','NULL_Y']
        X_ticks = result_dataframe.loc['total','TEXT_IDs']
        folds = result_dataframe.loc['total', 'FOLD']

    x_label='Text ID (sample)'
    y_label='Score (rating)'

    plt.ioff()
    plt.close()
    fig = plt.figure(num=1)
    DPI = float(fig.dpi)
    fig.set_size_inches(1300 / DPI,650 / DPI)

    plt.plot(Y_real)
    Y = Y_pred
    plt.plot(Y)
    plt.plot(Y_null)
    plt.legend(('real','predicted','null'))

    if result_dataframe is not None:
        plt.title(title + 'R=%0.3f, R2=%0.3f, MSE=%0.3f (null %f, ratio %f)' % (result_dataframe.loc['total', 'R'],
                                                                         result_dataframe.loc['total', 'R2'],
                                                                         result_dataframe.loc['total', 'MSE'],
                                                                         result_dataframe.loc['total', 'MSE_null'],result_dataframe.loc['total', 'MSE']/result_dataframe.loc['total', 'MSE_null']))
        assert np.abs(np.corrcoef(result_dataframe.loc['total', 'REAL_Y'],Y)[0,1] - result_dataframe.loc['total', 'R'])<1e-6,'Plotted data and stored values for R do not match!'

    plt.xticks(list(range(0,len(Y))),X_ticks, rotation=-45,fontsize=4)
    plt.xlabel('Text ID')
    plt.ylabel('Rating bias')

    plt.tight_layout(rect=(0.05,0.07,0.98,1.00)) # [left, bottom, right, top] in normalized (0, 1) figure coordinates.
    dx = len(Y)*0.011
    plt.xlim([-dx,len(Y)+dx])
    #plt.draw()
    #plt.show(block=False)
    #plt.pause(0.02)

    plt.savefig(filename + '.png')
    plt.savefig(filename + '.pdf')
    plt.close()

    difference = Y - Y_real

    # indices for sorted errors by decreasing magnitude
    ind = np.argsort(-np.abs(difference))

    IDs = X_ticks

    with open(filename + '.txt','w',encoding='utf-8') as f:
        f.write('Predicted model errors in decreasing order for %i samples (%s)\n' % (len(ind),title))
        f.write('sample_rank\ttext_ID\treal\tpred-real\t(pred-real)^2\tfold\n')
        for k,i in enumerate(ind):
            f.write('%i\t%s\t%f\t%f\t%f\t%i\n' % (k+1,IDs[i],Y_real[i],difference[i],difference[i]**2,folds[i]))

# compute average result over all folds
def pool_results(RESULTS,folds=-1):
    if folds>-1:
        for k_dim,dimension in enumerate(RESULTS.keys()):
            for type in ('train','test'):
                y_real=RESULTS[dimension][type].loc[1,'REAL_Y']
                y_pred=RESULTS[dimension][type].loc[1,'PREDICTED_Y']
                y_null=RESULTS[dimension][type].loc[1,'NULL_Y']
                y_fold=RESULTS[dimension][type].loc[1,'FOLD']
                text_ids = RESULTS[dimension][type].loc[1,'TEXT_IDs']
                for k_fold in range(2,folds+1):
                    y_pred=np.concatenate((y_pred,RESULTS[dimension][type].loc[k_fold,'PREDICTED_Y']))
                    y_real=np.concatenate((y_real,RESULTS[dimension][type].loc[k_fold, 'REAL_Y']))
                    y_null=np.concatenate((y_null,RESULTS[dimension][type].loc[k_fold, 'NULL_Y']))
                    y_fold =np.concatenate((y_fold,RESULTS[dimension][type].loc[k_fold, 'FOLD']))
                    text_ids = text_ids + RESULTS[dimension][type].loc[k_fold,'TEXT_IDs']
                if type=='test':
                    assert len(text_ids)==len(set(text_ids)),'Text IDs in pooled test set is not correct!'
                R = (np.corrcoef(y_real,y_pred))[0, 1]
                sR = spearmanr(y_pred,y_real).correlation
                R2 = (R) ** 2
                MSE = np.mean((y_real-y_pred)**2)
                MSE_null = np.mean((y_real-y_null)**2) #  mean_squared_error(y_real,y_null)
                RESULTS[dimension][type].loc['total','PREDICTED_Y'] = y_pred
                RESULTS[dimension][type].loc['total', 'REAL_Y'] = y_real
                RESULTS[dimension][type].loc['total', 'NULL_Y'] = y_null
                RESULTS[dimension][type].loc['total', 'R'] = R
                RESULTS[dimension][type].loc['total', 'spearmanR'] = sR
                RESULTS[dimension][type].loc['total', 'R2'] = R2
                RESULTS[dimension][type].loc['total', 'FOLD'] = y_fold
                RESULTS[dimension][type].loc['total', 'MSE'] = MSE
                RESULTS[dimension][type].loc['total', 'MSE_null'] = MSE_null
                RESULTS[dimension][type].loc['total','TEXT_IDs'] = text_ids
    return RESULTS

# check folder and create if not present
def isfolder(folder):
    if os.path.isdir(folder):
        pass
    else:
        os.makedirs(folder)
    if not folder[-1] == '/':
        folder+='/'
    assert os.path.isdir(folder), 'Failed to create folder %s' % folder

# sort prediction errors by difference from actual data
def get_major_errors(RESULTS):
    type = 'test'
    for k_dim, dimension in enumerate(RESULTS.keys()):
        vals = RESULTS[dimension][type].loc['total', 'PREDICTED_Y'] - RESULTS[dimension][type].loc['total', 'REAL_Y']
        ind = np.argsort(vals)

# format list as string
def list_to_string(mylist,mylist2=None):
    s=''
    for k,x in enumerate(mylist):
        if mylist2 is not None:
            s += ' %0.2f (%0.2f),' % (x,mylist2[k])
        else:
            s+=' %0.2f,' % x
    s=s.strip()
    return s[0:-1]

# do stratified k-fold for regression using linear binning. Must have #samples/#folds >= #folds, otherwise error is produced
def Kfold(Y,data_type,n_splits=10,random_state=None):
    from sklearn.model_selection import StratifiedKFold

    if random_state is None:
        random_state=n_splits
    y_train={}
    y_test={}
    for key in Y:
        if n_splits==1:
            y_test[key] = [[]]
            y_train[key] = [list(Y[key].copy())]
        elif n_splits>1 and n_splits<20:

            y = Y[key].copy()
            y_keys = list(y.keys())
            y_vals = [y[z] for z in y_keys]
            n_all = len(y_vals)
            assert n_splits*n_splits<n_all,'too many bins compared to samples!'
            if data_type=='classification':
                assert len(set(y_vals))>10,'Too many classes (over 10)!'
                y_vals_binned = y_vals
            else:
                y_vals=np.sort(y_vals)
                ind = np.linspace(-1,n_all-1,n_splits+1)
                ind = [int(np.round(x)) for x in ind[1:-1]]
                assert len(ind)==n_splits-1,'Bad binning!'
                bins=[y_vals[0]-0.0001]+[y_vals[k] for k in ind] + [y_vals[-1]+0.0001]
                y_vals_binned = np.digitize(y_vals, bins)
            y_train[key] = []
            y_test[key] = []

            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_ind, test_ind in kf.split(X=0*y_vals,y=y_vals_binned):
                y_train[key].append([y_keys[x] for x in train_ind])
                y_test[key].append([y_keys[x] for x in test_ind])
                assert len(set(y_train[key][-1]) & set(y_test[key][-1]))==0,'train and test sets intersect!'
                assert len(y_train[key][-1]) + len(y_test[key][-1])==n_all,'train and test split does contain all samples!'

            assert len(y_train[key])==n_splits and len(y_test[key])==n_splits,'fold count does not match!'

            testset=[]
            for y in y_test[key]:
                testset = testset + y
            assert len(testset)==len(set(testset))==n_all,'Indexing failed, vector sizes not equal!'

        else:
            raise(Exception('bad n_splits value entered!'))

    return y_train, y_test

# create balanced "smart" binning for classification
def do_binning(PREPROCESSED_Y,bins):
    prc = 100/bins
    class_ids = []
    for dimension in PREPROCESSED_Y.keys():
        key_list = PREPROCESSED_Y[dimension].keys()
        y=np.array([PREPROCESSED_Y[dimension][x] for x in key_list])
        new_y = y.copy()
        lower = np.min(y)-1e-6
        for k in range(1,bins):
            cut = np.percentile(y,k*prc)
            new_y[(y>=lower) & (y < cut)] = k
            lower=cut
            class_ids.append('class_%i_lims%0.2f-%0.2f' % (k,lower,cut))
        new_y[y >= lower] = k+1
        class_ids.append('class_%i_lims%0.2f-%0.2f' % (k+1, lower,np.max(y)))
        for k,x in enumerate(key_list):
            PREPROCESSED_Y[dimension][x]=new_y[k]
    return PREPROCESSED_Y,class_ids

# unwrap dictionary, should be replaced with a pre-build method (in itertools?)...
def extract_dict_data(mydict,keys):
    result0 = {x: mydict[x] for x in keys}
    result1 = {}
    if len(keys)>0:
        for type in list(result0[next(iter(result0))].keys()):
            result1[type] = []
            for key in result0:
                result1[type].append(result0[key][type])
    return result1

# one sample Wilcoxon sign-rank exact test for 2 - 20 samples (table from Matlab)
# This is a quick and dirty solution, did not find an equivalent Python implementation...
def signrank_test(vals):

    vals = np.array(vals)
    vals=vals[vals!=0]
    n=len(vals)
    if n<3:
        return 1.0
    pos = np.sum(vals>0)
    neg = n-pos
    m = np.minimum(pos,neg)

    assert 2<n<21,'valid number of samples is [3,20]'

    pval = {}
    pval[2] = (1,1,)
    pval[3] = (2.500000e-01, 1.000000e+00, 1.000000e+00,)
    pval[4] = (1.250000e-01, 6.250000e-01, 1.000000e+00,)
    pval[5] = (6.250000e-02, 3.750000e-01, 1.000000e+00, 1.000000e+00,)
    pval[6] = (3.125000e-02, 2.187500e-01, 6.875000e-01, 1.000000e+00,)
    pval[7] = (1.562500e-02, 1.250000e-01, 4.531250e-01, 1.000000e+00, 1.000000e+00,)
    pval[8] = (7.812500e-03, 7.031250e-02, 2.890625e-01, 7.265625e-01, 1.000000e+00,)
    pval[9] = (3.906250e-03, 3.906250e-02, 1.796875e-01, 5.078125e-01, 1.000000e+00, 1.000000e+00,)
    pval[10] = (1.953125e-03, 2.148438e-02, 1.093750e-01, 3.437500e-01, 7.539063e-01, 1.000000e+00,)
    pval[11] = (9.765625e-04, 1.171875e-02, 6.542969e-02, 2.265625e-01, 5.488281e-01, 1.000000e+00, 1.000000e+00,)
    pval[12] = (4.882813e-04, 6.347656e-03, 3.857422e-02, 1.459961e-01, 3.876953e-01, 7.744141e-01, 1.000000e+00,)
    pval[13] = (2.441406e-04, 3.417969e-03, 2.246094e-02, 9.228516e-02, 2.668457e-01, 5.810547e-01, 1.000000e+00, 1.000000e+00,)
    pval[14] = (1.220703e-04, 1.831055e-03, 1.293945e-02, 5.737305e-02, 1.795654e-01, 4.239502e-01, 7.905273e-01, 1.000000e+00,)
    pval[15] = (6.103516e-05, 9.765625e-04, 7.385254e-03, 3.515625e-02, 1.184692e-01, 3.017578e-01, 6.072388e-01, 1.000000e+00, 1.000000e+00,)
    pval[16] = (3.051758e-05, 5.187988e-04, 4.180908e-03, 2.127075e-02, 7.681274e-02, 2.101135e-01, 4.544983e-01, 8.036194e-01, 1.000000e+00,)
    pval[17] = (1.525879e-05, 2.746582e-04, 2.349854e-03, 1.272583e-02, 4.904175e-02, 1.434631e-01, 3.323059e-01, 6.290588e-01, 1.000000e+00, 1.000000e+00,)
    pval[18] = (7.629395e-06, 1.449585e-04, 1.312256e-03, 7.537842e-03, 3.088379e-02, 9.625244e-02, 2.378845e-01, 4.806824e-01, 8.145294e-01, 1.000000e+00,)
    pval[19] = (3.814697e-06, 7.629395e-05, 7.286072e-04, 4.425049e-03, 1.921082e-02, 6.356812e-02, 1.670685e-01, 3.592834e-01, 6.476059e-01, 1.000000e+00, 1.000000e+00,)
    pval[20] = (1.907349e-06, 4.005432e-05, 4.024506e-04, 2.576828e-03, 1.181793e-02, 4.138947e-02, 1.153183e-01, 2.631760e-01, 5.034447e-01, 8.238029e-01, 1.000000e+00,)

    return pval[n][m]

# helper function to extract ngrams
def get_ngrams(tokens, min_n, max_n):
    """
    Generates ngrams(word sequences of fixed length) from an input token sequence.
    tokens is a list of words.
    min_n is the minimum length of an ngram to return.
    max_n is the maximum length of an ngram to return.
    returns a list of ngrams (words separated by a space)
    """
    all_ngrams = list()
    n_tokens = len(tokens)
    for i in range(n_tokens):
        for j in range(i + min_n, min(n_tokens, i + max_n) + 1):
            all_ngrams.append(" ".join(tokens[i:j]))
    return all_ngrams

# helper function to add ngrams
def add_ngrams(mydict,sentence):
    """
    Generates a count of the number of times each unique item appears in a list
    """
    ngrams = get_ngrams(sentence,2,3)
    for ngram in ngrams:
        if ngram in mydict:
            mydict[ngram]+=1
        else:
            mydict[ngram]=1
    return mydict

# generate good POS ngrams from a given high-quality text, e.g., book texts
def generate_good_POS():
    filepath = r'D:/data/turkunlp_crawler'
    files=[]
    import os
    for file in os.listdir(filepath ):
        if file.endswith(".conllu"):
            files.append(os.path.join(filepath, file))

    #files=[files[0]]

    sent_lengths = []
    pos_ngrams={}

    for file in files:
        print('Reading file %s' % file)
        with open(file,'r',encoding='utf-8') as f:
            n_word = 0
            n_sent = 0
            sentence=[]
            for linenum, line in enumerate(f.readlines()):
                # print(line)
                line = line.split(sep='\t')
                if len(line) > 1 and line[0].isnumeric():
                    n_word += 1
                    sentence.append(line[3])
                else:
                    if len(sentence) > 1:
                        pos_ngrams = add_ngrams(pos_ngrams,sentence)
                        sent_lengths.append(len(sentence))
                        n_sent += 1
                        if n_sent % 50000 == 0:
                            print('... %s sentences processed (%i lines)' % (n_sent, n_word))
                    sentence = []

    med_sentence_length=np.median(sent_lengths)
    mean_sentence_length=np.mean(sent_lengths)
    print('median sentence length is: %f\n' % med_sentence_length)

    n_sent = len(sent_lengths)

    pos_ngrams = tuple(pos_ngrams.items())
    pos_ngrams = sorted(pos_ngrams, key=lambda x: x[1],reverse=True)
    N = len(pos_ngrams)

    vals = np.array([x[1] for x in pos_ngrams])
    med = np.median(vals)

    file = r'D:\JanneK\Documents\git_repos\LaureaTextAnalyzer\data\external\common_POS_ngrams.txt'
    with open(file, 'w', encoding='utf-8') as f:
        f.write('total %i files, %i sentences, median sentence length %f, mean sentence length %f\n' % (len(files),len(sent_lengths),med_sentence_length,mean_sentence_length))
        for file in files:
            f.write('... file: %s\n' % file)
        f.write('POS_ngram\ttotal_count\trate\n')
        for row in range(0,np.minimum(20000,N)):
            if pos_ngrams[row][1]==2:
                break
            f.write('%s\t%i\t%f\n' % (pos_ngrams[row][0],pos_ngrams[row][1],pos_ngrams[row][1]/n_sent))

    print('all done!')

# convert token sequences back into strings. We need to remove some excess spaces in the process.
def tokens_to_string(tokens):
    text = " ".join(tokens)
    text = text.replace('#', '')
    assert text.find('_TAG')==-1,'Upper case TAG found!'
    text=text.replace('_tag','')
    text =text.replace('END_OF_PARAGRAPH','|')
    text = text.replace('CARD_NUMBER','¤')
    #text =text.replace('- " ', '- "')
    k = text.count('"')
    pos=0
    var=True
    for kk in range(k):
        ind = text[pos:].find('"')+pos
        pos1=[ind-1,ind+1]
        if text[pos1[0]:(pos1[1]+1)]==' " ':
            if var:
                text = text[0:(pos1[0]+2)] + text[(pos1[1]+1):]
            else:
                text = text[0:(pos1[0])] + text[(pos1[1]-1):]
            var= not var
        pos=ind+1
        assert k == text.count('"'), 'Number of parenthesis was changed'

    text =text.replace(' . ', '. ')
    if text[-2:]==' .':
        text = text[0:-2]+'.'
    text =text.replace(' , ', ', ')
    text =text.replace(' : ', ': ')
    text =text.replace(' ; ', '; ')
    text = text.replace(' ( ', ' (')
    text = text.replace(' ) ', ') ')
    text = text.replace(' [ ', ' [')
    text = text.replace(' ] ', '] ')
    text = text.replace(' / ', '/')
    text =text.replace(' ! ', '! ')
    text =text.replace('! !', '!!')
    text = text.replace(' ? ', '? ')
    text =text.replace('? ?', '??')
    return text

# compute stringkernels for a given data (hard-coded ngram lengths for now)
def get_StringKernel(data,keys):

    ngramMinLength=1

    kernels = {}#'LEMMA': 0, 'NORMAL': 0, 'RAW': 0}

    for key in keys:
        kernels[key]={}

    for datatype in kernels.keys():
        for ngramMinLength,ngramMaxLength in ((1,5),(1,10),(1,15),(2,10),(3,10),(5,10),(5,15)):

            ngramLength_str = str(ngramMinLength) + 'to' + str(ngramMaxLength)

            # data is list of dictionaries, each containing text

            print('\nComputing kernel for \'%s\' with ngrams %i-%i' % (datatype, ngramMinLength,ngramMaxLength))
            N = len(data)
            keys = list(data.keys())
            ker_mat = np.zeros((N,N),dtype=np.float32)
            all_ngrams={}

            print('...obtaining ngrams for %i texts' % len(keys))
            for ii,key in enumerate(keys):
                ngrams = {}
                x = tokens_to_string(data[key][datatype])
                for l in range(0, len(x)):
                    for d in range(ngramMinLength,ngramMaxLength + 1):
                        if l + d <= len(x):
                            ngram = x[l:(l + d)]
                            if ngram in ngrams:
                                ngrams[ngram] += 1
                            else:
                                ngrams[ngram] = 1
                all_ngrams[ii]=ngrams

            print('...computing matrix row-by-row')
            for ii in range(0,N):
                if ii%10==0:
                    print('.....row = %i of %i' % (ii+1,N))
                for jj in range(ii, N):
                    ker = 0
                    ngrams1 = all_ngrams[ii]
                    ngrams2 = all_ngrams[jj]

                    keys_a = set(ngrams1.keys())
                    keys_b = set(ngrams2.keys())
                    intersection = list(keys_a & keys_b)

                    for ngram in intersection:
                        repeats = min(ngrams1[ngram],ngrams2[ngram])
                        ker += repeats

                    ker_mat[ii,jj]= ker
                    ker_mat[jj,ii]=ker

            kernels[datatype][ngramLength_str]=(keys,ker_mat)

    print('all done!')

    return kernels

# simplify results by converting daraframes into dictionaries, omit some of the data
# needed to reduce size of pickles for cluster computing
def reduce_results(results):
    keep = ('F1', 'F1_null', 'R', 'R2', 'MSE_null', 'MSE')
    results_new = {}
    for typ in results.keys():
        if typ in ('train','test'):
            results_new[typ] = {}
            for item in keep:
                temp = {index: float(results[typ][item][index]) for index in results[typ][item].index}
                results_new[typ][item] = temp
        else:
            results_new[typ] = results[typ]
    return results_new

# get the real memory trace/size of an object in bytes
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size