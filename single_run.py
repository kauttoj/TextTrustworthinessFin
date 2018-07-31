# run code with best parameters (one set for each dimension)

import pickle
import os
from MainPipeline import MainPipeline
import Utils
import copy

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    params={}

    params['reliability'] = {'LearnParams': {}}
    params['reliability']['LearnParams']['TextTypes'] = ('LEMMA','NORMAL')  #
    params['reliability']['LearnParams']['FeatureMethod'] = ('TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS')  # might drop BOW
    params['reliability']['LearnParams']['FeatureSelection'] = None
    params['reliability']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})
    params['reliability']['LearnParams']['pass1_features'] = 300
    params['reliability']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'

    params['sentiment'] = {'LearnParams': {}}
    params['sentiment']['LearnParams']['TextTypes'] = ('LEMMA',)  # might add NORMAL
    params['sentiment']['LearnParams']['FeatureMethod'] = ('TFIDF_2', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    params['sentiment']['LearnParams']['FeatureSelection'] = None#('regression', 'global', 1500)
    params['sentiment']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})  # 0.2-0.4
    params['sentiment']['LearnParams']['pass1_features'] = 1200
    params['sentiment']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'

    params['infovalue'] = {'LearnParams': {}}
    params['infovalue']['LearnParams']['TextTypes'] = ('LEMMA', 'NORMAL', 'POS',)
    params['infovalue']['LearnParams']['FeatureMethod'] = ('TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    params['infovalue']['LearnParams']['FeatureSelection'] = None
    params['infovalue']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})
    params['infovalue']['LearnParams']['pass1_features'] = 600
    params['infovalue']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'

    params['subjectivity'] = {'LearnParams': {}}
    params['subjectivity']['LearnParams']['TextTypes'] = ('NORMAL', 'LEMMA',)
    params['subjectivity']['LearnParams']['FeatureMethod'] = ('BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    params['subjectivity']['LearnParams']['FeatureSelection'] = ('regression', 'global', 1000,)
    params['subjectivity']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})
    params['subjectivity']['LearnParams']['pass1_features'] = 400
    params['subjectivity']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'

    params['textlogic'] = {'LearnParams': {}}
    params['textlogic']['LearnParams']['TextTypes'] = ('LEMMA', 'NORMAL',)
    params['textlogic']['LearnParams']['FeatureMethod'] = ('TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    params['textlogic']['LearnParams']['FeatureSelection'] = ['regression', 'global', 1000]
    params['textlogic']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})
    params['textlogic']['LearnParams']['pass1_features'] = 600
    params['textlogic']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'

    params['writestyle'] = {'LearnParams': {}}
    params['writestyle']['LearnParams']['TextTypes'] = ('LEMMA', 'POS')
    params['writestyle']['LearnParams']['FeatureMethod'] = ('BOW_3','TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    params['writestyle']['LearnParams']['FeatureSelection'] = None
    params['writestyle']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})
    params['writestyle']['LearnParams']['pass1_features'] = 300
    params['writestyle']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'

    # params['reliability'] = {'LearnParams': {}}
    # params['reliability']['LearnParams']['TextTypes'] = ('LEMMA','NORMAL')  #
    # params['reliability']['LearnParams']['FeatureMethod'] = ('BOW_3','TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS')  # might drop BOW
    # params['reliability']['LearnParams']['FeatureSelection'] = None
    # params['reliability']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})
    # params['reliability']['LearnParams']['pass1_features'] = 300
    # params['reliability']['LearnParams']['FeatureScaler'] = 'StandardScaler'
    #
    # params['sentiment'] = {'LearnParams': {}}
    # params['sentiment']['LearnParams']['TextTypes'] = ('LEMMA', 'NORMAL',)  # might add NORMAL
    # params['sentiment']['LearnParams']['FeatureMethod'] = ('TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    # params['sentiment']['LearnParams']['FeatureSelection'] = ('regression', 'all', 300)
    # params['sentiment']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})  # 0.2-0.4
    # params['sentiment']['LearnParams']['pass1_features'] = 4000
    # params['sentiment']['LearnParams']['FeatureScaler'] = 'StandardScaler' # not optimal (rank 19)
    #
    # params['infovalue'] = {'LearnParams': {}}
    # params['infovalue']['LearnParams']['TextTypes'] = ('LEMMA', 'NORMAL', 'POS',)
    # params['infovalue']['LearnParams']['FeatureMethod'] = ('TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    # params['infovalue']['LearnParams']['FeatureSelection'] = None
    # params['infovalue']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})
    # params['infovalue']['LearnParams']['pass1_features'] = 600
    # params['infovalue']['LearnParams']['FeatureScaler'] = 'StandardScaler'
    #
    # params['subjectivity'] = {'LearnParams': {}}
    # params['subjectivity']['LearnParams']['TextTypes'] = ('NORMAL', 'LEMMA',)
    # params['subjectivity']['LearnParams']['FeatureMethod'] = ('BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    # params['subjectivity']['LearnParams']['FeatureSelection'] = ('regression', 'global', 1000,)
    # params['subjectivity']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})
    # params['subjectivity']['LearnParams']['pass1_features'] = 300
    # params['subjectivity']['LearnParams']['FeatureScaler'] = 'StandardScaler'
    #
    # params['textlogic'] = {'LearnParams': {}}
    # params['textlogic']['LearnParams']['TextTypes'] = ('LEMMA', 'NORMAL', 'POS')
    # params['textlogic']['LearnParams']['FeatureMethod'] = ('TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    # params['textlogic']['LearnParams']['FeatureSelection'] = ['regression', 'global', 1000]
    # params['textlogic']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})
    # params['textlogic']['LearnParams']['pass1_features'] = 600
    # params['textlogic']['LearnParams']['FeatureScaler'] = 'StandardScaler'
    #
    # params['writestyle'] = {'LearnParams': {}}
    # params['writestyle']['LearnParams']['TextTypes'] = ('LEMMA', 'POS')
    # params['writestyle']['LearnParams']['FeatureMethod'] = ('BOW_3','TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    # params['writestyle']['LearnParams']['FeatureSelection'] = None
    # params['writestyle']['LearnParams']['Algorithm'] = ('SVM', {'kernel': 'linear'})
    # params['writestyle']['LearnParams']['pass1_features'] = 200
    # params['writestyle']['LearnParams']['FeatureScaler'] = 'StandardScaler'
    #
    # params['reliability'] = {'LearnParams': {}}
    # params['reliability']['LearnParams']['TextTypes'] = ('LEMMA',)  #
    # params['reliability']['LearnParams']['FeatureMethod'] = ('BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS')  # might drop BOW
    # params['reliability']['LearnParams']['FeatureSelection'] = None
    # params['reliability']['LearnParams']['Algorithm'] = ('ElasticNet', {'l1_ratio': 0.01})
    # params['reliability']['LearnParams']['pass1_features'] = 200
    # params['reliability']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'
    #
    # params['sentiment'] = {'LearnParams': {}}
    # params['sentiment']['LearnParams']['TextTypes'] = ('LEMMA',)  #
    # params['sentiment']['LearnParams']['FeatureMethod'] = ('TFIDF_2', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    # params['sentiment']['LearnParams']['FeatureSelection'] = None
    # params['sentiment']['LearnParams']['Algorithm'] = ('ElasticNet', {'l1_ratio': 0.01})
    # params['sentiment']['LearnParams']['pass1_features'] = 5000
    # params['sentiment']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'
    #
    # params['infovalue'] = {'LearnParams': {}}
    # params['infovalue']['LearnParams']['TextTypes'] = ('LEMMA', 'NORMAL',)
    # params['infovalue']['LearnParams']['FeatureMethod'] = ('TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    # params['infovalue']['LearnParams']['FeatureSelection'] = ('regression', 'global', 1500,)
    # params['infovalue']['LearnParams']['Algorithm'] = ('ElasticNet', {'l1_ratio': 0.01})
    # params['infovalue']['LearnParams']['pass1_features'] = 600
    # params['infovalue']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'
    #
    # params['subjectivity'] = {'LearnParams': {}}
    # params['subjectivity']['LearnParams']['TextTypes'] = ('NORMAL', 'LEMMA',)
    # params['subjectivity']['LearnParams']['FeatureMethod'] = ('BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    # params['subjectivity']['LearnParams']['FeatureSelection'] = ('regression', 'global', 1000,)
    # params['subjectivity']['LearnParams']['Algorithm'] = ('ElasticNet', {'l1_ratio': 0.01})
    # params['subjectivity']['LearnParams']['pass1_features'] = 400
    # params['subjectivity']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'
    #
    # params['textlogic'] = {'LearnParams': {}}
    # params['textlogic']['LearnParams']['TextTypes'] = ('LEMMA', 'NORMAL',)
    # params['textlogic']['LearnParams']['FeatureMethod'] = ('TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    # params['textlogic']['LearnParams']['FeatureSelection'] = ['regression', 'global', 1000]
    # params['textlogic']['LearnParams']['Algorithm'] = ('ElasticNet', {'l1_ratio': 0.01})
    # params['textlogic']['LearnParams']['pass1_features'] = 600
    # params['textlogic']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'
    #
    # params['writestyle'] = {'LearnParams': {}}
    # params['writestyle']['LearnParams']['TextTypes'] = ('LEMMA', 'POS')
    # params['writestyle']['LearnParams']['FeatureMethod'] = ('TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS',)
    # params['writestyle']['LearnParams']['FeatureSelection'] = ['regression', 'global', 600]
    # params['writestyle']['LearnParams']['Algorithm'] = ('ElasticNet', {'l1_ratio': 0.01})
    # params['writestyle']['LearnParams']['pass1_features'] = 5000
    # params['writestyle']['LearnParams']['FeatureScaler'] = 'MaxAbsScaler'

    RootPath = r'D:/JanneK/Documents/git_repos/LaureaTextAnalyzer/'

    CV_folds=10

    #for dim in params.keys():#set(params.keys()).intersection(['writestyle']):
    for dim in set(params.keys()):#.intersection(('reliability',)):

        par = copy.deepcopy(params[dim])

        print('Dimension = %s, %s' % (dim, Utils.print_parameters(par)))

        my_object = MainPipeline(RootPath=RootPath,RECOMPUTE_ALL=0,RANDOM_TEST=0,SINGLE_THREAD_TEST=0,
                     CV_folds = CV_folds,
                     Type = 'regression',
                     dimensions=dim,
                     verbose_level=1,
                     N_workers=5,
                     run_type='final',
                     **par,
                     )
        result = my_object.run()

        pickle.dump((result), open(RootPath + 'results' + os.sep + 'best_param_run' + os.sep + ('results_%s.pickle' % dim.upper()), 'wb'))
        Utils.plot_data(result[dim]['test'],RootPath + 'results' + os.sep + 'best_param_run'+ os.sep + ('results_%s' % dim.upper()),title=dim + ' (%i folds)' % CV_folds)