# run code with best parameters (one set for each dimension)

import pickle
import os
from MainPipeline import MainPipeline
import Utils
import copy

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    IS_CLUSTER = 1

    LearnParams={}

    # RELIABILITY
    LearnParams_SVR = (
        {'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features_SVD': None, 'pass1_features': 300, 'FeatureSelection': ['regression', 'global', 1500], 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA', 'NORMAL']},
        {'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features_SVD': None, 'pass1_features': 200, 'FeatureSelection': None, 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA']},
        )
    LearnParams_RIDGE = (
        {'TextTypes': ['LEMMA'], 'Algorithm': ('Ridge', {'n_alphas': 25}), 'pass1_features': 200, 'FeatureSelection': ['fisher', 'global', 800], 'FeatureScaler': 'MaxAbsScaler', 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'pass1_features_SVD': None},
        {'TextTypes': ['LEMMA', 'NORMAL'], 'Algorithm': ('Ridge', {'n_alphas': 30}), 'FeatureSelection': ['regression', 'global', 1000], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features': 200, 'pass1_features_SVD': None, 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']}
        )
    LearnParams_EN = (
        {'TextTypes': ['LEMMA'], 'pass1_features': 200, 'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('ElasticNet', {'l1_ratio': 0.01}), 'FeatureSelection': None, 'pass1_features_SVD': None, 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        {'TextTypes': ['LEMMA'], 'pass1_features': 200, 'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('ElasticNet', {'l1_ratio': 0.01}), 'FeatureSelection': ['regression', 'global', 750], 'pass1_features_SVD': None, 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']}
        )
    LearnParams['reliability']=LearnParams_SVR + LearnParams_EN + LearnParams_RIDGE

    # SENTIMENT
    LearnParams_SVR = (
        {'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('SVM', {'kernel': 'linear'}), 'FeatureSelection': None, 'pass1_features_SVD': None, 'FeatureMethod': ['TFIDF_2', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'TextTypes': ['LEMMA'], 'pass1_features': 1300},
        {'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('SVM', {'kernel': 'linear'}), 'FeatureSelection': None, 'pass1_features_SVD': None, 'FeatureMethod': ['TFIDF_2', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'TextTypes': ['LEMMA'], 'pass1_features': 3000},
        )
    LearnParams_RIDGE = (
        {'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureSelection': None, 'pass1_features': 4000, 'pass1_features_SVD': None, 'Algorithm': ('Ridge', {'n_alphas': 25}), 'TextTypes': ['LEMMA'], 'FeatureScaler': 'MaxAbsScaler'},
        {'pass1_features_SVD': None, 'FeatureScaler': 'MaxAbsScaler', 'FeatureSelection': None, 'TextTypes': ['LEMMA'], 'pass1_features': 3000, 'Algorithm': ('Ridge', {'n_alphas': 20}), 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        )
    LearnParams_EN = (
        {'TextTypes': ['LEMMA'], 'pass1_features': 5000, 'FeatureSelection': None, 'FeatureMethod': ['TFIDF_2', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'Algorithm': ('ElasticNet', {'l1_ratio': 0.01}), 'FeatureScaler': 'MaxAbsScaler'},
        {'FeatureSelection': None, 'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('ElasticNet', {'l1_ratio': 0.05}), 'TextTypes': ['LEMMA'], 'pass1_features': 5000, 'FeatureMethod': ['TFIDF_2', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        )
    LearnParams['sentiment']=LearnParams_SVR + LearnParams_EN + LearnParams_RIDGE

    # INFOVALUE
    LearnParams_SVR = (
        {'pass1_features_SVD': None, 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA', 'NORMAL', 'POS'], 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'StandardScaler', 'FeatureSelection': None, 'pass1_features': 600},
        {'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'pass1_features': 600, 'Algorithm': ('SVM', {'kernel': 'linear'}), 'pass1_features_SVD': None, 'FeatureScaler': 'StandardScaler', 'FeatureSelection': ['regression', 'global', 1500], 'TextTypes': ['LEMMA', 'POS']},
        )
    LearnParams_RIDGE = (
        {'TextTypes': ['LEMMA', 'NORMAL'], 'Algorithm': ('Ridge', {'n_alphas': 30}), 'FeatureSelection': None, 'FeatureScaler': 'StandardScaler', 'pass1_features': 500, 'pass1_features_SVD': None, 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        {'TextTypes': ['LEMMA', 'NORMAL'], 'Algorithm': ('Ridge', {'n_alphas': 30}), 'FeatureSelection': ['regression', 'global', 1500], 'FeatureScaler': 'StandardScaler', 'pass1_features': 600, 'pass1_features_SVD': None, 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        )
    LearnParams_EN = (
        {'TextTypes': ['LEMMA', 'NORMAL'], 'pass1_features': 600, 'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('ElasticNet', {'l1_ratio': 0.01}), 'FeatureSelection': ['regression', 'global', 1500], 'pass1_features_SVD': None, 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        {'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'TextTypes': ['LEMMA'], 'pass1_features_SVD': None, 'pass1_features': 1000, 'Algorithm': ('ElasticNet', {'l1_ratio': 0.01}), 'FeatureSelection': None, 'FeatureScaler': 'StandardScaler'},
        )
    LearnParams['infovalue']=LearnParams_SVR + LearnParams_EN + LearnParams_RIDGE

    # SUBJECTIVITY
    LearnParams_SVR = (
        {'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features_SVD': None, 'pass1_features': 400, 'FeatureSelection': ['regression', 'global', 1000], 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA', 'NORMAL']},
        {'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('SVM', {'kernel': 'linear'}), 'FeatureSelection': ['regression', 'global', 600], 'pass1_features_SVD': None, 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'TextTypes': ['LEMMA', 'NORMAL'], 'pass1_features': 400},
        )
    LearnParams_RIDGE = (
        {'TextTypes': ['LEMMA', 'NORMAL'], 'Algorithm': ('Ridge', {'n_alphas': 30}), 'FeatureSelection': ['regression', 'global', 1500], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features': 500, 'pass1_features_SVD': None, 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        {'TextTypes': ['LEMMA', 'NORMAL'], 'Algorithm': ('Ridge', {'n_alphas': 30}), 'FeatureSelection': ['regression', 'global', 1000], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features': 400, 'pass1_features_SVD': None, 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        )
    LearnParams_EN = (
        {'TextTypes': ['LEMMA', 'NORMAL'], 'pass1_features': 400, 'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('ElasticNet', {'l1_ratio': 0.01}), 'FeatureSelection': ['regression', 'global', 1000], 'pass1_features_SVD': None,
         'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        {'TextTypes': ['LEMMA', 'NORMAL'], 'pass1_features': 500, 'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('ElasticNet', {'l1_ratio': 0.01}), 'FeatureSelection': ['regression', 'global', 1500], 'pass1_features_SVD': None,
         'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        )
    LearnParams['subjectivity']=LearnParams_SVR + LearnParams_EN + LearnParams_RIDGE

    # TEXTLOGIC
    LearnParams_SVR = (
        {'pass1_features_SVD': None, 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA', 'NORMAL'], 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'MaxAbsScaler', 'FeatureSelection': ['regression', 'global', 1000], 'pass1_features': 600},
        {'FeatureMethod': ['TFIDF_2', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features_SVD': None, 'pass1_features': 700, 'FeatureSelection': ['regression', 'global', 1000], 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA', 'NORMAL']},
        )
    LearnParams_RIDGE = (
        {'pass1_features_SVD': None, 'FeatureScaler': 'StandardScaler', 'FeatureSelection': ['regression', 'all', 400], 'TextTypes': ['LEMMA', 'POS'], 'pass1_features': 5000, 'Algorithm': ('Ridge', {'n_alphas': 15}), 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        {'TextTypes': ['LEMMA', 'NORMAL'], 'Algorithm': ('Ridge', {'n_alphas': 30}), 'FeatureSelection': ['regression', 'global', 1000], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features': 600, 'pass1_features_SVD': None, 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        )
    LearnParams_EN = (
        {'TextTypes': ['LEMMA', 'NORMAL'], 'pass1_features': 600, 'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('ElasticNet', {'l1_ratio': 0.01}), 'FeatureSelection': ['regression', 'global', 1000], 'pass1_features_SVD': None, 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        {'TextTypes': ['LEMMA', 'NORMAL'], 'pass1_features': 600, 'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('ElasticNet', {'l1_ratio': 0.01}), 'FeatureSelection': ['regression', 'global', 750], 'pass1_features_SVD': None, 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        )
    LearnParams['textlogic']=LearnParams_SVR + LearnParams_EN + LearnParams_RIDGE

    # WRITESTYLE/CLARITY
    LearnParams_SVR = (
        {'pass1_features_SVD': None, 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA', 'POS'], 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'StandardScaler', 'FeatureSelection': ['regression', 'global', 1500], 'pass1_features': 200},
        {'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features_SVD': None, 'pass1_features': 300, 'FeatureSelection': None, 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA', 'POS']},
        )
    LearnParams_RIDGE = (
        {'TextTypes': ['LEMMA', 'POS'], 'Algorithm': ('Ridge', {'n_alphas': 25}), 'pass1_features': 200, 'FeatureSelection': ['fisher', 'global', 800], 'FeatureScaler': 'StandardScaler', 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'pass1_features_SVD': None},
        {'TextTypes': ['LEMMA', 'POS'], 'Algorithm': ('Ridge', {'n_alphas': 30}), 'FeatureSelection': ['regression', 'global', 1500], 'FeatureScaler': 'StandardScaler', 'pass1_features': 200, 'pass1_features_SVD': None, 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        )
    LearnParams_EN = (
        {'FeatureSelection': None, 'FeatureScaler': 'StandardScaler', 'Algorithm': ('ElasticNet', {'l1_ratio': 0.01}), 'TextTypes': ['NORMAL', 'POS'], 'pass1_features': 5000, 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        {'FeatureSelection': ['regression', 'global', 600], 'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('ElasticNet', {'l1_ratio': 0.01}), 'TextTypes': ['LEMMA', 'POS'], 'pass1_features': 5000, 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        )
    LearnParams['writestyle']=LearnParams_SVR + LearnParams_EN + LearnParams_RIDGE

    ##-------------------------------------

    RootPath = r'D:/JanneK/Documents/git_repos/LaureaTextAnalyzer/'
    ResultPath = r'D:/JanneK/Documents/git_repos/LaureaTextAnalyzer/results/best_param_run/sequential_ensemble_results/'

    CV_folds=10

    my_object = MainPipeline(RootPath=RootPath,RECOMPUTE_ALL=0,RANDOM_TEST=0,SINGLE_THREAD_TEST=0,
                 CV_folds = CV_folds,
                 Type = 'regression',
                 verbose_level=1,
                 N_workers=5,
                 run_type='final',
                 is_sequential = True,
                 LearnParams=LearnParams
                 )
    result = my_object.run()

    #algo_str = par['LearnParams']['Algorithm'][0]

    pickle.dump((result),open(ResultPath + 'sequential_linear_ensemble_model_sLEM.pickle', 'wb'))
    #Utils.plot_data(result[dim]['test'],ResultPath + ('%s_results_set%i_subset%i_%s' % (dim.upper(),k1+1,k2+1,algo_str)),title=dim + ' (%i folds)' % CV_folds)