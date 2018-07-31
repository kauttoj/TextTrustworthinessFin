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
         )
    LearnParams_RIDGE = (
        {'TextTypes': ['LEMMA'], 'Algorithm': ('Ridge', {'n_alphas': 25}), 'pass1_features': 200, 'FeatureSelection': ['fisher', 'global', 800], 'FeatureScaler': 'MaxAbsScaler', 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'pass1_features_SVD': None},
        )
    LearnParams_MLP = (
        {'pass1_features': 200, 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'Algorithm': ('MLP', {'hidden_layer_sizes': 25}), 'FeatureSelection': ['regression', 'global', 1500], 'TextTypes': ['LEMMA'], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features_SVD': None},
        )
    LearnParams['reliability']=LearnParams_SVR + LearnParams_MLP + LearnParams_RIDGE

    # SENTIMENT
    LearnParams_SVR = (
        {'FeatureScaler': 'MaxAbsScaler', 'Algorithm': ('SVM', {'kernel': 'linear'}), 'FeatureSelection': None, 'pass1_features_SVD': None, 'FeatureMethod': ['TFIDF_2', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'TextTypes': ['LEMMA'], 'pass1_features': 1300},
         )
    LearnParams_RIDGE = (
        {'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureSelection': None, 'pass1_features': 4000, 'pass1_features_SVD': None, 'Algorithm': ('Ridge', {'n_alphas': 25}), 'TextTypes': ['LEMMA'], 'FeatureScaler': 'MaxAbsScaler'},
        )
    LearnParams_MLP = (
        {'TextTypes': ['LEMMA'], 'FeatureScaler': 'MaxAbsScaler', 'FeatureSelection': None, 'Algorithm': ('MLP', {'hidden_layer_sizes': 5}), 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'pass1_features': 1500, 'pass1_features_SVD': None},
        )
    LearnParams['sentiment']=LearnParams_SVR + LearnParams_MLP + LearnParams_RIDGE

    # INFOVALUE
    LearnParams_SVR = (
        {'pass1_features_SVD': None, 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA', 'NORMAL', 'POS'], 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'StandardScaler', 'FeatureSelection': None, 'pass1_features': 600},
         )
    LearnParams_RIDGE = (
        {'TextTypes': ['LEMMA', 'NORMAL'], 'Algorithm': ('Ridge', {'n_alphas': 30}), 'FeatureSelection': None, 'FeatureScaler': 'StandardScaler', 'pass1_features': 500, 'pass1_features_SVD': None, 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        )
    LearnParams_MLP = (
        {'pass1_features': 600, 'FeatureMethod': ['TFIDF_2', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'Algorithm': ('MLP', {'hidden_layer_sizes': 4}), 'FeatureSelection': ['regression', 'global', 1500], 'TextTypes': ['LEMMA', 'NORMAL'], 'FeatureScaler': 'StandardScaler', 'pass1_features_SVD': None},
        )
    LearnParams['infovalue']=LearnParams_SVR + LearnParams_MLP + LearnParams_RIDGE

    # SUBJECTIVITY
    LearnParams_SVR = (
        {'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features_SVD': None, 'pass1_features': 400, 'FeatureSelection': ['regression', 'global', 1000], 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA', 'NORMAL']},
        )
    LearnParams_RIDGE = (
        {'TextTypes': ['LEMMA', 'NORMAL'], 'Algorithm': ('Ridge', {'n_alphas': 30}), 'FeatureSelection': ['regression', 'global', 1500], 'FeatureScaler': 'MaxAbsScaler', 'pass1_features': 500, 'pass1_features_SVD': None, 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
        )
    LearnParams_MLP = (
        {'FeatureSelection': None, 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'Algorithm': ('MLP', {'hidden_layer_sizes': 10}), 'TextTypes': ['LEMMA', 'NORMAL'], 'pass1_features': 300, 'FeatureScaler': 'StandardScaler', 'pass1_features_SVD': None},
        )
    LearnParams['subjectivity']=LearnParams_SVR + LearnParams_MLP + LearnParams_RIDGE

    # TEXTLOGIC
    LearnParams_SVR = (
        {'pass1_features_SVD': None, 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA', 'NORMAL'], 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'MaxAbsScaler', 'FeatureSelection': ['regression', 'global', 1000], 'pass1_features': 600},
        )
    LearnParams_RIDGE = (
        {'pass1_features_SVD': None, 'FeatureScaler': 'StandardScaler', 'FeatureSelection': ['regression', 'all', 400], 'TextTypes': ['LEMMA', 'POS'], 'pass1_features': 5000, 'Algorithm': ('Ridge', {'n_alphas': 15}), 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS']},
         )
    LearnParams_MLP = (
        {'FeatureSelection': ['regression', 'global', 1000], 'FeatureScaler': 'MaxAbsScaler', 'TextTypes': ['LEMMA', 'NORMAL'], 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'pass1_features': 600, 'pass1_features_SVD': None, 'Algorithm': ('MLP', {'hidden_layer_sizes': 4})},
        )
    LearnParams['textlogic']=LearnParams_SVR + LearnParams_MLP + LearnParams_RIDGE

    # WRITESTYLE/CLARITY
    LearnParams_SVR = (
        {'pass1_features_SVD': None, 'Algorithm': ('SVM', {'kernel': 'linear'}), 'TextTypes': ['LEMMA', 'POS'], 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'FeatureScaler': 'StandardScaler', 'FeatureSelection': ['regression', 'global', 1500], 'pass1_features': 200},
        )
    LearnParams_RIDGE = (
        {'TextTypes': ['LEMMA', 'POS'], 'Algorithm': ('Ridge', {'n_alphas': 25}), 'pass1_features': 200, 'FeatureSelection': ['fisher', 'global', 800], 'FeatureScaler': 'StandardScaler', 'FeatureMethod': ['TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'pass1_features_SVD': None},
        )
    LearnParams_MLP = (
        {'pass1_features': 200, 'Algorithm': ('MLP', {'hidden_layer_sizes': (20, 10)}), 'FeatureSelection': None, 'FeatureMethod': ['BOW_3', 'TFIDF_3', 'EMBEDDING_NORMAL', 'CUSTOM', 'TAGS'], 'pass1_features_SVD': None, 'FeatureScaler': 'StandardScaler', 'TextTypes': ['LEMMA', 'POS']},
        )
    LearnParams['writestyle']=LearnParams_SVR + LearnParams_MLP + LearnParams_RIDGE

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

    #pickle.dump((result),open(ResultPath + ('%s_results_set%i_subset%i_%s.pickle' % (dim.upper(),k1+1,k2+1,algo_str)), 'wb'))
    #Utils.plot_data(result[dim]['test'],ResultPath + ('%s_results_set%i_subset%i_%s' % (dim.upper(),k1+1,k2+1,algo_str)),title=dim + ' (%i folds)' % CV_folds)