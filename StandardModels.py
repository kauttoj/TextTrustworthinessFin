# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:23:33 2017

http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html
"""

#from mlxtend.classifier import StackingClassifier
from time import time
from sklearn.svm import SVR,SVC,NuSVR
from sklearn.kernel_ridge import KernelRidge
import xgboost
from sklearn.model_selection import KFold
import numpy as np
from sklearn.linear_model import BayesianRidge,ElasticNetCV,MultiTaskElasticNetCV,RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier,RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics.scorer import neg_mean_squared_error_scorer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import warnings
import time

def use_keras_CPU(num_cores=1):
    import tensorflow as tf
    from keras import backend as K
    num_CPU = 1
    num_GPU = 0
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)

def get_set_count(params):
    N=1
    for key in params:
        N=N*len(params[key])
    return N

class MLPRegressorCV(BaseEstimator):
    def __init__(self,n_alpha=25,hidden_layer_sizes=None):
        self.n_alpha = n_alpha
        self.hidden_layer_sizes = hidden_layer_sizes
    def fit(self, X, y):
        kf = KFold(n_splits=5,shuffle=True, random_state=1)
        kff = list(kf.split(X))
        errors = 9999*np.ones(self.n_alpha)
        alphas = np.logspace(np.log10(1),np.log10(400),self.n_alpha)
        for k,alpha in enumerate(alphas):
            model = MLPRegressor(activation="relu", solver ="lbfgs",learning_rate ="constant",
                         learning_rate_init = 0.001, max_iter = 400,random_state = None,
                         tol = 0.0001, epsilon = 1e-08,alpha=alpha,hidden_layer_sizes=self.hidden_layer_sizes)        
            err = 0
            for train_indices, test_indices in kff:
                model.fit(X[train_indices], y[train_indices])
                y_pred = model.predict(X[test_indices])
                err+=np.mean((y_pred-y[test_indices])**2)
            errors[k]=err
            
        ind_best = np.argmax(-errors)
        alpha = alphas[ind_best]
        print('--> MLPRegressorCV best alpha: %f' % alpha)
        self.model = MLPRegressor(activation="relu", solver ="lbfgs",learning_rate ="constant",
                     learning_rate_init = 0.001, max_iter = 600,random_state = None,
                     tol = 0.0001, epsilon = 1e-08,alpha=alpha,hidden_layer_sizes=self.hidden_layer_sizes)          
        self.model.fit(X,y)        
        return self
    def predict(self, X):
        return self.model.predict(X)

class KerasENet(BaseEstimator):
    def __init__(self, l1_ratio=0.5,alpha=0.01,epochs=1000):
        self.l1_ratio = l1_ratio
        self.alpha=alpha
        self.model = None
        self.epochs = epochs
        assert 0<=self.l1_ratio<=1,'l1 mixing ratio!'
        assert self.alpha>=0,'Bad regularization constant!'
        assert self.epochs>5,'Bad epoch value!'
    def fit(self, X, y):
        from keras.regularizers import l1_l2
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.initializers import Constant
        from keras.callbacks import EarlyStopping
        from keras import backend as K
        K.clear_session()
        stop_here = EarlyStopping(patience=5,monitor='loss',min_delta=0.001)
        reg = l1_l2(l1=self.alpha*self.l1_ratio,l2=(1.0 - self.l1_ratio)*self.alpha)
        init = Constant(value=np.mean(y))
        self.model = Sequential()
        self.model.add(Dense(1,input_shape=(X.shape[1],),activation='linear',kernel_regularizer=reg,kernel_initializer='glorot_normal',bias_initializer=init))
        self.model.compile(loss='mse',optimizer='sgd')
        #self.model.fit(X,y,epochs=50,batch_size=X.shape[0],verbose=1) # do at least 50 iterations
        self.model.fit(X, y, epochs=20, batch_size=X.shape[0], verbose=0)
        self.model.fit(X, y, epochs=self.epochs, batch_size=X.shape[0], verbose=0, callbacks=[stop_here])
        return self
    def predict(self, X):
        return self.model.predict(X)
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alpha": self.alpha, "l1_ratio": self.l1_ratio}
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self
    #def score(self, X, y=None):
    #    # counts number of values bigger than mean
    #    return mean_squared_error(self.predict(X),y)

def main(X,Y,Params,print_info=False,is_regression=True,Y_other=None):

    parameters = Params['Algorithm'][1]
    is_cv_run = False
    starttime = time.time()

    if print_info:
        print('Fitting model \'%s\' for %s' % (Params['Algorithm'][0],'regression' if is_regression else 'classification'))

    if Params['Algorithm'][0] =='BayesianRidge':
        if not is_regression:
            model = BayesianRidge(n_iter=300, tol=0.001,compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False,**parameters)
            #parameters = {'alpha_1': [1e-6,1e-5,1e-4],'alpha_2': [1e-6,1e-5,1e-4], 'lambda_1': [1e-6,1e-5,1e-4], 'lambda_2': [1e-6,1e-5,1e-4]}
        else:
            model = BayesianRidge(n_iter=300, tol=0.001, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False, **parameters)
    elif Params['Algorithm'][0] == 'StringKernel':
        if not is_regression:
            raise (Exception('not implemented'))
        else:
            # we create an instance of SVM and fit out data.
            #
            # model = KernelRidge(alpha=parameters['alpha'], kernel='precomputed')
            model = SVR(kernel='precomputed', gamma='auto', coef0=0.0, shrinking=True, tol=0.001, cache_size=400, verbose=False, max_iter=-1)
            param_grid = {'C': np.logspace(np.log10(0.0001),np.log10(500),25)}

            model = NuSVR(kernel='precomputed')#cache_size=400, coef0=0.0, gamma='auto', max_iter=-1, shrinking=True, tol=0.001, verbose=False,**parameters)
            param_grid = {'nu':(0.50,)}

            model = GridSearchCV(model, param_grid,n_jobs=1, iid=True, refit=True, cv=7, verbose=0,scoring=neg_mean_squared_error_scorer)
            is_cv_run = True

    elif Params['Algorithm'][0] == 'XGBoost':
        # max_depth = 3, learning_rate = 0.1, n_estimators = 100, silent = True, objective = 'reg:linear',
        # booster = 'gbtree', n_jobs = 1, nthread = None, gamma = 0, min_child_weight = 1,
        # max_delta_step = 0, subsample = 1, colsample_bytree = 1, colsample_bylevel = 1, reg_alpha = 0,
        # reg_lambda = 1, scale_pos_weight = 1, base_score = 0.5, random_state = 0, seed = None,
        # missing = None
        if not is_regression:
            model = xgboost.XGBClassifier(missing=None, silent=True,
                                       learning_rate=0.10,
                                       objective='rank:pairwise',
                                       booster='gbtree',
                                       n_jobs=1,
                                       max_delta_step=0,
                                       colsample_bylevel=1,
                                       scale_pos_weight=1,
                                       base_score=0.5,
                                       random_state=666,
                                       colsample_bytree=0.75, # default 1
                                       subsample=0.75,
                                       gamma=0,
                                       reg_alpha=0.01, # default 0
                                       min_child_weight=6,
                                       **parameters)
        else:
            # model=xgboost.XGBRegressor(missing=None, silent=True,
            #                            learning_rate=0.10,
            #                            objective='reg:linear',#'rank:pairwise' booster='gbtree'
            #                            n_jobs=1,
            #                            booster='gbtree',
            #                            max_delta_step=0,
            #                            colsample_bylevel=1,
            #                            scale_pos_weight=1,
            #                            base_score=0.5,
            #                            random_state=666,
            #                            colsample_bytree=0.75, # default 1
            #                            subsample=0.75,
            #                            gamma=0,
            #                            reg_alpha=0.01, # default 0
            #                            reg_lambda=1.0,
            #                            min_child_weight=6,
            #                            **parameters)

            model=xgboost.XGBRegressor(missing=None, silent=True,
                                       learning_rate=0.10,
                                       objective='reg:linear',#'rank:pairwise' booster='gbtree'
                                       n_jobs=1,
                                       booster='gbtree',
                                       random_state=666,
                                       **parameters)

            param_grid = {'colsample_bytree': (0.75,1.0),'subsample':(0.75,1.0),'min_child_weight':(3,6,9),'reg_lambda':(0.80,1.0,1.20),'reg_alpha':(0.001,0.01)}
            model = GridSearchCV(model, param_grid,n_jobs=1, iid=True, refit=True, cv=7, verbose=0,scoring=neg_mean_squared_error_scorer)
            is_cv_run = True

    elif Params['Algorithm'][0]== "Keras_ElasticNet":

        #use_keras_CPU()

        if not is_regression:
            raise (Exception('ElasticNet is only for regression!'))
        else:
            param_grid = {'l1_ratio':(Params['Algorithm'][1]['l1_ratio'],),'alpha':np.logspace(-3,1,15)}

            model = GridSearchCV(KerasENet(), param_grid,n_jobs=1, iid=True, refit=True, cv=5, verbose=0,scoring=neg_mean_squared_error_scorer)
            # first_output = Dense(1,activation='sigmoid')(first_output)
            is_cv_run = True

    elif Params['Algorithm'][0] == "Ridge":
        if not is_regression:
            raise (Exception('Ridge is only for regression!'))
        else:
            model = RidgeCV(alphas=np.logspace(-1, np.log10(700),parameters['n_alphas']),fit_intercept=True, normalize=False, scoring=None, cv=8, gcv_mode=None, store_cv_values=False)

    elif Params['Algorithm'][0]== "ElasticNet":
        tol = 0.0001
        selection='cyclic'
        n_alphas=90
        max_iter=1300
        if X.shape[1]>4000:
            tol = 0.001
            selection='random'
            n_alphas=60
            max_iter=1000
        if not is_regression:
            raise (Exception('ElasticNet is only for regression!'))
        else:
            if Params['is_multitarget']:
                model = MultiTaskElasticNetCV(eps=0.001, alphas=None, fit_intercept=True, normalize=False, max_iter=max_iter, tol=tol, cv=7, copy_X=True, verbose=0, n_alphas=n_alphas, n_jobs=1, random_state=666, selection=selection, **parameters)
            else:
                model = ElasticNetCV(eps=0.001,alphas=None,fit_intercept=True,normalize=False,max_iter=max_iter, tol=tol,cv=7, copy_X=True, verbose=0,n_alphas=n_alphas,n_jobs=1,random_state=666,selection=selection,**parameters)

    elif Params['Algorithm'][0]== "RandomForest":

        if not is_regression:
            raise(Exception('not set up (lazy)'))
        else:
            model = RandomForestRegressor(criterion='mse',min_samples_leaf=1,min_weight_fraction_leaf = 0.0, max_leaf_nodes = None, min_impurity_decrease = 0.0, min_impurity_split = None, bootstrap = True, oob_score = False, n_jobs = 1, random_state = None, verbose = 0, warm_start = False,**parameters)
            param_grid = {'max_features': ('auto', 'sqrt'),'min_samples_split':(2,4,),}
            model = GridSearchCV(model, param_grid, n_jobs=1, iid=True, refit=True, cv=7, verbose=0, scoring=neg_mean_squared_error_scorer)
            is_cv_run = True

    elif Params['Algorithm'][0] == 'SVM':
        # 0.001, 0.005, 0.01, 0.05, 0.1, 0.5,1.0,1.5,2.0,3.0,4.0,5.0,10.0
        if not is_regression:
            model = SVC(cache_size=400, coef0=0.0, gamma='auto', max_iter=-1, shrinking=True, tol=0.001, verbose=False,**parameters)
            #parameters = {'reg__C':[0.5],'reg__epsilon':[0.1]}
        else:
            model = SVR(cache_size=400, coef0=0.0, gamma='auto', max_iter=-1, shrinking=True, tol=0.001, verbose=False,**parameters)
            param_grid = {'C': np.logspace(np.log10(0.0005),np.log10(10),30)}
            #param_grid = {'nu':(0.1,0.3,0.5,0.7,0.9)}
            model = GridSearchCV(model, param_grid, n_jobs=1, iid=True, refit=True, cv=8, verbose=0, scoring=neg_mean_squared_error_scorer)
            is_cv_run = True
			
    elif Params['Algorithm'][0] == 'GradientBoosting':
        if not is_regression:
            model = GradientBoostingClassifier(random_state=1,**parameters)
            #parameters = {'reg__n_estimators': [140], 'reg__max_depth': [6],'learning_rate':[0.01,0.03,0.1],'min_samples_leaf':[2,3,4]}
        else:
            model = GradientBoostingRegressor(random_state=1,**parameters)
            #parameters = {'reg__n_estimators': [140], 'reg__max_depth': [6]}
    elif Params['Algorithm'][0] == 'MLP':
        #parameters['hidden_layer_sizes']=[parameters['hidden_layer_sizes']]
        #model = MLPRegressorCV(hidden_layer_sizes=parameters['hidden_layer_sizes'])
        model = MLPRegressor(activation="relu", solver ="lbfgs",learning_rate ="constant",
                         learning_rate_init = 0.0011, max_iter = 450,random_state = None,
                         tol = 0.00013, epsilon = 1e-08,hidden_layer_sizes=parameters['hidden_layer_sizes'])

        param_grid = {'alpha': np.logspace(0,np.log10(350),20)}
        model = GridSearchCV(model, param_grid, n_jobs=1, iid=True, refit=True, cv=7, verbose=0, scoring=neg_mean_squared_error_scorer)
        is_cv_run = True
        #model = MLPRegressor(activation="relu", solver ="lbfgs",learning_rate ="constant",
        #             learning_rate_init = 0.001, power_t = 0.5, max_iter = 500, shuffle = True, random_state = None,
        #             tol = 0.0001, verbose = False, warm_start = False, momentum = 0.9, epsilon = 1e-08,**parameters)
    elif Params['Algorithm'][0] == 'MLP_KERAS':

        from keras.models import Sequential
        from keras import regularizers
        from keras.layers import Dense, Dropout
        from keras.callbacks import EarlyStopping
        from sklearn.preprocessing import LabelEncoder
        from keras.utils import np_utils
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        model = Sequential()
        model.add(Dense(parameters['layers_and_nodes'][0], activation='tanh', input_shape=(X.shape[1],), kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(parameters['l2_regularization']), ))
        model.add(Dropout(parameters['dropout'], noise_shape=None, seed=1))
        for layer in range(1, len(parameters['layers_and_nodes'])):
            model.add(Dense(parameters['layers_and_nodes'][layer], activation='relu', input_shape=(parameters['layers_and_nodes'][layer - 1],),
                            kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(parameters['l2_regularization'])))
            model.add(Dropout(parameters['dropout'], noise_shape=None, seed=1))

        if not is_regression:
            model.add(Dense(1, activation='softmax', input_shape=(parameters['nodes'][-1],)))
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['f1'])
            encoder = LabelEncoder()
            encoder.fit(Y)
            encoded_Y = encoder.transform(Y)
            # convert integers to dummy variables (i.e. one hot encoded)
            Y = np_utils.to_categorical(encoded_Y)
        else:
            model.add(Dense(1, activation='linear',input_shape=(parameters['layers_and_nodes'][-1],)))
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

        model.fit(X,Y, batch_size=X.shape[0],epochs=100,validation_split=0,verbose=0)#,callbacks=[early_stopping])

        return model

    else:

        raise(Exception('unknown model'))
    #decomposer = LatentDirichletAllocation(n_topics=10, max_iter=10,learning_method='online',learning_offset=50.,random_state=1)
    #decomposer = TruncatedSVD(n_components=100,random_state=666)
       
    """
    X = data.iloc[:]['text'].values
    y = data.iloc[:]['mylabel'].values.astype(str)
    
    dat = vect.fit_transform(X)
    dat = tfidf.fit_transform(dat)
    dat = decomposer.fit_transform(dat)  
    
    for a in numpy.unique(y):
        plt.scatter(dat[y==a,0],dat[y==a,1])
    """
     
    """
    START LOOP
    """



    #t0 = time()
    # if get_set_count(parameters)>1:
    #     grid_search = GridSearchCV(model, parameters, n_jobs=6,verbose=1,cv=10,refit=True)
    #     grid_search.fit(X=X,y=Y)
    #     best_parameters = grid_search.best_estimator_.get_params()
    #     print('--> best parameters: %s' % best_parameters)
    #     return grid_search
    # else:

    if 1:
        start_time = time.time()
        print('... training model (X.shape=%s)' % str(X.shape),end='')

    warnings.filterwarnings("ignore")

    if Y_other is not None and Params['is_multitarget']:
        Y = np.expand_dims(Y, axis=1)
        model.fit(X=X, y=np.concatenate((Y,Y_other),axis=1))
    else:
        Y = Y.flatten()
        model.fit(X=X,y=Y)

    if is_cv_run:
        print(' [best gridsearch params: %s] ' % model.best_params_,end='')

    if 1:
        end_time = time.time()
        print(' ... done (%1.1f min)' % ((end_time - start_time)/60.0))

    #elapsedtime = (time.time() - starttime) / 60.0
    #print('fit done (took %f minutes)' % elapsedtime)

    return model
