#!/usr/bin/env python
# Modelling tools for fire prediction project

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import time

#For mapping
import folium
from branca.utilities import split_six
from shapely.geometry import Point

plt.style.use('seaborn-talk')


def setupdataset(dataset,fire_to_predict='SF_pred',yearto_holdout=2018):

    '''
    Divide the dataframe into three component - the train and test, the
    holdout and the use_for_future parts
    '''

    #This is a datasat for which we will predict the 2019 fires
    use_for_future = dataset[dataset['IDyear_x']==yearto_holdout]
    #This is the dataset that we'll use to predict the 2018 fires
    holdout = dataset[dataset['IDyear_x']==yearto_holdout-1]
    #This is the dataset that we'll train and test on
    traintest = dataset[dataset['IDyear_x']<yearto_holdout-1]

    firecols = ['SF_pred','VF_pred','EF_pred']
    todrop = ['GISYEARJOIN']

    fires = traintest[firecols] #This is what we're trying to predict

    #This is the dataset of predictors we're using to do the prediction
    X = traintest.drop(firecols,axis=1)
    X.drop(todrop,inplace=True,axis=1)
    y = fires[fire_to_predict].apply(classify)

    #holdout dataset
    firesholdout = holdout[firecols] #This is what we're trying to predict
    yholdout = firesholdout[fire_to_predict].apply(classify)
    Xholdout = holdout.drop(firecols,axis=1) #This is what we're using to predict
    Xholdout.drop(todrop,inplace=True,axis=1)

    #usefor future
    Xuseforfuture = use_for_future.drop(firecols,axis=1)
    Xuseforfuture.drop(todrop,inplace=True,axis=1)

    return X,y,Xholdout,yholdout,Xuseforfuture,holdout,use_for_future


def classify(val):

    '''Function to one-hot encode a column in a dataframe, for modelling purposes'''
    
    if val >= 1:
        return int(1)
    else:
        return int(0)

def generate_pipeline_logReg(scale=True,select=True,test_parameters=False):

    #Scaling
    SC = StandardScaler()

    #Classifier 
    LRC = LogisticRegression()
    #Feature selector
    FS = SelectFromModel(estimator=LRC,threshold='mean')
    #Pipeline
    if (scale == True) and (select == True):
        LRC_pipeline = Pipeline([('scale',SC),('select',FS),('classify', LRC)])
    elif (scale == True) and (select == False):
        LRC_pipeline = Pipeline([('scale',SC),('classify', LRC)])
    elif (scale == False) and (select == True ):
        LRC_pipeline = Pipeline([('select',SC),('classify', LRC)])
    else:
        LRC_pipeline = Pipeline([('classify', LRC)])

    if test_parameters == False:

        #Parameters to search over
        test_parameters = {
    
        'classify__C': (0.1,1,10),
        'classify__penalty': ('L1','L2'),
        'classify__class_weight': ('balanced',None),
          }

    #Number of folds
    nfolds=5

    #Grid search object to set up
    grid_search = GridSearchCV(LRC_pipeline, test_parameters, \
                           scoring='roc_auc',verbose=1, cv=nfolds, n_jobs=4)

    return LRC_pipeline, grid_search, test_parameters

def generate_pipeline_RC(scale=False,select=True,test_parameters=False):

    '''
    Set up modelling pipline for random forest classifier
    User selects parameters to conduct grid search and number of folds of CV
    '''

    #Scaling
    SC = StandardScaler()

    #Classifier 
    RC = RandomForestClassifier()
    #Feature selector
    FS = SelectFromModel(estimator=RC,threshold='mean')
    #Pipeline
    if (scale == True) and (select == True):
        RC_pipeline = Pipeline([('scale',SC),('select',FS),('classify', RC)])
    elif (scale == True) and (select == False):
        RC_pipeline = Pipeline([('scale',SC),('classify', RC)])
    elif (scale == False) and (select == True ):
        RC_pipeline = Pipeline([('select',FS),('classify', RC)])
    else:
        RC_pipeline = Pipeline([('classify', RC)])

    if test_parameters == False:

        #Parameters to search over
        test_parameters = {
        'classify__n_estimators': (10,20,30),
        'classify__max_depth': (5,8,10),
        'classify__min_samples_leaf':(50,60,80),
        }

    #Number of folds
    nfolds=5

    #Grid search object to set up
    grid_search = GridSearchCV(RC_pipeline, test_parameters, \
                           scoring='roc_auc',verbose=1, cv=nfolds, n_jobs=4)

    return RC_pipeline, grid_search, test_parameters


def generate_pipeline_GB(scale=False,select=True,test_parameters=False):

    '''
    Set up modelling pipline for gradient boosting classifier
    User selects parameters to conduct grid search and number of folds of CV
    '''

    #Scaling
    SC = StandardScaler()

    #Classifier 
    GB = GradientBoostingClassifier()
    #Feature selector
    FS = SelectFromModel(estimator=GB,threshold='mean')
    #Pipeline
    if (scale == True) and (select == True):
        GB_pipeline = Pipeline([('scale',SC),('select',FS),('classify', GB)])
    elif (scale == True) and (select == False):
        GB_pipeline = Pipeline([('scale',SC),('classify', GB)])
    elif (scale == False) and (select == True ):
        GB_pipeline = Pipeline([('scale',SC),('classify', GB)])
    else:
        GB_pipeline = Pipeline([('classify', GB)])

    if test_parameters == False:

        #Parameters to search over
        test_parameters = {
        'classify__n_estimators': (10,20,30),
        'classify__n_learning_rate': (0.5,1,2),
        'classify__max_depth': (5,8,10),
        'classify__min_samples_leaf': (5,10,50),
        }

    #Number of folds
    nfolds=5

    #Grid search object to set up
    grid_search = GridSearchCV(GB_pipeline, test_parameters, \
                           scoring='roc_auc',verbose=1, cv=nfolds, n_jobs=4)

    return GB_pipeline, grid_search, test_parameters


def perform_grid_search(pipeline_obj,grid_search_obj,X_train,y_train,test_parameters,Xcols,select=True):
    
    '''
    Perform a grid search over hyperparmeters to generate the best model. Takes a generic pipeline object and grid search
    object. Prints the length of time it takes to run a model. Runs GridSearchCV over some input range of parameters to
    select the best model based in AOC score
    '''
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline_obj.steps])
    print("parameters:")
    print(test_parameters)
    t0 = time.time()
    
    #Do the grid search on the training dataset
    grid_search_obj.fit(X_train, y_train)
    
    print("done in %0.3fs" % (time.time() - t0))
    print()

    print("Best score: %0.3f" % grid_search_obj.best_score_)
    print("Best parameters set:")
    best_estimator = grid_search_obj.best_estimator_
    best_parameters = best_estimator.get_params()
    for param_name in sorted(test_parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
    #Get the names of the columns corresponding to the features we want to keep
    if select == True:
        support = best_estimator.named_steps['select'].get_support()
        X_new_cols = [Xcols[i] for i in range(len(support)) if support[i] == True]
    else:
        X_new_cols = Xcols
 
    #Return the best estimator object for use with the holdout dataset
    
    return X_new_cols, best_estimator


def plot_ROC_curve(Xdata,Ytrue,model,title):

    '''
    Plot ROC curve for this model 
    '''

    fires_holdout_predict = model.predict_proba(Xdata)

    fpr, tpr, thresholds1 = roc_curve(np.array(Ytrue).astype(int), fires_holdout_predict[:,1])

    score = roc_auc_score(np.array(Ytrue).astype(int), fires_holdout_predict[:,1])

    score_label = 'AOC score: %.3f' %score 

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.plot(fpr,tpr,label=score_label)
    ax.set_xlabel("False Positive")
    ax.set_ylabel("True Positive")
    ax.plot(np.linspace(0,1,10),np.linspace(0,1,10),'k--')
    ax.set_title('%s' %title)

        
    ax.fill_between(fpr, np.zeros(len(fpr)),
                     tpr, alpha=0.1,
                     color="b")
    ax.grid()
    plt.legend(loc='best')
    fname = 'AOC_%s.png' %title
    plt.savefig(fname,dpi=400)

    return fig, fpr, tpr, thresholds1

def plot_Learning_curve(estimator, title, X, y, ylim=None, cv=None,
                  n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),score_type='roc_auc'):

    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    fig = plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111)
    ax.set_title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("ROC AUC Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring=score_type)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid()

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    ax.legend(loc="best")

    fname = 'LC_%s.png' %title
    plt.savefig(fname,dpi=400)


    return fig


def generate_hazard_map_html(model,X,mapdata,html_map_name):

    '''
    Generate new hazard map as a html file
    INPUTS
    model: an estimator object
    X: the data for which we want to predict fires
    mapdata: block data in the form of SF_blocks_years
    html_map_name: name of the html file to be produced
    '''


    GISCELLS = list(X['GISYEARJOIN'])
    mapgeom = mapdata[mapdata['GISYEARJOI'].isin(GISCELLS)]
    fires_holdout_predict = model.predict_proba(X)
    riskmap = gpd.GeoDataFrame({'geometry':mapdata['geometry'],'fire_prob':fires_holdout_predict[:,1]})

    riskmap['BLOCKID'] = np.arange(len(riskmap_2018))
    gdf_wgs84 = riskmap.copy()
    gdf_wgs84.crs = {'init': 'epsg:4326', 'no_defs': True}

    thresh_scale = split_six(riskmap_2018['fire_prob'])

    m = folium.Map(location=[37.76, -122.42],zoom_start=13,
               tiles="CartoDB positron",
              width='100%',
              height='100%')

    m.choropleth(geo_data=gdf_wgs84.to_json(),data=riskmap_2018,columns=['BLOCKID','fire_prob'],
             key_on = 'feature.properties.{}'.format('BLOCKID'),
             max_zoom=16,
             fill_opacity=0.8,
             fill_color='OrRd',
             line_opacity=0.1,
             highlight=True,
             legend_name='Probability of fire',
             legend_scale=thresh_scale)

    m.add_child(folium.LatLngPopup())

    m.save(html_map_name)







