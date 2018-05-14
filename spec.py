import numpy as num
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC, MultiTaskLasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.linear_model import ElasticNetCV

def elas(cols, indep, crime, ctest):
    lasso = ElasticNet(random_state=0)
    alphas = num.logspace(-4, 1, 30)

    tuned_parameters = [{'alpha': alphas}]
    n_folds = 3

    clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
    clf.fit(indep, crime)
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']

    std_error = scores_std / num.sqrt(n_folds)

    indexs=list(scores).index(num.max(scores)) #find the alpha having the highest score
    optimal=alphas[indexs]
    lasso = ElasticNet(alpha=optimal)
    ll=lasso.fit(indep, crime)
    aa=ll.coef_
    try:
        aaa=[cols[p] for p in range(len(aa)) if aa[p] != 0]
    except:
        aa=aa.reshape(aa.shape[1],)
        aaa=[cols[p] for p in range(len(aa)) if aa[p] != 0]

    y_pred_lasso = ll.predict(indep)
    error_training=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-crime
    error_test=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-ctest

    r2_score_traing = r2_score(crime, y_pred_lasso)
    r2_score_test = r2_score(ctest, y_pred_lasso)

    #print(lasso)
    #print("r^2 on training data : %f" % r2_score_traing)
    #print ("optimal alpha "+str(optimal))
    #print (aaa)
    return aaa

def elascv(cols, indep, crime, ctest):
    model_lcv = ElasticNetCV()
    model_lcv.fit(indep, crime)
    alpha_lcv_ = model_lcv.alpha_
    lcv_coef=model_lcv.coef_
    elascols=[cols[p] for p in range(len(lcv_coef)) if lcv_coef[p] != 0]
    y_pred_lasso = model_lcv.predict(indep)
    error_training=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-crime
    error_test=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-ctest

    r2_score_traing = r2_score(crime, y_pred_lasso)
    r2_score_test = r2_score(ctest, y_pred_lasso)
    #print(model_lcv)
    #print ("alpha_lcv "+str(alpha_lcv_))
    #print("r^2 on training data : %f" % r2_score_traing)
    #print("r^2 on test data : %f" % r2_score_test)
    #print("error training:" +str(num.sum(error_training)))
    #print("error test:" +str(num.sum(error_test)))
    #print(cross_val_score(model_lcv, indep, crime)) 
    #print (lcvcols)
    return elascols

def melas(cols, indep, crime, ctest):
    lasso = MultiTaskElasticNet(random_state=0)
    alphas = num.logspace(-4, 1, 30)

    tuned_parameters = [{'alpha': alphas}]
    n_folds = 3

    clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
    clf.fit(indep, crime)
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']

    std_error = scores_std / num.sqrt(n_folds)

    indexs=list(scores).index(num.max(scores)) #find the alpha having the highest score
    optimal=alphas[indexs]
    lasso = MultiTaskElasticNet(alpha=optimal)
    ll=lasso.fit(indep, crime)
    aa=ll.coef_
    try:
        aaa=[cols[p] for p in range(len(aa)) if aa[p] != 0]
    except:
        aa=aa.reshape(aa.shape[1],)
        aaa=[cols[p] for p in range(len(aa)) if aa[p] != 0]

    y_pred_lasso = ll.predict(indep)
    error_training=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-crime
    error_test=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-ctest

    r2_score_traing = r2_score(crime, y_pred_lasso)
    r2_score_test = r2_score(ctest, y_pred_lasso)

    #print(lasso)
    #print("r^2 on training data : %f" % r2_score_traing)
    #print ("optimal alpha "+str(optimal))
    #print (aaa)
    return aaa

def lass(cols, indep, crime, ctest):
    lasso = Lasso(random_state=0)
    alphas = num.logspace(-4, 1, 30)

    tuned_parameters = [{'alpha': alphas}]
    n_folds = 3

    clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
    clf.fit(indep, crime)
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']

    std_error = scores_std / num.sqrt(n_folds)

    indexs=list(scores).index(num.max(scores)) #find the alpha having the highest score
    optimal=alphas[indexs]
    lasso = Lasso(alpha=optimal)
    ll=lasso.fit(indep, crime)
    aa=ll.coef_
    try:
        aaa=[cols[p] for p in range(len(aa)) if aa[p] != 0]
    except:
        aa=aa.reshape(aa.shape[1],)
        aaa=[cols[p] for p in range(len(aa)) if aa[p] != 0]

    y_pred_lasso = ll.predict(indep)
    error_training=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-crime
    error_test=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-ctest

    r2_score_traing = r2_score(crime, y_pred_lasso)
    r2_score_test = r2_score(ctest, y_pred_lasso)

    #print(lasso)
    #print("r^2 on training data : %f" % r2_score_traing)
    #print ("optimal alpha "+str(optimal))
    #print (aaa)
    return aaa

def mlass(cols, indep, crime, ctest):
    lasso = MultiTaskLasso(random_state=0)
    alphas = num.logspace(-4, 1, 30)

    tuned_parameters = [{'alpha': alphas}]
    n_folds = 3

    clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
    clf.fit(indep, crime)
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']

    std_error = scores_std / num.sqrt(n_folds)

    indexs=list(scores).index(num.max(scores)) #find the alpha having the highest score
    optimal=alphas[indexs]
    lasso = MultiTaskLasso(alpha=optimal)
    ll=lasso.fit(indep, crime)
    aa=ll.coef_
    try:
        aaa=[cols[p] for p in range(len(aa)) if aa[p] != 0]
    except:
        aa=aa.reshape(aa.shape[1],)
        aaa=[cols[p] for p in range(len(aa)) if aa[p] != 0]

    y_pred_lasso = ll.predict(indep)
    error_training=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-crime
    error_test=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-ctest

    r2_score_traing = r2_score(crime, y_pred_lasso)
    r2_score_test = r2_score(ctest, y_pred_lasso)

    #print(lasso)
    #print("r^2 on training data : %f" % r2_score_traing)
    #print ("optimal alpha "+str(optimal))
    #print (aaa)
    return aaa

def larcv(cols, indep, crime, ctest):
    model_lcv = LassoLarsCV()
    model_lcv.fit(indep, crime)
    alpha_lcv_ = model_lcv.alpha_
    lcv_coef=model_lcv.coef_
    lcvcols=[cols[p] for p in range(len(lcv_coef)) if lcv_coef[p] != 0]
    y_pred_lasso = model_lcv.predict(indep)
    error_training=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-crime
    error_test=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-ctest

    r2_score_traing = r2_score(crime, y_pred_lasso)
    r2_score_test = r2_score(ctest, y_pred_lasso)
    #print(model_lcv)
    #print ("alpha_lcv "+str(alpha_lcv_))
    #print("r^2 on training data : %f" % r2_score_traing)
    #print("r^2 on test data : %f" % r2_score_test)
    #print("error training:" +str(num.sum(error_training)))
    #print("error test:" +str(num.sum(error_test)))
    #print(cross_val_score(model_lcv, indep, crime)) 
    #print (lcvcols)
    return lcvcols

def orthcv(cols, indep, crime, ctest):
    model_orth = OrthogonalMatchingPursuitCV()
    model_orth.fit(indep, crime)
    #alpha_orth_ = model_orth.alpha_
    orth_coef=model_orth.coef_
    orthcvcols=[cols[p] for p in range(len(orth_coef)) if orth_coef[p] != 0]
    y_pred_lasso = model_orth.predict(indep)
    error_training=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-crime
    error_test=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-ctest

    r2_score_traing = r2_score(crime, y_pred_lasso)
    r2_score_test = r2_score(ctest, y_pred_lasso)
    #print(model_orth)
    #print ("alpha_orth "+str(alpha_orth_))
    #print("r^2 on training data : %f" % r2_score_traing)
    #print("r^2 on test data : %f" % r2_score_test)
    #print("error training:" +str(num.sum(error_training)))
    #print("error test:" +str(num.sum(error_test)))
    #print(cross_val_score(model_orth, indep, crime)) 
    #print (orthcols)
    return orthcvcols

def orth(cols, indep, crime, ctest):
    model_orth = OrthogonalMatchingPursuit()
    model_orth.fit(indep, crime)
    #alpha_orth_ = model_orth.alpha_
    orth_coef=model_orth.coef_
    orthcols=[cols[p] for p in range(len(orth_coef)) if orth_coef[p] != 0]
    y_pred_lasso = model_orth.predict(indep)
    error_training=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-crime
    error_test=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-ctest

    r2_score_traing = r2_score(crime, y_pred_lasso)
    r2_score_test = r2_score(ctest, y_pred_lasso)
    #print(model_orth)
    #print ("alpha_orth "+str(alpha_orth_))
    #print("r^2 on training data : %f" % r2_score_traing)
    #print("r^2 on test data : %f" % r2_score_test)
    #print("error training:" +str(num.sum(error_training)))
    #print("error test:" +str(num.sum(error_test)))
    #print(cross_val_score(model_orth, indep, crime)) 
    #print (orthcols)
    return orthcols

def bays(cols, indep, crime, ctest):
    model_orth = BayesianRidge()
    model_orth.fit(indep, crime)
    #alpha_orth_ = model_orth.alpha_
    orth_coef=model_orth.coef_
    bays=[cols[p] for p in range(len(orth_coef)) if num.round(orth_coef[p],1) != 0]
    y_pred_lasso = model_orth.predict(indep)
    error_training=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-crime
    error_test=y_pred_lasso.reshape(y_pred_lasso.shape[0],1)-ctest

    r2_score_traing = r2_score(crime, y_pred_lasso)
    r2_score_test = r2_score(ctest, y_pred_lasso)
    #print(model_orth)
    #print ("alpha_orth "+str(alpha_orth_))
    #print("r^2 on training data : %f" % r2_score_traing)
    #print("r^2 on test data : %f" % r2_score_test)
    #print("error training:" +str(num.sum(error_training)))
    #print("error test:" +str(num.sum(error_test)))
    #print(cross_val_score(model_orth, indep, crime)) 
    #print (orthcols)
    return bays
