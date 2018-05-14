import numpy as num
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import hamming_loss

def accuracy(ypred, ctest):
    ct=[1*(ctest.reshape(ctest.shape[0],)>i) for i in [0.5,0.8,1,2,5]] #hotspot
    ct1=[ctest.reshape(ctest.shape[0],)*(ctest.reshape(ctest.shape[0],)>i) for i in [0.5,0.8,1,2,5]] #the maximum obtainable number of calls-for-service for the amount of area forecasted (a). 
    yy=pd.DataFrame()
    yy['index']=range(len(ypred))
    yy['pred']=ypred
    yy['ctest']=ctest
    yy['area']=1
    yy=yy.sort_values(by=['pred'], ascending=False)
    yy['predcum']=yy['pred'].cumsum()
    yy['ctestcum']=yy['ctest'].cumsum()
    yy['areacum']=yy['area'].cumsum()
    predsum=yy['pred'].sum()
    areasum=yy['pred'].count()
    outsum=yy['ctest'].sum()
    yy['numpai']=yy['ctestcum']*areasum/outsum
    yy['denpai']=range(len(ypred)+1)[1:]
    yy['PAI']=yy['numpai']/yy['denpai']

    hot=[1*(ypred>i) for i in [0.5,0.8,1,2,5]] #hotspot
    fhot=[ctest.reshape(ctest.shape[0],)*(ypred>i) for i in [0.5,0.8,1,2,5]] #is the number of calls-for-service forecasted 

    ascores=', '.join([str(num.round(accuracy_score(p.reshape(ctest.shape[0],),q),4)) for p in ct for q in hot])
    f1scores=', '.join([str(num.round(f1_score(p.reshape(ctest.shape[0],),q, average='micro'),4)) for p in ct for q in hot])
    pscores=', '.join([str(num.round(precision_score(p.reshape(ctest.shape[0],),q),4)) for p in ct for q in hot])
    rescores=', '.join([str(num.round(recall_score(p.reshape(ctest.shape[0],),q),4)) for p in ct for q in hot])
    rocscores=', '.join([str(num.round(roc_auc_score(p.reshape(ctest.shape[0],),q),4)) for p in ct for q in hot])
    kappas=', '.join([str(num.round(cohen_kappa_score(p.reshape(ctest.shape[0],),q),4)) for p in ct for q in hot])
    matts=', '.join([str(num.round(matthews_corrcoef(p.reshape(ctest.shape[0],),q),4)) for p in ct for q in hot])
    ham=', '.join([str(num.round(hamming_loss(p.reshape(ctest.shape[0],),q),4)) for p in ct for q in hot])
    corrs=num.round(num.corrcoef(ctest.reshape(ctest.shape[0],),ypred)[0,1],4)
    r2=num.round(r2_score(ctest.reshape(ctest.shape[0],),ypred),4)
    evar=num.round(explained_variance_score(ctest.reshape(ctest.shape[0],),ypred),4)
    mae=num.round(mean_absolute_error(ctest.reshape(ctest.shape[0],),ypred),4)
    mse=num.round(mean_squared_error(ctest.reshape(ctest.shape[0],),ypred),4)
    NIJPAI=', '.join([str(num.round((num.sum(p)/num.sum(ctest))/(num.sum(q)/areasum),4)) for p in fhot for q in hot])
    NIJPAI1=[(num.sum(p)/num.sum(ctest))/(num.sum(q)/areasum) for p in fhot for q in hot]
    NIJPEI1=num.array(NIJPAI1)/num.array([((num.sum(p)/num.sum(ctest))/(num.sum(q)/areasum)) for p in ct1 for q in hot])
    NIJPEI=', '.join([str(num.round(i,4)) for i in NIJPEI1])
    PAI=num.round(num.max(yy['PAI']),4)
    return ascores, f1scores, pscores, rescores, rocscores, kappas, matts, ham, corrs, r2, evar, mae, mse, NIJPAI, NIJPEI, PAI


