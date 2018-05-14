import numpy as num
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score

def explan(plsca, numcom,insp_1,depsp_1):
    xscore = plsca.x_scores_
    yscore = plsca.y_scores_
    xload = plsca.x_loadings_
    yload = plsca.y_loadings_
    xweigt = plsca.x_weights_
    yweight = plsca.y_weights_
    xrot = plsca.x_rotations_
    yrot = plsca.y_rotations_
    coe = plsca.coef_
    r2=plsca.score(scale(insp_1), scale(depsp_1))

    print ('the coefficient of determination R^2 of the prediction: '+str(r2))

    #Y explanation
    r2_sum = 0
    yexplain=[]
    for i in range(0,numcom):
        Y_pred=num.dot(plsca.x_scores_[:,i].reshape(-1,1),plsca.y_loadings_[:,i].reshape(-1,1).T)*depsp_1.std(axis=0, ddof=1)+depsp_1.mean(axis=0)
        aa= r2_score(depsp_1,Y_pred)
        r2_sum += round(aa,numcom)    
        print('R2 for %d component: %g' %(i+1,round(aa,numcom)))
        yexplain.append(aa)

    try:
        yyy=[q for p, q in zip(num.round(yexplain,3), range(numcom)) if p >= 0.01][-1]+1
        yexplain1=yexplain[:yyy]
        reY=num.dot(yload.T[:yyy].T,yscore.T[:yyy])
    except:
        print ('low y explanation')
        reY=num.dot(yload,yscore.T)

    try:
        #X explanation
        # variance in transformed X data for each latent vector:
        variance_in_x = num.var(plsca.x_scores_, axis = 0, ddof=1)
        # normalize variance by total variance:
        xexplain = variance_in_x / num.sum(variance_in_x)

        xxx=[q for p, q in zip(num.round(xexplain,3), range(numcom)) if p >= 0.01][-1]+1
        xexplain1=xexplain[:xxx]
        xgg=xscore.T[:xxx].T
        reX=num.dot(xload.T[:xxx].T,xscore.T[:xxx])
    except:
        print ('low x explanation')
        xgg=xscore*1
        reX=num.dot(xload,xscore.T)

    #global model
    X = sm.add_constant(xgg)
    Y = yscore.T[0].T
    #Y = yscore.T[4].T
    model = sm.OLS(Y, X)
    results = model.fit()
    print ('fit for the first component')
    print(results.summary())
    return reX, reY

