# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 23:14:04 2021

@author: jollybenito
"""

## REQUISITES/LIBRARIES
import numpy as np
from scipy.optimize import minimize
from scipy.stats import kurtosis
from scipy.stats import skew
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.linalg import block_diag
import matlab.engine
from sklearn.covariance import LedoitWolf
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro
from scipy.stats import ks_2samp  
from scipy.stats import jarque_bera
import seaborn as sns

#----------------------------------------------------
# MARCENKO PASTUR PDF
def mpPDF(var, q, pts):
    """
    Creates a Marchenko-Pastur Probability Density Function
    Args:
        var (float): Variance
        q (float): T/N where T is the number of rows and N the number of columns
        pts (int): Number of points used to construct the PDF
    Returns:
        pd.Series: Marchenko-Pastur PDF
    """
    # Marchenko-Pastur pdf
    # Adjusting code to work with 1 dimension arrays
    if isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = var[0]
    c = 1.*1/q
    dmin = (1.0 - np.sqrt(c))**2
    dmax = (1.0 + np.sqrt(c))**2
    x=np.linspace(dmin,dmax,pts)
    marcenko = np.zeros([pts])
    for i in range(pts):
        if (x[i] < dmin or x[i] > dmax):
            rho = 0
        else:   
            rho = np.sqrt((dmax-x[i])*(x[i]-dmin))/(2*np.pi*c*x[i])
        marcenko[i] = rho
    pdf=pd.Series(marcenko,index=x, name="density")
    return(pdf)  
#----------------------------------------------------------
# Get the Eigenvalues and Eigenvector values
def getPCA(matrix):
    """
    Gets the Eigenvalues and Eigenvector values from a Hermitian Matrix
    Args:
        matrix pd.DataFrame: Correlation matrix
    Returns:
         (tuple): tuple containing:
            np.ndarray: Eigenvalues of correlation matrix
            np.ndarray: Eigenvectors of correlation matrix
    """
    # Get eVal,eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec
#----------------------------------------------------------
# Fit kernel
def fitKDE(obs, bWidth=.25, kernel='gaussian', x=None):
    """
    Fit kernel to a series of obs, and derive the prob of obs x is the array of values
        on which the fit KDE will be evaluated. It is the empirical PDF
    Args:
        obs (np.ndarray): observations to fit. Commonly is the diagonal of Eigenvalues
        bWidth (float): The bandwidth of the kernel. Default is .25
        kernel (str): The kernel to use. Valid kernels are [‘gaussian’|’tophat’|
            ’epanechnikov’|’exponential’|’linear’|’cosine’] Default is ‘gaussian’.
        x (np.ndarray): x is the array of values on which the fit KDE will be evaluated
    Returns:
        pd.Series: Empirical PDF
    """
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten(), name="density")
    return pdf
#----------------------------------------------------------
# Generates a Random Covariance Matrix
def getRndCov(nCols, nFacts):
    """
    Generates a Random Covariance Matrix
    Args:
        nCols (int): number of columns of random normal. This will be the dimensions of
            the output
        nFacts (int): number of rows of random normal
    Returns:
        cov (np.ndarray): random covariance matrix
    """
    w = np.random.normal(size=(nCols, nFacts))
    cov = np.dot(w, w.T)  # random cov matrix, however not full rank
    cov += np.diag(np.random.uniform(size=nCols))  # full rank cov
    return cov
#----------------------------------------------------------
def cov2corr(cov):
    """
    Derive the correlation matrix from a covariance matrix
    Args:
        cov (np.ndarray): covariance matrix
    Returns:
        corr (np.ndarray): correlation matrix
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error
    return corr
#----------------------------------------------------------
def errPDFs(var, eVal, q, bWidth, pts=1000):
    """
    Fit error of Empirical PDF (uses Marchenko-Pastur PDF)
    Args:
        var (float): Variance
        eVal (np.ndarray): Eigenvalues to fit.
        q (float): T/N where T is the number of rows and N the number of columns
        bWidth (float): The bandwidth of the kernel.
        pts (int): Number of points used to construct the PDF
    Returns:
        sse (float): sum squared error
    """
    # Fit error
    pdf0 = mpPDF(var, q, pts)  # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)
    return sse
#----------------------------------------------------------
def findMaxEval(eVal, q, bWidth):
    """
    Find max random eVal by fitting Marchenko’s dist (i.e) everything else larger than
        this, is a signal eigenvalue
    Args:
        eVal (np.ndarray): Eigenvalues to fit on errPDFs
        q (float): T/N where T is the number of rows and N the number of columns
        bWidth (float): The bandwidth of the kernel.
    Returns:
         (tuple): tuple containing:
            float: Maximum random eigenvalue
            float: Variance attributed to noise (1-result) is one way to measure
                signal-to-noise
    """

    out = minimize(lambda *x: errPDFs(*x), .5, args=(eVal, q, bWidth),
                   bounds=((1E-5, 1 - 1E-5),))
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    eMax = var * (1 + (1. / q) ** .5) ** 2
    return eMax, var
#--------------------------------------------------
# Applies the clipping method.
def clean_corrmatrix(corr,n):
    """
    Find the correlation matrix estimation, via the clipping method.
    Args:
        corr (np.array): Correlation matrix (Sample estimator).
        n (float): The number of rows/observations of the original database.
    Returns:
         (tuple): tuple containing:
            d (np.array): Clipped eigenvalues.
            Denoised_corr (np.array): The new denoised correlation matrix.
    """
    w, v = np.linalg.eigh(corr)
    p = len(w)
    q = 1.0*p/n
    dmax = (1.0 + np.sqrt(q))**2
    # Find the mean eigenvalue noise.
    noise = []
    for i in range(p):
        if (w[i] < dmax):
            noise.append(w[i])
    eignoise = np.mean(noise)
    # Replace noise by mean eigenvalue noise.
    d = np.zeros([p])
    for i in range(p):
        if (w[i] < dmax):
            d[i] = eignoise
        else:
            d[i] = w[i]
    escale = 1.*np.sum(d)
    D = np.eye(p) * d*p/escale
    # Calculate the clean correlation matrix
    Denoised_corr = np.dot(np.dot(v,D),np.linalg.inv(v)) 
    return(d,Denoised_corr)
#---------------------------------------------------
# TW Checker 
def twtest(twtable, x1):
    """
    Checks the TWtable csv file. To find probability corresponding the value.
    Args:
        twtable (np.array): Table of PDF values for the TW distribution.
        x1 (float): Value to check against the table.
    Returns:
        TW probability corresponding the x1 value.
            
    Parameters
    ----------
    twtable : TYPE
        DESCRIPTION.
    x1 : TYPE
        DESCRIPTION.
    """
    l = np.shape(twtable)[0]
    i = 0
    while(i < l and x1 >= twtable[i,0]):
        i += 1
    if(i == l):
        return(0)
    if(i == 0):
        return(1)
    else:
        return(twtable[i-1,1]+(twtable[i,1] - twtable[i-1,1])*(x1 - twtable[i-1,0]) / (twtable[i,0]-twtable[i-1,0]))
#---------------------------------------------------
# Applies the Tracy-Widom method.
def TW_Wishart_order2(corr,n,twtable,alpha):
    """
    Find the correlation matrix estimation, via the TW method.
    Args:
        corr (np.array): Correlation matrix (Sample estimator).
        n (float): The number of rows/observations of the original database.
    Returns:
         (tuple): tuple containing:
            d (np.array): Denoised eigenvalues.
            Denoised_corr (np.array): The new denoised correlation matrix.
    """
    lambdas, vectors1 = getPCA(corr)
    lambdas=np.diag(lambdas)
    p = len(lambdas)
    # n = np.shape(X)[1]
    k = 0
    sum_noise = 0.0
    d = np.zeros(p)    
    mu = (np.sqrt(n-0.5) + np.sqrt(p-0.5))**2
    sigma = np.sqrt(mu)*(1.0/np.sqrt(n-0.5)+1.0/np.sqrt(p-0.5))**(1./3)
    for j in range(len(lambdas)):
        x_lambda1 = (n*lambdas[j]-mu)/sigma
        
        probab = twtest(twtable, x_lambda1)
        ### (i) The test rejects the null hypothesis if the probab is geq than alpha
        if (probab < alpha):
            d[j] = lambdas[j]
        else:
            sum_noise = sum_noise + lambdas[j]        
            k = k + 1
    if (k>0):
        d[p-k-1:] = sum_noise/k            
    D = np.eye(p) * d
    # Denoised correlation matrix
    Denoised_corr = np.dot(np.dot(vectors1,D),np.linalg.inv(vectors1)) 
    return(d, Denoised_corr)
#-----------------------------------------------------
# Applies the Ledoit-Wolf linear shrinkage.
def Ledoit_Wolf(cov, corr,n):
    """
    Find the correlation matrix estimation, via the LW linear shrinkage.
    Args:
        cov (np.array): Covariance matrix (Sample estimator).
        corr (np.array): Correlation matrix (Sample estimator).
        n (float): The number of rows/observations of the original database.
    Returns:
         (tuple): tuple containing:
            d (np.array): Denoised eigenvalues.
            Denoised_corr (np.array): The new denoised correlation matrix.
    """
    shrinkage_est=LedoitWolf().fit(cov).shrinkage_
    lambdas,vectors=np.linalg.eigh(corr)
    p = len(lambdas)
    d=1+(lambdas-1)*shrinkage_est
    D = np.eye(p) * d 
    Denoised_corr = np.dot(np.dot(vectors,D),np.linalg.inv(vectors))
    return(d[::-1], Denoised_corr)
#---------------------------------------------------------
# Applies the arbitrary shrinkage, it's linear shrinkage with a constant.
def ArbitraryLin(corr,n):
    """
    Find the correlation matrix estimation, via an arbitrary linear shrinkage.
    In this case the linear shrinkage estimator is equal to 0.5.
    Args:
        corr (np.array): Correlation matrix (Sample estimator).
        n (float): The number of rows/observations of the original database.
    Returns:
         (tuple): tuple containing:
            d (np.array): Denoised eigenvalues.
            Denoised_corr (np.array): The new denoised correlation matrix.
    """
    shrinkage_est=0.5
    lambdas,vectors=np.linalg.eigh(corr)
    p = len(lambdas)
    d=1+(lambdas-1)*shrinkage_est
    D = np.eye(p) * d
    Denoised_corr = np.dot(np.dot(vectors,D),np.linalg.inv(vectors))
    return(d[::-1], Denoised_corr)
#-------------------------------------------
def QUEST(corr, n, name):
    """
    Find the correlation matrix estimation, via Ledoit Wolf non linear shrinkage.
    We need to use an external matlab code. See sources in thesis or go to:
    http://www.econ.uzh.ch/en/people/faculty/wolf/publications.html and find the 
    QUEST.zip file.
    Args:
        corr (np.array): Correlation matrix (Sample estimator).
        n (float): The number of rows/observations of the original database.
        name (string): Name of the file where the original database is.
    Returns:
         (tuple): tuple containing:
            d (np.array): Denoised eigenvalues.
            Denoised_corr (np.array): The new denoised correlation matrix.
    """
    lambdas,vectors=np.linalg.eigh(corr)  
    eng = matlab.engine.start_matlab()
    eng.non_lin_shrink(name,nargout=0)
    eng.pause(1)
    namef1=name+"_NonlinearShrink"+".csv"
    CovarianceEst=np.array(pd.read_csv(namef1, header=None))
    CorrEst=cov2corr(CovarianceEst)
    # We do not use these vectors, we use the vectors of the original correlation
    d,vectors_not=np.linalg.eigh(CorrEst)   
    p = len(d)
    # escale = 1.*np.sum(d)
    D = np.eye(p) * d # *n/escale   
    Denoised_corr = np.dot(np.dot(vectors,D),np.linalg.inv(vectors)) #clean correlation matrix
    return(d[::-1], Denoised_corr)
#---------------------------------------------------
def corr2cov(corr,std):
    """
    Turn correlation into covariance.
    Args:
        corr (np.array): Correlation matrix (Sample estimator).
        std (np.array): The standard deviations from the original database.
    Returns:
        cov: The covariance matrix (Sample estimator).
    """
    cov=corr*np.outer(std,std)
    return cov
#---------------------------------------------------
def formBlockMatrix(nBlocks,bSize,bCorr):
    """
    Create a Block matrix.
    Args:
        nBlocks (integer): Number of blocks in the matrix.
        bSize (integer): Block size.
        bCorr (float): (Intra)Correlation to use inside the blocks.
    Returns:
        corr (np.array): The block matrix.
    """
    block=np.ones((bSize,bSize))*bCorr
    block[range(bSize),range(bSize)]=1
    corr=block_diag(*([block]*nBlocks))
    return corr
#---------------------------------------------------
def formTrueMatrix(nBlocks,bSize,bCorr):
    """
    Create a True matrix.
    Args:
        nBlocks (integer): Number of blocks in the matrix.
        bSize (integer): Block size.
        bCorr (float): (Intra)Correlation to use inside the blocks.
    Returns:
         (tuple): tuple containing:
            mu0 (np.array): The mean of the True matrix.
            cov0 (np.array) : The covariance of the True matrix.
    """
    corr0=formBlockMatrix(nBlocks,bSize,bCorr)
    corr0=pd.DataFrame(corr0)
    cols=corr0.columns.tolist()
    np.random.shuffle(cols)
    corr0=corr0[cols].loc[cols].copy(deep=True)
    std0=np.random.uniform(.05,.2,corr0.shape[0])
    cov0=corr2cov(corr0,std0)
    mu0=np.random.normal(std0,std0,cov0.shape[0]).reshape(-1,1)
    return mu0,cov0
#---------------------------------------------------
def getCovSub(nObs,nCols,sigma,random_state=None):
    # Sub correl matrix
    rng=check_random_state(random_state)
    if nCols==1:return np.ones((1,1))
    ar0=rng.normal(size=(nObs,1))
    ar0=np.repeat(ar0,nCols,axis=1)
    ar0+=rng.normal(scale=sigma,size=ar0.shape)
    ar0=np.cov(ar0,rowvar=False)
    return ar0
#---------------------------------------------------
def getRndBlockCov(nCols,nBlocks,minBlockSize=1,sigma=1., random_state=None):
    # Generate a block random correlation matrix
    rng=check_random_state(random_state)
    parts=rng.choice(range(1,nCols-(minBlockSize-1)*nBlocks), \
    nBlocks-1,replace=False)
    parts.sort()
    parts=np.append(parts,nCols-(minBlockSize-1)*nBlocks)
    parts=np.append(parts[0],np.diff(parts))-1+minBlockSize
    cov=None
    for nCols_ in parts:
        cov_=getCovSub(int(max(nCols_*(nCols_+1)/2.,100)), \
        nCols_,sigma,random_state=rng)
        if cov is None:cov=cov_.copy()
        else:cov=block_diag(cov,cov_)
    return cov
#---------------------------------------------------
def randomBlockCorr(nCols,nBlocks,random_state=None, minBlockSize=1):
    # Form block corr
    rng=check_random_state(random_state)
    cov0=getRndBlockCov(nCols,nBlocks,
    minBlockSize=minBlockSize,sigma=.5,random_state=rng)
    cov1=getRndBlockCov(nCols,1,minBlockSize=minBlockSize,
    sigma=1.,random_state=rng) # add noise
    cov0+=cov1
    corr0=cov2corr(cov0)
    corr0=pd.DataFrame(corr0)
    return corr0
#---------------------------------------------------
def clusterKMeansBase(corr0,maxNumClusters=10,n_init=10):
    x,silh=((1-corr0.fillna(0))/2.)**.5, pd.Series(dtype='float64')# observations matrix
    for init in range(n_init):
        for i in range(2,maxNumClusters+1):
            kmeans_=KMeans(n_clusters=i,n_init=1)
            kmeans_=kmeans_.fit(x)
            silh_=silhouette_samples(x,kmeans_.labels_)
            stat=(silh_.mean()/silh_.std(),silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0]>stat[1]:
                silh,kmeans=silh_,kmeans_
    newIdx=np.argsort(kmeans.labels_)
    corr1=corr0.iloc[newIdx] # reorder rows
    corr1=corr1.iloc[:,newIdx] # reorder columns
    clstrs={i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() \
            for i in np.unique(kmeans.labels_) } # cluster members
    silh=pd.Series(silh,index=x.index)
    return corr1,clstrs,silh
#-------------------------------------------
def corr2cov_(C_clean_, cov_1):
    p = C_clean_.shape[0]
    #Clean covariance matrix
    vec = np.zeros([p])
    for i in range(p):
        vec[i] = np.sqrt(cov_1[i,i])
    V =   np.eye(p)*vec 
    Cov = np.dot(np.dot(V,C_clean_),V)
    return(Cov)
#--------------------------------------------
def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]   
#--------------------------------------------
# Risk function (True & In)
def Risk2_same(g, w, C):
    Risk2 = g**2/np.dot(np.dot(np.transpose(w),np.linalg.inv(C)), w)
    return(Risk2)
#--------------------------------------------
# Risk function (Out)
def Risk2_out(g, w, E, C):
    Risk2 = (g**2)*np.dot(np.dot(np.dot(np.dot(np.transpose(w),np.linalg.inv(E)),C),np.linalg.inv(E)),w)/(np.dot(np.dot(np.transpose(w),np.linalg.inv(E)), w))**2
    return(Risk2)
#----------------------------------------------------
def norm_graph(x,name):    
    data=x.flatten()
    mu, std = norm.fit(data)
    
    # Plot the histogram.
    plt.hist(data, bins=50, density=True, alpha=0.6, color='g')
    # bins=50, 
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    name2 = name + '_Normal_Graph' +'.png'
    plt.xlabel('Valor')
    plt.ylabel('Densidad de probabilidad')
    plt.savefig(name2,dpi=300) 
    plt.show()
    plt.clf()
    
    Shapiro_test=shapiro(x)
    JB_test=jarque_bera(x)
    Normality_tests=np.zeros([3,2])
    Normality_tests[0,0]=Shapiro_test.statistic
    Normality_tests[0,1]=Shapiro_test.pvalue
    Normality_tests[1,0]=JB_test.statistic
    Normality_tests[1,1]=JB_test.pvalue
    Normality_tests=pd.DataFrame(Normality_tests)
    Normality_tests.to_csv(name + "_Normality_formal" +".csv")
    return (0)
#---------------------------------------------------
def basic_stats(return_vec,n,name):
    # Basic statistics
    ## Quantiles
    mean_ = np.mean(return_vec, axis=1) 
    mean_ = mean_ - 0
    variance_ = np.var(return_vec, axis=1) 
    std_ = np.std(return_vec, axis=1)
    kurtosis_ = kurtosis(return_vec, axis=1)
    skew_ = skew(return_vec, axis=1)
    
    JB_pandas = np.zeros(len(mean_))
    adf_pandas = np.zeros([4,len(mean_)])
    for m in range(len(mean_)):
         JB_pandas[m]=n/6*(skew_[m]**2+((kurtosis_[m]-3)**2)/4)         
         adf_pandas[:,m]= adfuller(return_vec[:,m], autolag='AIC')[0:4]
    Basic_stats=np.stack((mean_, variance_, kurtosis_,skew_,JB_pandas,
                         adf_pandas[0,:],adf_pandas[1,:]))
    Basic_stats=pd.DataFrame(np.transpose(Basic_stats))
    #change column names
    Basic_stats.columns = ['Media', '´Varianza', 'Kurtosis', 'Asimetría', 'Prueba JB', 
                           'Estadistico ADF','p-valor ADF']
    # Save basic statistics
    Basic_stats.to_csv(name+"_Basic_stats"+".csv")  
    alpha = 0.05 #95% confidence level
    VaR_n = norm.ppf(1 - alpha) * std_**2 - (mean_)
    CVaR_n = alpha ** -1 * norm.pdf(norm.ppf(alpha)) * std_**2 - (mean_)
    
    Metric_stats = np.stack((VaR_n,CVaR_n))
    Metric_stats=pd.DataFrame(np.transpose(Metric_stats))
    Metric_stats.columns = ['Valor en riesgo', 'Valor en riesgo condicional']
    
    return (0)
#----------------------------------------------------
def Heatmap(corr0,col_names,name):
    corr0 = pd.DataFrame(corr0, columns = col_names)
    ax = sns.heatmap(corr0, vmin=-1, vmax=1, center=0,
                cmap=sns.diverging_palette(20, 220, n=200),
                square=True,xticklabels="auto", 
                yticklabels="auto")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig(name + "_Heatmap.png",dpi=300)
    plt.show()
    plt.clf()
    return(0)
#----------------------------------------------------
def eigen_graph(cov0,name, p, n, alpha,twtable):
    corr0 = cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0) 
    # Based on z we pick a bias cleaning method
    eVal3, CleanIn = TW_Wishart_order2(corr0,n, twtable,alpha)
    eVal4, CleanIn = Ledoit_Wolf(cov0, corr0, n)
    eVal5, CleanIn = QUEST(corr0, n, name + "M")
    eVal6, CleanIn = clean_corrmatrix(corr0, n) 
    ### IN-SAMPLE
    plt.style.use(['default'])
    # Titles in english for other uses
    # names = ['Noise', 'Clipping', 'TW', 'Linear S.', 'Non-linear S.']
    # Titles in spanish for purposes of thesis
    names = ['Con sesgo', 'Recorte', 'TW', 'C. lineal', 'C. no lineal']
    colors = ['k', 'r', 'b', 'orange', 'g']
    line_styles = ["solid", "solid", "dashed","solid", "dotted"]
    plt.plot(np.diag(eVal0),np.diag(eVal0), linestyle=line_styles[0], label=names[0], color=colors[0])
    plt.plot(np.diag(eVal0),eVal6[::-1], linestyle=line_styles[1], label=names[1], color=colors[1])
    plt.plot(np.diag(eVal0),eVal3, linestyle=line_styles[2], label=names[2], color=colors[2])
    plt.plot(np.diag(eVal0),eVal4, linestyle=line_styles[3], label=names[3], color=colors[3])
    plt.plot(np.diag(eVal0),eVal5, linestyle=line_styles[4], label=names[4], color=colors[4])
    plt.legend()
    s_name=  '_Bias_reduction.png'
    #plt.title(s_name)
    limit1=min(5,max(np.diag(eVal0)))
    plt.xlim(0, limit1)
    plt.ylim(0, limit1)
    plt.xlabel('λ')
    plt.ylabel('ξ')
    s2_name = name + s_name
    plt.savefig(s2_name,dpi=300)
    plt.show()
    plt.clf()
    return(0)
#-----------------------------------------------------
def MarPas(p, n, lambdas,name):
    series0=mpPDF(1.,q=n/p,pts=1000) #obs/var
    pdf0=series0.to_frame()
    pdf0.reset_index(level=0, inplace=True)
    list1=[]
    for i in range(1000):
     list1.append("Teórica")
    # Using 'indicator' as the column name
    # and equating it to the list
    pdf0['Indicador'] = list1    
    series1=fitKDE(np.diag(lambdas),bWidth=.01) # empirical pdf
    pdf1=series1.to_frame()
    pdf1.reset_index(level=0, inplace=True)
    list2=[]
    for i in range(p):
     list2.append("Empírica")  
    # Using 'indicator' as the column name
    # and equating it to the list
    pdf1['Indicador'] = list2
    sns.lineplot(data=pdf0[pdf0["index"]<4],  x="index", y="density",
                 color="blue", palette="pastel", 
                 label="Teórica")
    sns.kdeplot(data=pdf1[pdf1["index"]<4], x="index",bw_adjust=.2,
                 color="orange", palette="pastel")
    sns.histplot(data=pdf1[pdf1["index"]<4], x="index", element="step",
                 color="moccasin", stat="density", 
                 #label="Sintética")
                 label="Empírica")
    sns.rugplot(data=pdf1[pdf1["index"]<4], color="orange", x="index")
    plt.legend(loc='upper left',bbox_to_anchor = (0.7, .85)) 
    name2 = name + 'MarcenkoPastur_Seaborn' + '.png'
    plt.xlabel(r'$\lambda$') 
    plt.ylabel('Densidad de probabilidad para q=' + str(n/(2*p)))
    plt.savefig(name2,dpi=300) 
    plt.tight_layout()
    plt.show()
    plt.clf()

    """
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    ## Fig 1
    fig = plt.figure(figsize=(5, 5),facecolor='white')
    ax = fig.add_subplot(111)
    # the main axes is subplot(111) by default
    
    sns.lineplot(data=pdf0[pdf0["index"]<4],  x="index", y="density",
                 color="blue", palette="pastel", 
                 label="Teórica")
    sns.kdeplot(data=pdf1[pdf1["index"]<4], x="index",bw_adjust=.2,
                 color="orange", palette="pastel")
    sns.histplot(data=pdf1[pdf1["index"]<4], x="index", element="step",
                 color="moccasin", stat="density", 
                 label="Empírica")
    sns.rugplot(data=pdf1[pdf1["index"]<4], color="orange", x="index")
    plt.legend(loc='upper left',bbox_to_anchor = (0.7, .35)) 
    name2 = name + '_MarcenkoP_' + '.png'
    plt.xlabel(r'$\lambda$') 
    plt.ylabel('Densidad de probabilidad')
    plt.tight_layout()
    # this is an inset axes over the main axes
    inset_axes = inset_axes(ax, 
                    width="30%", # width = 30% of parent_bbox
                    height=1.0, # height : 1 inch
                    loc=1)       
    sns.kdeplot(data=pdf1, x="index",bw_adjust=.2,
                 color="orange", palette="pastel")
    plt.xlim([4, max(pdf1["index"])])
    plt.xlabel(r"")
    plt.ylabel(r"") 
    plt.show()
    plt.clf()
    """

    #perform Kolmogorov-Smirnov test
    KSResult=ks_2samp(series0, series1)
    KS=np.array([KSResult.statistic,KSResult.pvalue])
    KS=pd.DataFrame(KS)
    KS.to_csv(name+"_Kolmogorov.csv")
    return pdf1
#---------------------------------------------------
# Portfolio Optimization without constraints
def optPort(cov, G, mu=None):
    """
    Portfolio optimization without constraints according to Markowtiz model.
    Args:
        cov (np.array): Covariance matrix (Sample estimator).
        G (float): Expected earnings of the portfolio.
        mu (Optional)(np.array): Mean vector of the database.
    Returns:
        w (np.array): Optimized weights of the portfolio.
    """
    inv=np.linalg.inv(cov)
    ones=np.ones(shape=(inv.shape[0],1))
    if mu is None:mu=ones
    if G is None:
        w=np.zeros(shape=(inv.shape[0],1))
    else:
        w=G*np.dot(inv,mu)/np.dot(mu.T,np.dot(inv,mu))
    return w  
#------------------------------------------------------
def calculations_IN(cov, mu):
    points = 100
    G = np.linspace(0.01,100,points)
    VAR_in = np.zeros([points])   
    for i in range(points):
        VAR_in[i] = Risk2_same(G[i],mu,cov)
        initial_min = VAR_in[0]
        MaxSharpe = 0
        MaxSharpe_id = 0
        # MARKOWITZ WEIGHTS
        # Calculate optimal weights for the modified covariance portfolios            
        weights = optPort(cov, G[i]) 
        # Sharpe Ratio
        # volatility=np.sqrt(np.dot(np.dot(weights.T,cov),weights)) 
        Sharpe_temp = np.dot(weights.flatten(), mu)/VAR_in[i]
        if (VAR_in[i]<=initial_min):
            initial_min = VAR_in[i]
            MinVar_id = i
        if (Sharpe_temp>MaxSharpe):
            MaxSharpe = Sharpe_temp
            MaxSharpe_id = i
    Metrics_array=np.array([MaxSharpe])
    return(VAR_in,MaxSharpe, MaxSharpe_id,MinVar_id, Metrics_array)                
#----------------------------------------------------
def calculations_OUT(covIn,covOut, mu):
    points = 100
    G = np.linspace(0.01,100,points)
    VAR_out = np.zeros([points])   
    for i in range(points):
        VAR_out[i] = Risk2_out(G[i],mu,covIn,covOut)
        initial_min = VAR_out[0]
        MaxSharpe = 0
        # MARKOWITZ WEIGHTS
        # Calculate optimal weights for the modified covariance portfolios            
        weights = optPort(covOut, G[i]) 
        Sharpe_temp = np.dot(weights.flatten(), mu)/VAR_out[i]
        if (VAR_out[i]<=initial_min):
            initial_min = VAR_out[i]
            MinVar_id = i
        if (Sharpe_temp>MaxSharpe):
            MaxSharpe = Sharpe_temp
            MaxSharpe_id = i
    Metrics_array=np.array([MaxSharpe])
    return(VAR_out,MaxSharpe, MaxSharpe_id,MinVar_id, Metrics_array)
#----------------------------------------------------
# Frontier graphs made function for proper implementation
def frontier_graph(Cov_in1, Cov_in2, Cov_out1, Cov_out2, 
                         name,name2, mu=None):
    """
        Create efficient frontier graph and calculate the risks for the Markowitz
        portfolio.
        Args:
            Cov_in1 (np.array): Covariance in-sample built with modified eigenvalues.
            Cov_in2 (np.array): Covariance in-sample built with the original eigenvalues.
            Cov_out1 (np.array): Covariance out-sample built with modified eigenvalues.
            Cov_out2 (np.array): Covariance out-sample built with the original eigenvalues.
            name (string): Name of the database.
            name2 (string): Name of the method used to clean the eigenvalues.
            mu (Optional)(np.array): Mean vector of the database.
        Returns:
            Png file with the efficient frontier graph.
            Tuple containing:
                VAR_in1 (np.array): Estimated risks in-sample built with modified eigenvalues.
                VAR_in2 (np.array) : Estimated risks out-sample built with the original eigenvalues.
                VAR_out1 (np.array): Estimated risks out-sample built with modified eigenvalues.
                VAR_out2  (np.array) : Estimated risks out-sample built with the original eigenvalues.
    """
    p=Cov_in1.shape[0]  
    if mu is None: mu = np.ones(p)
    points = 100
    G = np.linspace(0.01,100,points)
    VAR_min = list()
    VAR_max = list()
    G_id1 = list()
    G_id2 = list()
    Metrics_tab = np.zeros([2,1])
    ### IN-SAMPLE
    #clean    
    VAR_in1, MaxSharpe, MaxSharpe_id, MinVar_id, Metrics_tab[0] = calculations_IN(Cov_in1, mu)          
    VAR_min.append(VAR_in1[MinVar_id])
    VAR_max.append(VAR_in1[MaxSharpe_id])
    G_id1.append(G[MinVar_id])
    G_id2.append(G[MaxSharpe_id])

    #dirty    
    VAR_in2, MaxSharpe, MaxSharpe_id, MinVar_id, Metrics_tab[1] = calculations_IN(Cov_in2, mu)
    VAR_min.append(VAR_in2[MinVar_id])
    VAR_max.append(VAR_in2[MaxSharpe_id])
    G_id1.append(G[MinVar_id])
    G_id2.append(G[MaxSharpe_id])
    ### OUT-SAMPLE
    #clean    
    VAR_out1, MaxSharpe, MaxSharpe_id, MinVar_id, Not_Useful = calculations_OUT(Cov_in1,Cov_out1, mu)               
    VAR_min.append(VAR_out1[MinVar_id])
    VAR_max.append(VAR_out1[MaxSharpe_id])
    G_id1.append(G[MinVar_id])
    G_id2.append(G[MaxSharpe_id])
    #dirty    
    VAR_out2, MaxSharpe, MaxSharpe_id, MinVar_id, Not_Useful = calculations_OUT(Cov_in2,Cov_out2, mu)
    VAR_min.append(VAR_out2[MinVar_id])
    VAR_max.append(VAR_out2[MaxSharpe_id])
    G_id1.append(G[MinVar_id])
    G_id2.append(G[MaxSharpe_id])
    
    Metrics_tab=pd.DataFrame(Metrics_tab)
    Metrics_tab.columns=["Razón de Sharpe"]
    Metrics_tab.index = [r"$\xi_i$ In_" + name2,r"$\lambda_i$ In" + name2]
    return (VAR_in1, VAR_in2, VAR_out1, VAR_out2, Metrics_tab)
#---------------------------------------------------
# Portfolio Optimization without constraints
def optPort2(cov, G, mu=None):
    """
    Portfolio optimization without constraints according to Markowtiz model.
    Args:
        cov (np.array): Covariance matrix (Sample estimator).
        G (float): Expected earnings of the portfolio.
        mu (Optional)(np.array): Mean vector of the database.
    Returns:
        w (np.array): Optimized weights of the portfolio.
    """
    if is_invertible(cov):
        inv=np.linalg.inv(cov)
        ones=np.ones(shape=(inv.shape[0],1))
        if mu is None:mu=ones
        if G is None:
            w=np.zeros(shape=(inv.shape[0],1))
        else:
            w=G*np.dot(inv,mu)/np.dot(mu.T,np.dot(inv,mu))
    else:
        w = None        
    return w  
#---------------------------------------------------
# Prado's portofolio NCO method modified for multiple G values.
def optPort_nco2(cov, k, mu=None):
    """
        Optimize portfolios for k cases and return the risks and weights of
        each case.
        Args:
            cov (np.array): Covariance matrix (Sample estimator).
            k (float): Number of cases to optimize the portfolio.
            mu (Optional)(np.array): Mean vector of the database.
        Returns:
            Tuple containing:
                Risks (np.array): Estimated risks for each portfolio.
                weights_matrix (np.array) : Estimated weights of each portfolio.
    """
    cov_df=pd.DataFrame(cov)
    G_cursive= np.linspace(0.01,100,k)
    weights_matrix = np.zeros([k,cov_df.shape[0]])
    Risks = np.zeros(k)
    for h in range(k):
        if mu is not None:mu=pd.Series(mu)
        corr1=cov2corr(cov_df)
        corr1,clstrs,_=clusterKMeansBase(corr1,maxNumClusters=10,n_init=10)
        wIntra=pd.DataFrame(0,index=cov_df.index,columns=clstrs.keys())
        for i in clstrs:
            cov_=cov_df.loc[clstrs[i],clstrs[i]].values
            if mu is None:mu_=None
            else:mu_=mu.loc[clstrs[i]].values.reshape(-1,1)
            temp_w=optPort2(cov_, G_cursive[h], mu_)
            if temp_w is None: break
            wIntra.loc[clstrs[i],i]=temp_w.flatten()
        if temp_w is None: 
            print("break")
            break
        # wIntra = wIntra[(wIntra.T != 0).any()]    
        cov_=wIntra.T.dot(np.dot(cov_df,wIntra)) # reduce covariance matrix
        mu_=(None if mu is None else wIntra.T.dot(mu))
        temp_w=optPort2(cov_, G_cursive[h],mu_)
        wInter=pd.Series(temp_w.flatten(),index=cov_.index)
        nco=wIntra.mul(wInter,axis=1).sum(axis=1).values.reshape(-1,1)
        # Reescaling the weights
        nco2=nco/np.sum(nco)*G_cursive[h]
        Risks[h] = np.dot(nco2.T,(np.dot(cov_df,nco2)))
        
        weights_matrix[h,:] = nco2.flatten()
    return(Risks,weights_matrix)

#--------------------------------------------------------
# Efficient frontier graphs.
def frontier_graph_cluster(Risk_in1, Risk_in2, Risk_out1, Risk_out2, A, Covariances, Risk_T, w_total,mu, name,name2):
    """
        Create and save efficient frontier graphs for the NCO portfolio
        optimization.
        Args:
            VAR_in1 (np.array): Estimated risks in-sample built with modified eigenvalues.
            VAR_in2 (np.array) : Estimated risks out-sample built with the original eigenvalues.
            VAR_out1 (np.array): Estimated risks out-sample built with modified eigenvalues.
            VAR_out2  (np.array) : Estimated risks out-sample built with the original eigenvalues.
            name (string): Name of the database.
            name2 (string): Name of the method used to clean the eigenvalues.
        Returns:
            An efficient frontier graph.
    """
    VAR_min = list()
    VAR_max = list()
    G_id1 = list()
    G_id2 = list()
    Metrics = list()
    ### IN-SAMPLE
    #clean
    G = np.linspace(0.01,100,100)
    
    for B in range(len(G)): 
        Volatility=np.dot(np.dot(w_total[A-1,B],Covariances[A-1]),w_total[A-1,B])
        Sharpe_temp = np.dot(w_total[A-1,B], mu)/Volatility
        #VAR_in1[A,B]
        if(B==0):
            initial_min = Risk_in1[B]
            MinVar_id = B
            MaxSharpe = Sharpe_temp
            MaxSharpe_id = B 
        else:
            if (Risk_in1[B]<=initial_min):
                initial_min = Risk_in1[B]
                MinVar_id = B
            if (Sharpe_temp>MaxSharpe):
                MaxSharpe = Sharpe_temp
                MaxSharpe_id = B                         
    VAR_min.append(Risk_in1[MinVar_id])
    VAR_max.append(Risk_in1[MaxSharpe_id])
    G_id1.append(G[MinVar_id])
    G_id2.append(G[MaxSharpe_id])
    Metrics.append(np.array(MaxSharpe))
    for B in range(len(G)):        
        Volatility=np.dot(np.dot(w_total[0,B],Covariances[0]),w_total[0,B])
        Sharpe_temp = np.dot(w_total[0,B], mu)/Volatility
        #VAR_in1[A,B]
        if(B==0):
            initial_min = Risk_in2[B]
            MinVar_id = B
            MaxSharpe = Sharpe_temp
            MaxSharpe_id = B 
        else:
            if (Risk_in2[B]<=initial_min):
                initial_min = Risk_in2[B]
                MinVar_id = B
            if (Sharpe_temp>MaxSharpe):
                MaxSharpe = Sharpe_temp
                MaxSharpe_id = B         
    VAR_min.append(Risk_in2[MinVar_id])
    VAR_max.append(Risk_in2[MaxSharpe_id])
    G_id1.append(G[MinVar_id])
    G_id2.append(G[MaxSharpe_id])
    Metrics.append(np.array(MaxSharpe))
    for B in range(len(G)):
        Volatility=np.dot(np.dot(w_total[A,B],Covariances[A]),w_total[A,B])
        Sharpe_temp = np.dot(w_total[A,B], mu)/Volatility
        #VAR_in1[A,B]
        if(B==0):
            initial_min = Risk_out2[B]
            MinVar_id = B
            MaxSharpe = Sharpe_temp
            MaxSharpe_id = B 
        else:
            if (Risk_out2[B]<=initial_min):
                initial_min = Risk_out2[B]
                MinVar_id = B
            if (Sharpe_temp>MaxSharpe):
                MaxSharpe = Sharpe_temp
                MaxSharpe_id = B             
    VAR_min.append(Risk_out2[MinVar_id])
    VAR_max.append(Risk_out2[MaxSharpe_id])
    G_id1.append(G[MinVar_id])
    G_id2.append(G[MaxSharpe_id])
    for B in range(len(G)):
        Volatility=np.dot(np.dot(w_total[1,B],Covariances[1]),w_total[1,B])
        Sharpe_temp = np.dot(w_total[1,B], mu)/Volatility
        #VAR_in1[A,B]
        if(B==0):
            initial_min = Risk_out1[B]
            MinVar_id = B
            MaxSharpe = Sharpe_temp
            MaxSharpe_id = B 
        else:
            if (Risk_out1[B]<=initial_min):
                initial_min = Risk_out1[B]
                MinVar_id = B
            if (Sharpe_temp>MaxSharpe):
                MaxSharpe = Sharpe_temp
                MaxSharpe_id = B          
    VAR_min.append(Risk_out1[MinVar_id])
    VAR_max.append(Risk_out1[MaxSharpe_id])
    G_id1.append(G[MinVar_id])
    G_id2.append(G[MaxSharpe_id])
    
    Metrics_tab = pd.DataFrame(Metrics)
    return (Metrics_tab)
#--------------------------------------------------------------------
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
#--------------------------------------------------------------------
def find_idx_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
#-------------------------------------------------------------------
# MAIN FUNCTION
def main(name, method, descriptive, alpha=.995, p=100, nFact=50, double_q=2, qmax=4,portfolio="MinVar"):
    """
        Main function.
        Args:
            Name (string): Name of the database to use.
            Method (string): Determine if it's real or synthetic data.
            alpha (float): Level of alpha to use in Tracy-Widom method and the construction of synthetic data.
            p (integer): Number of variables/columns in the database.
            nFact (integer): Number of factors in the generation of the database.
            q (float): Preferably use q>1. Ratio of n/p. Or also 1/c if you're familiar with that notation.
            Portfolio (string): Determine if we use minimum variance portfolio or real portfolio.
        Returns:
             (tuple): tuple containing:
                mu0 (np.array): The mean of the True matrix.
                cov0 (np.array) : The covariance of the True matrix.        
    """
    if(method=="Prado"): 
        print("Prado")
        cov=np.cov(np.random.normal(size=(int(p*int(double_q/2)),p)),rowvar=0)
        cov0=alpha*cov+(1-alpha)*getRndCov(p,nFact) # noise+signal
        corr0=cov2corr(cov0)
        # Getting eigenvalues and eigenvectors
        return_vec= np.random.multivariate_normal(np.zeros(np.shape(cov0)[0]), cov0, int(p*double_q))
        return_vec=pd.DataFrame(return_vec)
        return_vec.to_csv(name + ".csv")
        col_names = range(return_vec.shape[1])     
        return_vec=np.transpose(return_vec)
    elif(method=="Synthetic"):
        print("Synthetic")
        Z =  np.random.normal(0,1, (p, p*double_q)) # Matriz aleatoria de Wishart
        D = np.diag(np.repeat(0.2, p)) # Factor que hace varianza de .2
        return_vec =  np.dot(D,Z) # con varianza de .2
        return_vec = pd.DataFrame(np.transpose(return_vec))
        return_vec.to_csv(name + ".csv")
        col_names = range(p)
        cov0 = np.cov(return_vec.T) # Mátriz de Covarianza muestral S
        #print(cov0.shape)
        corr0=np.corrcoef(return_vec.T)
        return_vec=np.transpose(return_vec)
    else:
        print("Real")
        string3=name+".csv"
        name = name + "_" + str(test) + "q"
        X =  pd.read_csv(string3)
        # Eliminates first column, because it's the index.
        return_vec = X.drop(X.columns[[0]], axis=1)     
        t1,t2=np.shape(return_vec)
        if t1>t2: return_vec = np.transpose(return_vec) 
        col_names = return_vec.T.columns
        if isinstance(col_names[0],str): col_names = [elem[:4] for elem in col_names] 
        cov0 = np.cov(return_vec)
        p, n = np.shape(return_vec)
        # SI ES POSIBLE, SE USA C=1/(k/2)=1/2 PARA CADA SUBMUESTRA, DE LO CONTRARIO SE USA C=1/(k/2)
        double_q=min(int(n/p),qmax) 
    points=100
    corr0=np.corrcoef(return_vec)
    eVal0,eVec0=getPCA(corr0)   
    n=int(p*double_q)
    n_In=int(n/2)
    return_vec =np.array(return_vec).astype(float)
    x_synth_In=return_vec[:,:int(n/2)]
    x_synth_Out=return_vec[:,int(n/2):int(n)]
    cov0In = np.cov(x_synth_In)
    cov0Out = np.cov(x_synth_Out)

    # Basic Stats Universal
    # basic_stats(return_vec,n,name)
        
    corr0In = np.corrcoef(x_synth_In)
    corr0Out = np.corrcoef(x_synth_Out) 
    return_vec = pd.DataFrame(return_vec)
    x_synth_In=return_vec.iloc[:,:int(n/2)].T
    x_synth_Out=return_vec.iloc[:,int(n/2):int(n)].T
    return_vec.T.to_csv(name + "M.csv")
    x_synth_In.to_csv(name + "In.csv")
    x_synth_Out.to_csv(name + "Out.csv")

    
    twtable=pd.read_csv("twtable_.csv", header=0)
    twtable=twtable.drop(twtable.columns[[0]], axis=1)  # df.columns is zero-based pd.Index
    twtable=np.array(twtable)    

    name2="Recorte"
    eVal1In,CleanIn = clean_corrmatrix(corr0In, n_In)  
    eVal1Out,CleanOut = clean_corrmatrix(corr0Out, n_In)        
    Cov1In=corr2cov_(CleanIn, cov0In)
    Cov1Out=corr2cov_(CleanOut, cov0Out)
    name2="TW"
    eVal2In, CleanIn=TW_Wishart_order2(corr0In, n_In, twtable,alpha)
    eVal2Out, CleanOut=TW_Wishart_order2(corr0Out, n_In, twtable,alpha)
    Cov2In=corr2cov_(CleanIn, cov0In)
    Cov2Out=corr2cov_(CleanOut, cov0Out)
    name2="Lineal"
    eVal3In, CleanIn = Ledoit_Wolf(cov0In, corr0In, n_In)
    eVal3Out, CleanOut = Ledoit_Wolf(cov0Out, corr0Out, n_In)
    Cov3In=corr2cov_(CleanIn, cov0In)
    Cov3Out=corr2cov_(CleanOut, cov0Out)
    name2="No Lineal"
    eVal4In, CleanIn = QUEST(corr0In, n_In, name + "In")
    eVal4Out, CleanOut = QUEST(corr0Out, n_In, name + "Out")
    Cov4In=corr2cov_(CleanIn, cov0In)
    Cov4Out=corr2cov_(CleanOut, cov0Out)

    Covariances = np.stack((cov0In, cov0Out, Cov1In, Cov1Out,
                            Cov2In, Cov2Out, Cov3In, Cov3Out,
                            Cov4In, Cov4Out))    
        
    # Determines the values of mu depending on the portfolio to use.
    if(portfolio!="MinVar"): 
        muIn = np.mean(x_synth_In,axis=0)
        mu = muIn 
    else:
        mu = np.ones(p)
    # Name of the methods used.
    name2 = ["Recorte", "TW", "Lineal", "No lineal"]
    VAR_=np.zeros([10,points])
    Risk_T=np.zeros([10,points])
    w_total=np.zeros([10,points,p])   
    Metrics_Mark = np.zeros([8,1])
    Metrics_cluster = np.zeros([8,1])

    # Calculate optimal weights for the original covariance of the portfolio.
    Risk_T[0], w_total[0]=optPort_nco2(cov0In, points, mu)
    w_total[1]=w_total[0]

    Risk_T[2], w_total[2]=optPort_nco2(Cov1In, points, mu)    
    w_total[3]=w_total[2]

    Risk_T[4], w_total[4]=optPort_nco2(Cov2In, points, mu)
    w_total[5]=w_total[4]

    Risk_T[6], w_total[6]=optPort_nco2(Cov3In, points, mu)
    w_total[7]=w_total[6]

    Risk_T[8], w_total[8]=optPort_nco2(Cov4In, points, mu)
    w_total[9]=w_total[8]
    for B in range(points):
        Risk_T[1,B] = np.dot(w_total[1,B].T,np.dot(cov0Out,w_total[1,B]))  
        Risk_T[3,B] = np.dot(w_total[3,B].T,np.dot(Cov1Out,w_total[3,B]))  
        Risk_T[5,B] = np.dot(w_total[5,B].T,np.dot(Cov2Out,w_total[5,B]))  
        Risk_T[7,B] = np.dot(w_total[7,B].T,np.dot(Cov3Out,w_total[7,B]))  
        Risk_T[9,B] = np.dot(w_total[9,B].T,np.dot(Cov4Out,w_total[9,B]))  
    
    # Markowitz Frontier
    VAR_[2], VAR_[0], VAR_[3], VAR_[1], Metrics_Mark[0:2] = frontier_graph(
        Cov1In, cov0In, Cov1Out, cov0Out, name,name2[0])
    # HRP Frontier 
    Metrics_cluster[0:2] = frontier_graph_cluster(Risk_T[2], Risk_T[0], Risk_T[3], Risk_T[1], 3,
                           Covariances, Risk_T, w_total,mu, name,name2[0])

    # Markowitz Frontier
    VAR_[4], VAR_[0], VAR_[5], VAR_[1], Metrics_Mark[2:4] = frontier_graph(
        Cov2In, cov0In, Cov2Out, cov0Out, name,name2[1])
    # HRP Frontier 
    Metrics_cluster[2:4] = frontier_graph_cluster(Risk_T[4], Risk_T[0], Risk_T[5], Risk_T[1], 5,
                           Covariances, Risk_T, w_total,mu, name,name2[1])

    # Markowitz Frontier
    VAR_[6], VAR_[0], VAR_[7], VAR_[1], Metrics_Mark[4:6] = frontier_graph(
        Cov3In, cov0In, Cov3Out, cov0Out, name,name2[2])
    # HRP Frontier 
    Metrics_cluster[4:6] = frontier_graph_cluster(Risk_T[6], Risk_T[0], Risk_T[7], Risk_T[1], 7,
                           Covariances, Risk_T, w_total,mu, name,name2[2])

    # Markowitz Frontier
    VAR_[8], VAR_[0], VAR_[9], VAR_[1], Metrics_Mark[6:8] = frontier_graph(
        Cov4In, cov0In, Cov4Out, cov0Out, name,name2[3])
    # HRP Frontier 
    Metrics_cluster[6:8] = frontier_graph_cluster(Risk_T[8], Risk_T[0], Risk_T[9], Risk_T[1], 9,
                           Covariances, Risk_T, w_total,mu, name,name2[3])

    MarPas(p, n, eVal0, name)  
    
    if(descriptive == True):
        eigen_graph(cov0,name,p,n_In,alpha,twtable)
        norm_graph(np.array(return_vec),name)   
        Heatmap(corr0,col_names,name) 
          
        
    # IN VS OUT SAMPLE SCORE
    METRICSCORE=np.zeros([5,3])
    G = np.linspace(0.01,100,100)
    sequencerange = [0,2,4,6,8]
    for j in sequencerange:
        MSE=0
        MSE_C=0
        for k in range(100):
            nearest_idx = find_idx_nearest(VAR_[j], VAR_[j+1,k])
            MSE+=(G[nearest_idx]-G[k])**2/G[k]  
            nearest_idx = find_idx_nearest(Risk_T[j], Risk_T[j+1,k])
            MSE_C+=(G[nearest_idx]-G[k])**2/G[k]   
        MSE=MSE/(k+1)
        MSE_C=MSE_C/100
        FrobeniusPt1 = np.sum((Covariances[0] - Covariances[1])**2)
        FrobeniusPt2 = np.sum((Covariances[j] - Covariances[j+1])**2)
        Frobenius = FrobeniusPt2/FrobeniusPt1
        METRICSCORE[int(j/2)]=np.array([MSE, MSE_C, Frobenius])
        
    VAR_2 = pd.DataFrame(VAR_)
    Risk_T_2 = pd.DataFrame(Risk_T) 
    VAR_2.to_csv(name + "MarkowitzRisk_Portfolio.csv")
    Risk_T_2.to_csv(name + "HRPRisk_Portfolio.csv")

    METRICSCORE=pd.DataFrame(METRICSCORE)
    METRICSCORE.columns = ['EMC Marko','EMC HRP',
                           'Frobenius']
    METRICSCORE.index = ['Ninguna','Recorte','TW','Lineal',
                           'No-Lineal']
    METRICSCORE.to_csv(name + "_MetricasCaseras.csv")
    
    Metrics_tab =pd.DataFrame(Metrics_Mark[[0,1,2,4,6]])
    Metrics_tab["HRP"] = Metrics_cluster[[0,1,2,4,6]]
    # Metrics_tab=pd.DataFrame(Metrics_Cumulative2)  
    Metrics_tab.columns=["Razón de Sharpe", "Sharpe HRP"]
    Metrics_tab.index = ["Recorte", "Ninguno", "TW", "Lineal", "No lineal"]   
    
    Metrics_tab.to_csv(name  + "_MetricasFinancieras.csv")
    
    return(Covariances, cov0, VAR_)

# -------------------------------------------------------------------------------

q_test = [100, 10, 2, 11/10]
sizes = [150,120, 100, 80, 50]
nFact_size = 50
method="Synthetic"
descriptive = False
for p_size in sizes:    
    for test in q_test: 
        name = "Data_Storage/Synth_" + str(p_size) + "obs/" + str(p_size) + "obs_" + str(test) + "q_" + str(nFact_size) + "s"
        Covariances, cov0, Risks=main(name, method, descriptive, 
                                      p=p_size, nFact=nFact_size, 
                                      double_q=int(test*2),
                                      portfolio = "MinVar", qmax=int(test*2))

"""
q_test = [100, 10, 2, 11/10]
name = "Data_Storage/BMV/BMV"
descriptive = False
method="Real"
for test in q_test: 
    Covariances, cov0, Risks=main(name, method, descriptive, 
                                  double_q=int(test*2),
                                  portfolio = "MinVar", qmax=int(test*2))
"""