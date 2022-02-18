### CHAOS STATISTICS
### Delay, Dimension and Lyapunov

import math # Mathematic operations
import nolds # Nonlinear measures for dynamical systems
import numpy as np # Array processing for numbers, strings, records, and objects
import sklearn.metrics # Machine Learning in Python
import sklearn.neighbors # Machine Learning in Python
from operator import sub # Standard operators as functions
from toolz import curry # List processing tools and functional utilities

def chaos_analysis(x):   
    # Chaos analyis of the time series
    # Use Eckmann et al algorithm for lyapunov exponents and FNN for embedding dimension
    # Returns the time delay, the embedding dimension and the lyapunov spectrum
    # Returns caos=1 if existe deterministic chaos otherwise caos=0
    lag=delay(x)
    mmax=2*int(np.floor(2*math.log10(len(x))))+1 
    fnn=global_false_nearest_neighbors(x, lag, min_dims=1, max_dims=mmax)            
    if len(fnn[1][fnn[1]<=0.15])!=0:
        m=np.where(fnn[1]<=0.15)[0][0]+1
        lyapunov=nolds.lyap_e(x,emb_dim=2*(m)-1,matrix_dim=m,tau=lag)
        if sum(lyapunov)<0 and max(lyapunov)>0:
            caos=1
        else:
            caos=0 
    else:
        caos=0
        m=99
        lyapunov=99
    return lag,m,lyapunov,caos
  
def dynamical_analysis(x,steps):
    lag=delay(x)
    mmax=2*int(np.floor(2*math.log10(len(x))))+1 
    if lag*mmax>len(x):
        mmax=len(x)/lag
    fnn=global_false_nearest_neighbors(x, lag, min_dims=1, max_dims=mmax)            
    if len(fnn[1][fnn[1]<=0.15])!=0:
        m=np.where(fnn[1]<=0.15)[0][0]+1
    else:
        m=mmax
    return[lag,m]
  
def construct_initial(x, lag, n_dims):
    phase_space=reconstruct(x, lag, n_dims)
    distances=sklearn.metrics.pairwise.euclidean_distances(phase_space,phase_space) 
    search=[]
    for i in range(len(phase_space)-1):
        ini=phase_space[i,:]
        obj=phase_space[i+1,:]
        dist_temp=np.array(distances[:,i])
        dist_temp[i]=np.max(dist_temp)   
        arg_min=np.argmin(dist_temp[i])
        nearest_ini=phase_space[arg_min]
        nearest_obj=phase_space[arg_min+1]
        vector=np.concatenate((ini, nearest_ini, nearest_obj, obj), axis=0)
        search.append(vector)
    vector=np.array(search)
    vector_x=vector[:,:-1]
    vector_y=vector[:,-1]
    return vector_x,vector_y

def construct(x, lag, n_dims):
    phase_space=reconstruct(x, lag, n_dims)
    ini=phase_space[-1,:]
    distances=sklearn.metrics.pairwise.euclidean_distances(phase_space[:-1],np.reshape(ini,(1,n_dims)))
    nearest_ini=phase_space[np.argmin(distances)]
    nearest_obj=phase_space[np.argmin(distances)+1]  
    obj= phase_space[-lag,1:]  
    vector=np.concatenate((ini, nearest_ini, nearest_obj, obj), axis=0)
    return vector
 
def delay(x):
    # Returns the optimal Time-Delay from a time series
    # Use the autocorrelation function and the mutual information score
    da=0;dm=0
    n = len(x)
    y = x-x.mean()
    r = np.correlate(y, y, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(y[:n-k]*y[-(n-k):]).sum() for k in range(n)]))
    auto = r/(x.var()*(np.arange(n, 0, -1)))
    while (auto[da]*auto[da+1])>0:
        da=da+1
    da=da+1    
    while sklearn.metrics.mutual_info_score(None,None,contingency=np.histogram2d(x, np.roll(x,dm),20)[0])>=sklearn.metrics.mutual_info_score(None,None, contingency=np.histogram2d(x, np.roll(x,dm+1),20)[0]):
        dm=dm+1
    lag=min(da,dm)+1
    return lag

def global_false_nearest_neighbors(x, lag, min_dims=1, max_dims=15, **cutoffs):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    x = _vector(x)
    dimensions = np.arange(min_dims, max_dims + 1)
    false_neighbor_pcts = np.array([_gfnn(x, lag, n_dims, **cutoffs) for n_dims in dimensions])
    return dimensions, false_neighbor_pcts


def _gfnn(x, lag, n_dims, **cutoffs):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    # Global false nearest neighbors at a particular dimension.
    # Returns percent of all nearest neighbors that are still neighbors when the next dimension is unfolded.
    # Neighbors that can't be embedded due to lack of data are not counted in the denominator.
    offset = lag*n_dims
    is_true_neighbor = _is_true_neighbor(x, _radius(x), offset)
    return np.mean([
        not is_true_neighbor(indices, distance, **cutoffs)
        for indices, distance in _nearest_neighbors(reconstruct(x, lag, n_dims))
        if (indices + offset < x.size).all()
    ])
      
@curry
def _is_true_neighbor(x, attractor_radius, offset, indices, distance,relative_distance_cutoff=15, relative_radius_cutoff=2):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    distance_increase = np.abs(sub(*x[indices + offset])) 
    return (distance_increase / distance < relative_distance_cutoff and
            distance_increase / attractor_radius < relative_radius_cutoff)
    
def _nearest_neighbors(y):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    distances, indices = sklearn.neighbors.NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(y).kneighbors(y)
    for distance, index in zip(distances, indices):
        yield index, distance[1]

def _radius(x):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    return np.sqrt(((x - x.mean())**2).mean())

def reconstruct(x, lag, n_dims):
    # create the delayed vector from a time serie
    x = _vector(x)
    lags = lag * np.arange(n_dims)
    return np.vstack(x[lag:lag - lags[-1] or None] for lag in lags).transpose()

def deconstruct(x, lag, n_dims):
    # create the time serie from a delayed vector
    dec=np.empty(len(x)+lag*(n_dims-1))* np.nan
    dec[:len(x)]=x[:,0]
    dec[len(x):]=x[-lag*(n_dims-1):,-1]
    return dec

def _vector(x):
    # Created by hsharrison
    # pypsr taken from https://github.com/hsharrison/pypsr MIT License
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError('x(t) must be a 1-dimensional signal')
    return x    
