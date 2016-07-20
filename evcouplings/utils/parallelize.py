"""
Author: Eli Draizen
"""

#Standard libraries
import os, sys
import logging

#Requried Libraries
from joblib import Parallel, delayed
try:
    #Import libraries to use use cluster
    from cluster_helper.cluster import cluster_view
    
    try:
        import json

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cluster_info.json")) as ci:
            cluster_info = json.load(ci)
        
        try:
            scheduler, queue = cluster_info["scheduler"], cluster_info["queue"]
            use_cluster = scheduler is not None and queue is not None
        
        except KeyError:
            use_cluster = False

    except IOError:
        use_cluster = False

except ImportError:
    #Default to joblib
    use_cluster = False
    scheduler, queue = None, None

def cluster_call(args):
    """Private function to load on the cluster. This copies the origina pythonpath
    and environment so the cluster can access the all of the same modules, and calls
    the appropriate function.

    Parameters
    ----------
    args : list of Arguments in the following order
        obj : object or None
            Initialized object that contains methid name. If None, assumes 
            method is a function in __main__
        function : str
            Name of method of function to parallelize
        environmnt : dict
            The original envirment variables from os.environ
        *args : ...
            Any arguments to pass to function

    Returns
    -------
    The ouput from the method or function
    """
    try:
        function, environment = args[:2]
    except IndexError:
        raise RuntimeError("cluster_call must contain obj, func_name, and environ before the arguments")
    
    arguments = args[2:] if len(args) > 2 else []

    os.environ.update(environment)
    try:
        sys.path += os.environ["PYTHONPATH"].split(":")
    except KeyError:
        pass

    if hasattr(function, '__call__'):
        return function(*arguments)
    else:
        raise RuntimeError("Function {} not found".format(function))

def method_call(*a):
    return getattr(a[0], a[1])(*a[2:])

def map(func, args, n_jobs=None, use_joblib=False):
    """Parallelize a function or method using multiple cores or on a cluster 
    with joblib or ipython-cluster-helper respectively. To use cluster,
    ipython-cluster-helper must be installed and global parameters 'use_cluster',
    'scheduler', and 'queue' must be set. These can be set automatically from file 
    'cluster_info.json' in this directory. This is done automically during setup 
    by calling:
        python setup.py use_cluster install --schduler=lsf --queue=mcore

    Note: If parallelizing a method, the object that owns the method must be 
    added into each parallel process due to python pickeling limitations.

    Parameters
    ----------
    func : function or Obj.method
        Function or method to parallelize
    args : list
        Arguments to pass into func. Use list or tuple to pass multiple argeuments.
    n_jobs : int
        The maximum number of concurrently running jobs, such as the number of 
        Python worker processes when backend=multiprocessing or the size of 
        the thread-pool when backend=threading. If -1 all CPUs are used. If 1 
        is given, no parallel computing code is used at all, which is useful for 
        debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for 
        n_jobs = -2, all CPUs but one are used (Default). 
            Taken from joblib documentation.
    use_joblib : bool
        Force the use of joblib, even if a cluster has been initialized. 
        Default is False.

    Returns
    -------
    results : list
        A list of the output of each call to function
    """
    args = [list(a) if isinstance(a, (list, tuple)) else [a] for a in args]

    environment = [os.environ] if use_cluster and not use_joblib else []
    if hasattr(func, "im_class"):
        #Update function to call parent object with the method name
        method = [func.__self__, func.__name__]
        args = [[method_call]+environment+method+a for a in args]
        func_name = "{}.{}".format(func.__self__.__class__.__name__, func.__name__)
        func = method_call
    else:
        func_name = func.__name__
        if use_cluster and not use_joblib:
            args = [[func]+environment+a for a in args]

    if not use_cluster or use_joblib:
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel((delayed(func)(*arg) for arg in args))
    else:
        resources = {'resources': "W=6:0;minconcores=2", 'tag':"evtools-{}".format(func_name)}
        global scheduler, queue 
        with cluster_view(scheduler=scheduler, queue=queue, cores_per_job=os.environ.get("num_cores", 2), num_jobs=len(args), extra_params=resources) as view:
            results = view.map(cluster_call, args)
        

    return results
