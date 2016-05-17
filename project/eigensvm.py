class EigenSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=100, nu=0.1, gamma=0.5):
        self.n_components = n_components
        self.nu = nu
        self.gamma = gamma
    
    def to_eigenspace(self, X):
        #computing PCA
        pca = PCA(n_components=self.n_components, copy=True)
        pca.fit(X)
        #storing pca object
        self.pca = pca
        return pca.transform(X)
    
    def build_classes(self, X, y, spc):
        #dimensions of input matrix
        M,N = X.shape
        #indexes of X
        indexes = np.arange(M)
        np.random.shuffle(indexes)
        #dimensions of output matrices
        P = int(sp.misc.comb(spc,2))*152
        Q = self.n_components
        """
        > building the within class difference set C1
        > all combinations between samples of the same
          class are computed
        """
        C1 = np.empty((P,Q))
        p = 0 #index of C1
        for m in xrange(0, M, spc):
            for i in xrange(m, m+spc):
                for j in xrange(i+1, m+spc):
                    #difference space representation
                    C1[p] = np.abs(X[i] - X[j])
                    p += 1
        """
        > building the between class difference set
        > randomly selecting two samples of different classes
        """
        C2 = np.empty((P,Q))
        p = 0 #index of C2
        while p < P:
            i,j = np.random.choice(indexes, 2)
            #indexes must be of different classes
            if y[i]==y[j]: continue
            C2[p] = np.abs(X[i] - X[j])
            p += 1
        #return both classes
        return (C1,C2)
    
    def fit(self, X, y):
        #counting samples per class
        frec = np.bincount(y-1)
        spc = frec[0]
        #if any class has different frec
        if not np.all(spc==frec): return -1
        
        #to eigenspace
        eig_X = self.to_eigenspace(X)
        
        #building the classes
        C1,C2 = self.build_classes(eig_X, y, spc)
        data = np.concatenate((C1,C2), axis=0)
        #label arrays
        lab1 = np.ones(C1.shape[0],dtype=int)
        lab2 = 2*lab1
        labels = np.concatenate((lab1,lab2))
        
        #building svm with stratified cross validation
        strat_kf = StratifiedKFold(labels, n_folds=5, shuffle=True)
        #parameters to try
        Nu = np.linspace(0.1, 0.5, 5, endpoint=True)
        Gamma = np.linspace(0.25, 1.5, 5, endpoint=True)
        params = {'nu':Nu, 'gamma':Gamma}
        #grid search
        clf = svm.NuSVC(kernel='rbf')
        gs = grid_search.GridSearchCV(clf, params, cv=strat_kf, n_jobs=2)
        gs.fit(data,labels)
        #fitting with best parameters
        eigsvm = svm.NuSVC(kernel='rbf', nu=gs.best_params_['nu'], gamma=gs.best_params_['gamma'])
        eigsvm.fit(data,labels)
        
        #storing important things to make predictions
        self.spc = spc
        self.data = eig_X
        self.eigsvm = eigsvm
        return self
    
    def predict(self, X):
        #X data to spatial histogram format
        X = self.to_eigenspace(X)
        M = X.shape[0]
        predictions = np.empty(M, dtype=np.uint8)
        for m in xrange(M):
            diff = np.abs(X[m]-self.data)
            df = self.eigsvm.decision_function(diff)
            N = df.shape[0]
            min_val = np.inf
            min_ind = 0
            for i in xrange(0,N,self.spc):
                mean_score = np.sum(df[i:i+self.spc])/self.spc
                if mean_score < min_val:
                    min_val = mean_score
                    min_ind = i/self.spc
            predictions[m] = min_ind
        #shift to match the labels
        predictions += 1
        return predictions
    
    def score(self, X, y):
        y_pd = self.predict(X)
        return precision(y, y_pd)


"""
Stratified 5-fold cross validation y Grid search
para determinar el mejor parametro Nu y Gamma en rbf svm
"""
def cross_eigsvm(X, y, N, Nu, Gamma, n_folds):
    #generating stratified 5-fold cross validation iterator
    strat_kf = StratifiedKFold(y, n_folds=n_folds, shuffle=True)
    #parameters to try
    params = {'N':N, 'nu':Nu, 'gamma':Gamma}
    #Setting grid search for rbf-svm
    clf = dSVM()
    gs = grid_search.GridSearchCV(clf, params, cv=strat_kf, n_jobs=2)
    #make it
    gs.fit(X, y)
    #return best parameters and grid scores
    return gs.best_params_['N'], gs.best_params_['nu'], gs.best_params_['gamma'], gs.grid_scores_
    

def solve_eigsvm(dataset, spc, verbose=False):
    #samples per class on training and testing set
    spc_tr = spc
    spc_ts = 20-spc_tr
    #training and testing paths
    tr_path = './db/train'+dataset[-2:]+'/tr-{0}pc-{1}/'
    ts_path = './db/test'+dataset[-2:]+'/ts-{0}pc-{1}/'
    #errors through all datasets
    tr_err = list()
    ts_err = list()
    #iterating through datasets
    for set_num in xrange(20):
        #loading training and testing sets
        X_tr,y_tr = load_data(tr_path.format(spc_tr,set_num))
        X_ts,y_ts = load_data(ts_path.format(spc_ts,set_num))
        #choosing best nu (and gamma) through stratified
        #5-fold cross-validation and grid search
        #n,nu,gamma = cross_dsvm(X_tr, y_tr, N, Nu, Gamma, n_folds=spc)
        #fitting the model with this parameters
        clf = EigenSVM()    
        clf.fit(X_tr,y_tr)
        #computing training error
        tr_err.append(1.-clf.score(X_tr,y_tr))
        #computing testing error
        ts_err.append(1.-clf.score(X_ts,y_ts))
        if verbose:
            print "#####################################################################################"
            print "{0}: {1} samples per class (dataset {2})".format(dataset, spc, set_num)
            print "Training error rate: {0}".format(tr_err[-1])
            print "Testing error rate: {0}".format(ts_err[-1])
        #releasing memory of big objects
        del X_tr, X_ts, clf
    return np.array(tr_err),np.array(ts_err)