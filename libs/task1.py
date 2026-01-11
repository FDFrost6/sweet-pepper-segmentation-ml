class KNN():
        def __init__(self, K=1 ):
            self.K = K

        def fit(self, X, y ):
            self.X = X
            self.y = y

        def euclid(self, x ):
            resids = self.X - x
            sqrd = resids**2
            smmd = np.sum( sqrd, axis=1 )
            return np.sqrt( smmd )

        def predict(self, X ):
            cls = np.zeros( (X.shape[0],) )
            for i, x in enumerate( X ):
                dist = self.euclid( x )
                ids = dist.argsort()[:self.K]
                nn = self.y[ids]
                cls[i], _ = mode( nn )
            return cls
        



