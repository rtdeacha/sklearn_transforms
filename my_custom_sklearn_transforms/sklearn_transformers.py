from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
    

class over_under(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X):
        return self
    
    def transform(self, X):
        return self
    
    def fit_transform(self, X):
        
        features = X.iloc[:,:-1]
        target = X.iloc[:,-1]
        # Primero hacemos over sobre la variable minoritaria
        over = SMOTE()
        features, target = over.fit_resample(features,target)
        
        # Luego hacemos under sobre la variable mayoritaria
        under = RandomUnderSampler()
        features, target = under.fit_resample(features,target)
        
        X = pd.concat([pd.DataFrame(features),pd.DataFrame(target).rename(columns ={0:'OBJETIVO'})], axis=1).rename(columns = self.columns, inplace = False)
        
        return X 
