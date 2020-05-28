from sklearn.metrics import confusion_matrix

def logistic_model(X, y):
    from sklearn.linear_model import LogisticRegression
    
    #select the columns with numerical/categorical data
    #X = X.loc[:, (X.dtypes == np.float64) or (X.dtypes == np.int64)]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y)
    
    
   
    
    #fit model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_valid, y_valid)
    coeffs = pd.DataFrame(model.coef_, columns=X_train.columns)
    print(coeffs)
    tn, fp, fn, tp = confusion_matrix(model.predict(X_valid), y_valid).ravel()
    
    return model, accuracy, (tn, fp, fn, tp)


def SGD_model(X, y):
    from sklearn.linear_model import SGDClassifier
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y)
    
    #fit model
    model = SGDClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_valid, y_valid)
    tn, fp, fn, tp = confusion_matrix(model.predict(X_valid), y_valid).ravel()
  
    
    return model, accuracy, (tn, fp, fn, tp)
    
def rf_model(X,y):
    from sklearn.ensemble import RandomForestClassifier
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y)
    
    #fit model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_valid, y_valid)
    tn, fp, fn, tp = confusion_matrix(model.predict(X_valid), y_valid).ravel()
    
    
    return model, accuracy, (tn, fp, fn, tp)

def gradient_model(X,y):
    from sklearn.ensemble import GradientBoostingClassifier
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y)
    
    #fit model
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_valid, y_valid)
    tn, fp, fn, tp = confusion_matrix(model.predict(X_valid), y_valid).ravel()
    
    
    return model, accuracy, (tn, fp, fn, tp)