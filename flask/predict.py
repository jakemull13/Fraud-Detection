from joblib import dump, load
import csv
import numpy as np
import pandas as pd

def pred():

#    with open('test_script_examples.csv') as fd:
#        reader=csv.reader(fd)
#        interestingrow=[row for idx, row in enumerate(reader) if #idx in               (0,np.random.randint(1,101))][1]
#    
    #interestingrow
    
    with open('test_script_examples.csv') as fd:
        reader=csv.reader(fd)
        interestingrows=[row for idx, row in enumerate(reader)]
        interestingrow = interestingrows[np.random.randint(1,100)]
    
    
    
    l=[]
    for x in interestingrow:
        try:l.append(float(x))
        except:l.append(x)
    
    
    
    transaction=pd.Series(l[:-1])
    truth=pd.Series(l[-1])
    
    
    #unpickle model
    model = load('modelpickle.joblib')
    
    #predict
    prediction = model.predict(np.array(transaction).reshape(1,-1))
    proba = model.predict_proba(np.array(transaction).reshape(1,-1))
    
    #print(proba)
#    print(f'We predict that this is {prediction}. Chance of Fraud     = ####{proba[1]} kjdsflkajsdlkfjlksjdflkajdsf')
#    return str(proba)

    return f'We predict that this is {prediction}. Chance of Fraud = {proba[0][1]}'
