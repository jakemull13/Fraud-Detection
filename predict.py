from joblib import dump, load


with open('test_script_examples.csv') as fd:
    reader=csv.reader(fd)
    interestingrow=[row for idx, row in enumerate(reader) if idx in (0,np.random.randint(1,100))][1]

interestingrow

l=[]
for x in interestingrow:
    try:l.append(float(x))
    except:l.append(x)



input=pd.Series(l[:-1])
truth=pd.Series(l[-1])


#unpickle model
model = load('modelpickle.joblib')

#predict
prediction = model.predict_proba(np.array(input).reshape(1,-1))

