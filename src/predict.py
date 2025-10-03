import sys, joblib
pipe = joblib.load("models/model.joblib")
text = " ".join(sys.argv[1:]) or "Stocks surge as market rallies"
pred = pipe.predict([text])[0]
print(pred)
