from flask import Flask,render_template,request,url_for 
import pandas as pd
import numpy as np 
import pickle 
scaler=pickle.load(open("models/scaled.pkl","rb"))
svc=pickle.load(open("models/svc.pkl","rb"))
application=Flask(__name__)
app=application 

@app.route("/",methods=['POST','GET'])
def predict():

  

    if request.method=="POST":
        SL=float(request.form.get("sepal_length"))
        SW=float(request.form.get("sepal_width"))
        PL=float(request.form.get("petal_length"))
        PW=float(request.form.get("petal_width"))
        data=[[SL,SW,PL,PW]] 
        Sdata=scaler.transform(data)
        result=svc.predict(data)
        print(result)
        if result[0]==0:
            img_url_setosa=url_for('static', filename='photos/Iris-setosa.png') 
            return render_template("main.html",prediction="Iris Setosa",source=img_url_setosa)
        elif result[0]==1:
            img_url_versicolor=url_for('static', filename='photos/versicolor.png') 
            return render_template("main.html",prediction="Iris Versicolor",source=img_url_versicolor)
        elif result[0]==2:
            img_url_virginica=url_for('static', filename='photos/virginica.png')
            return render_template("main.html",prediction="Iris Virginica",source=img_url_virginica)

    return render_template("main.html") 

if __name__=="__main__":
    
    app.run(debug=1)

