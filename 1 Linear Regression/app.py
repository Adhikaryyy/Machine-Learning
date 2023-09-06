import bmi as bmi

from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# <!--input_data=(age,sex,bmi,children,smoker,region-->
@app.route('/', methods=['post'])
def predict_cost():
    age = int(request.form.get("age"))
    sex = int(request.form.get("sex"))
    bmi_value = float(request.form.get("bmi"))
    children = int(request.form.get("children"))
    smoker = int(request.form.get("smoker"))
    region = int(request.form.get("region"))
    result = model.predict(np.array([age, sex, bmi_value, children, smoker, region]).reshape(1, 6))

    return str(result)



if __name__ == '__main__':
    app.run(debug=True)
