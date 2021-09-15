# load all the required libraries
from flask import Flask,jsonify,request,render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

# create object
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

# Get file From User
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'



# create end point to  train your model and save training data in pickle file
@app.route('/train_model')
def train():
    # load data
    data = pd.read_excel('False Alarm Cases.xlsx')
    # split columns
    x = data.iloc[:, 1:7]
    y = data['Spuriosity Index(0/1)']
    # create object for Algo class
    logm = LogisticRegression()
    # train the model
    logm.fit(x, y)
    # Save trainig results in pickle file
    joblib.dump(logm, 'train.pkl')
    completed= 'Training is Completed'
    return render_template('index.html',completed=completed)


#  load pickle file and test your model, pass test data via POSt method
#  First we need to load pickle file for it to get training data ref
@app.route('/test_model', methods=['POST'])
def test():
    # load pickle file
    pkl_file = joblib.load('train.pkl')

    f1 = request.form['Ambient_Temperature']
    f2 = request.form['Calibration']
    f3 = request.form['Unwanted_substance_deposition']
    f4 = request.form['Humidity']
    f5 = request.form['H2S_Content']
    f6 = request.form['detected_by']
    my_test_data = [f1, f2, f3, f4, f5, f6]
    my_data_array = np.array(my_test_data)
    test_array = my_data_array.reshape(1, 6)
    df_test = pd.DataFrame(test_array,
                           columns=['Ambient Temperature', 'Calibration', 'Unwanted substance deposition', 'Humidity',
                                    'H2S Content', 'detected by'])
    y_pred = pkl_file.predict(df_test)
    Fault= ' False Alram , NO Danger'
    Tru= ' True Alram , Danger'
    if y_pred == 1:
        return render_template('index.html',fal=Fault)
        print("false ")
    else:
        return render_template('index.html',tru=Tru)
        print("true")

#  run the application on port
app.run(port=5000,debug=True)


