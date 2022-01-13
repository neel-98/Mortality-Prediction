import flask
import pickle
import pandas as pd
import numpy as np
import sys
import joblib
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
# server=app.server
with open(f'model/covid_classification.pkl', 'rb') as f:
    covid_c_model = pickle.load(f)
with open(f'model/covid_scaler.pkl', 'rb') as f:
    covid_scaler = pickle.load(f)

with open(f'model/sofa_classification.pkl', 'rb') as f:
    sofa_c_model = pickle.load(f)
with open(f'model/sofa_scaler.pkl', 'rb') as f:
    sofa_scaler = pickle.load(f)

# with open(f'model/covid_regression.hdf5', 'rb') as f:

covid_r_model = tf.keras.models.load_model(f'model/covid_regression.hdf5')
sofa_r_model = tf.keras.models.load_model(f'model/sofa_regression.hdf5')



app = flask.Flask(__name__, template_folder='templates')

templet = ""


@app.route('/', methods=['GET', 'POST'])
def main():

    if flask.request.method == 'GET':
        return(flask.render_template('homepage.html'))

    if flask.request.method == 'POST' and flask.request.form.get('Covid') == 'Covid':
        return (flask.render_template('covid.html'))

    if flask.request.method == 'POST' and flask.request.form.get('Sofa Score') == 'Sofa Score':
        return (flask.render_template('sofaScore.html'))

    if flask.request.method == 'POST' and flask.request.form.get('Predict Covid') == 'Predict Covid':
        sex = flask.request.form['sex']
        sex = int(sex)
        intubated = flask.request.form['intubated']
        intubated = int(intubated)
        pneumonia = flask.request.form['pneumonia']
        pneumonia = int(pneumonia)
        age = flask.request.form['age']
        age = int(age)
        pregnancy = flask.request.form['pregnancy']
        pregnancy = int(pregnancy)
        diabetes = flask.request.form['diabetes']
        diabetes = int(diabetes)
        copd = flask.request.form['copd']
        copd = int(copd)
        asthma = flask.request.form['asthma']
        asthma = int(asthma)
        immunosuppression = flask.request.form['immunosuppression']
        immunosuppression = int(immunosuppression)
        hypertension = flask.request.form['hypertension']
        hypertension = int(hypertension)
        other_disease = flask.request.form['other_disease']
        other_disease = int(other_disease)
        cardiovascular = flask.request.form['cardiovascular']
        cardiovascular = int(cardiovascular)
        obesity = flask.request.form['obesity']
        obesity = int(obesity)
        renal_chronic = flask.request.form['renal_chronic']
        renal_chronic = int(renal_chronic)
        tobacco = flask.request.form['tobacco']
        tobacco = int(tobacco)
        icu = flask.request.form['icu']
        icu = int(icu)
        hospitalized = flask.request.form['hospitalized']
        hospitalized = int(hospitalized)

        input_variables = pd.DataFrame([[sex, intubated, pneumonia, age, pregnancy, diabetes, copd, asthma, immunosuppression, hypertension, other_disease, cardiovascular, obesity, renal_chronic, tobacco, icu, hospitalized]], columns=[
            'sex', 'intubated', 'pneumonia', 'age', 'pregnancy', 'diabetes', 'copd', 'asthma', 'immunosuppression', 'hypertension', 'other_disease', 'cardiovascular', 'obesity', 'renal_chronic', 'tobacco', 'icu', 'hospitalized'])
        # input_variables = in_variables.to_numeric()
        death = covid_c_model.predict(input_variables)[0]
        
        message = ''
        
        if(death == 1):  
            normalized_data = covid_scaler.transform(input_variables)
            days = covid_r_model.predict(normalized_data)[0]
            
            d = int(days)
            h = int((days - d) * 24)
        
            message = '<h1 style = "font-size = 30px; color:red;">Patient may die in {} days and {} hours.</h1>'.format(
            d,h)
        else:
            message = '<h1 style = "font-size = 30px; color:green;">Patient is Safe.</h1>'
            
        prediction = message
        return flask.render_template('covid.html', original_input={'sex': sex, 'intubated': intubated, 'pneumonia': pneumonia, 'age': age, 'pregnancy': pregnancy, 'diabetes': diabetes, 'copd': copd, 'asthma': asthma, 'immunosuppression': immunosuppression, 'hypertension': hypertension, 'other_disease': other_disease, 'cardiovascular': cardiovascular, 'obesity': obesity, 'renal_chronic': renal_chronic, 'tobacco': tobacco, 'icu': icu, 'hospitalized': hospitalized}, result=prediction)

    if flask.request.method == 'POST' and flask.request.form.get('Predict') == 'Predict':
        Age = flask.request.form['Age']
        Age = int(Age)
        Gender = flask.request.form['Gender']
        Gender = int(Gender)
        BMI = flask.request.form['BMI']
        BMI = float(BMI)
        Apache3_score = flask.request.form['Apache3_score']
        Apache3_score = int(Apache3_score)
        Blood_presure = flask.request.form['Blood_presure']
        Blood_presure = int(Blood_presure)
        Heart_rate = flask.request.form['Heart_rate']
        Heart_rate = int(Heart_rate)
        Oxygen_saturation = flask.request.form['Oxygen_saturation']
        Oxygen_saturation = int(Oxygen_saturation)
        Myocardial_infarction = flask.request.form['Myocardial_infarction']
        Myocardial_infarction = int(Myocardial_infarction)
        Congestive_heart_failure = flask.request.form['Congestive_heart_failure']
        Congestive_heart_failure = int(Congestive_heart_failure)
        Cerebro_vascular_accident = flask.request.form['Cerebro_vascular_accident']
        Cerebro_vascular_accident = int(Cerebro_vascular_accident)
        Kidney_disease_severity = flask.request.form['Kidney_disease_severity']
        Kidney_disease_severity = int(Kidney_disease_severity)
        Cormorbility_score = flask.request.form['Cormorbility_score']
        Cormorbility_score = int(Cormorbility_score)
        Diabetes_mellitus = flask.request.form['Diabetes_mellitus']
        Diabetes_mellitus = int(Diabetes_mellitus)
        Prior_cancer = flask.request.form['Prior_cancer']
        Prior_cancer = int(Prior_cancer)
        Prior_lung_disease = flask.request.form['Prior_lung_disease']
        Prior_lung_disease = int(Prior_lung_disease)
        RESPsofa = flask.request.form['RESPsofa']
        RESPsofa = int(RESPsofa)
        COAGsofa = flask.request.form['COAGsofa']
        COAGsofa = int(COAGsofa)
        LIVERsofa = flask.request.form['LIVERsofa']
        LIVERsofa = int(LIVERsofa)
        CVsofa = flask.request.form['CVsofa']
        CVsofa = int(CVsofa)
        CNSsofa = flask.request.form['CNSsofa']
        CNSsofa = int(CNSsofa)
        RENALsofa = flask.request.form['RENALsofa']
        RENALsofa = int(RENALsofa)
        Respiration_rate = flask.request.form['Respiration_rate']
        Respiration_rate = int(Respiration_rate)
        Diastolic = flask.request.form['Diastolic']
        Diastolic = int(Diastolic)
        Glasgow_coma_scale = flask.request.form['Glasgow_coma_scale']
        Glasgow_coma_scale = int(Glasgow_coma_scale)
        Ventilation = flask.request.form['Ventilation']
        Ventilation = int(Ventilation)
        Previous_Dialysis = flask.request.form['Previous_Dialysis']
        Previous_Dialysis = int(Previous_Dialysis)
        On_dialysis = flask.request.form['On_dialysis']
        On_dialysis = int(On_dialysis)
        onintraaorticbaloonpump = flask.request.form['onintraaorticbaloonpump']
        onintraaorticbaloonpump = int(onintraaorticbaloonpump)
        Preanesthesia_checkup = flask.request.form['Preanesthesia_checkup']
        Preanesthesia_checkup = int(Preanesthesia_checkup)
        Packed_red_blood_cells = flask.request.form['Packed_red_blood_cells']
        Packed_red_blood_cells = int(Packed_red_blood_cells)
        Cardiogenic_shock = flask.request.form['Cardiogenic_shock']
        Cardiogenic_shock = int(Cardiogenic_shock)
        Cardiomyopathy = flask.request.form['Cardiomyopathy']
        Cardiomyopathy = int(Cardiomyopathy)
        Heart_failure = flask.request.form['Heart_failure']
        Heart_failure = int(Heart_failure)
        Irregular_heartbeat = flask.request.form['Irregular_heartbeat']
        Irregular_heartbeat = int(Irregular_heartbeat)
        Cardiac_arrest = flask.request.form['Cardiac_arrest']
        Cardiac_arrest = int(Cardiac_arrest)
        Acute_coronary_syndrome = flask.request.form['Acute_coronary_syndrome']
        Acute_coronary_syndrome = int(Acute_coronary_syndrome)
        Coronary_artery_disease = flask.request.form['Coronary_artery_disease']
        Coronary_artery_disease = int(Coronary_artery_disease)
        Angiogram_hospital_stay = flask.request.form['Angiogram_hospital_stay']
        Angiogram_hospital_stay = int(Angiogram_hospital_stay)
        Coronary_stent = flask.request.form['Coronary_stent']
        Coronary_stent = int(Coronary_stent)
        SOFA_score = flask.request.form['SOFA_score']
        SOFA_score = int(SOFA_score)
        OASIS_score = flask.request.form['OASIS_score']
        OASIS_score = int(OASIS_score)

        input_variables1 = pd.DataFrame([[Age, Gender, BMI, Apache3_score, Blood_presure, Heart_rate, Oxygen_saturation, Myocardial_infarction, Congestive_heart_failure, Cerebro_vascular_accident, Kidney_disease_severity, Cormorbility_score, Diabetes_mellitus, Prior_cancer, Prior_lung_disease, RESPsofa, COAGsofa, LIVERsofa, CVsofa, CNSsofa, RENALsofa, Respiration_rate, Diastolic, Glasgow_coma_scale, Ventilation, Previous_Dialysis, On_dialysis, onintraaorticbaloonpump, Preanesthesia_checkup, Packed_red_blood_cells, Cardiogenic_shock, Cardiomyopathy, Heart_failure, Irregular_heartbeat, Cardiac_arrest, Acute_coronary_syndrome, Coronary_artery_disease, Angiogram_hospital_stay, Coronary_stent, SOFA_score, OASIS_score]], columns=[
            'Age', 'Gender', 'BMI', 'Apache3_score', 'Blood_presure', 'Heart_rate', 'Oxygen_saturation', 'Myocardial_infarction', 'Congestive_heart_failure', 'Cerebro_vascular_accident', 'Kidney_disease_severity', 'Cormorbility_score', 'Diabetes_mellitus', 'Prior_cancer', 'Prior_lung_disease', 'RESPsofa', 'COAGsofa', 'LIVERsofa', 'CVsofa', 'CNSsofa', 'RENALsofa', 'Respiration_rate', 'Diastolic', 'Glasgow_coma_scale', 'Ventilation', 'Previous_Dialysis', 'On_dialysis', 'onintraaorticbaloonpump', 'Preanesthesia_checkup', 'Packed_red_blood_cells', 'Cardiogenic_shock', 'Cardiomyopathy', 'Heart_failure', 'Irregular_heartbeat', 'Cardiac_arrest', 'Acute_coronary_syndrome', 'Coronary_artery_disease', 'Angiogram_hospital_stay', 'Coronary_stent', 'SOFA_score', 'OASIS_score'])
        # input_variables = in_variables.to_numeric()
        death = sofa_c_model.predict(input_variables1)[0]
        print(input_variables1)
        message = ''
        print(death)    
        if(death == 1):  
            normalized_data = sofa_scaler.transform(input_variables1[['Age','BMI','Blood_presure','Heart_rate','Oxygen_saturation','Respiration_rate','Diastolic','Apache3_score','OASIS_score']])
            normalized_data = pd.DataFrame(normalized_data, columns=['Age','BMI','Blood_presure','Heart_rate','Oxygen_saturation','Respiration_rate','Diastolic','Apache3_score','OASIS_score'])
            for column in ['Age','BMI','Blood_presure','Heart_rate','Oxygen_saturation','Respiration_rate','Diastolic','Apache3_score','OASIS_score']:
                input_variables1[column] = normalized_data[column]
            print(input_variables1)    
            days = sofa_r_model.predict(input_variables1)[0]
            
            d = int(days)
            h = int((days - d) * 24)
        
            message = '<h1 style = "font-size = 30px; color:red;">Patient may die in {} days and {} hours.</h1>'.format(
            d,h)
        else:
            message = '<h1 style = "font-size = 30px; color:green;">Patient is Safe.</h1>'
            
        prediction = message
        
        return flask.render_template('sofaScore.html', original_input={'Age': Age, 'Gender': Gender, 'BMI': BMI, 'Apache3_score': Apache3_score, 'Blood_presure': Blood_presure, 'Heart_rate': Heart_rate, 'Oxygen_saturation': Oxygen_saturation, 'Myocardial_infarction': Myocardial_infarction, 'Congestive_heart_failure': Congestive_heart_failure, 'Cerebro_vascular_accident': Cerebro_vascular_accident, 'Kidney_disease_severity': Kidney_disease_severity, 'Cormorbility_score': Cormorbility_score, 'Diabetes_mellitus': Diabetes_mellitus, 'Prior_cancer': Prior_cancer, 'Prior_lung_disease': Prior_lung_disease, 'RESPsofa': RESPsofa, 'COAGsofa': COAGsofa, 'LIVERsofa': LIVERsofa, 'CVsofa': CVsofa, 'CNSsofa': CNSsofa, 'RENALsofa': RENALsofa, 'Respiration_rate': Respiration_rate, 'Diastolic': Diastolic, 'Glasgow_coma_scale': Glasgow_coma_scale, 'Ventilation': Ventilation, 'Previous_Dialysis': Previous_Dialysis, 'On_dialysis': On_dialysis, 'onintraaorticbaloonpump': onintraaorticbaloonpump, 'Preanesthesia_checkup': Preanesthesia_checkup, 'Packed_red_blood_cells': Packed_red_blood_cells, 'Cardiogenic_shock': Cardiogenic_shock, 'Cardiomyopathy': Cardiomyopathy, 'Heart_failure': Heart_failure, 'Irregular_heartbeat': Irregular_heartbeat, 'Cardiac_arrest': Cardiac_arrest, 'Acute_coronary_syndrome': Acute_coronary_syndrome, 'Coronary_artery_disease': Coronary_artery_disease, 'Angiogram_hospital_stay': Angiogram_hospital_stay, 'Coronary_stent': Coronary_stent, 'SOFA_score': SOFA_score, 'OASIS_score': OASIS_score}, result=prediction)


if __name__ == '__main__':
    app.run(debug=True)
