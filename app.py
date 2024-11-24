import cv2 
import os
import shutil
from flask import Flask,request,render_template, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask_sqlalchemy import SQLAlchemy
import json

local_server=True
with open('config.json', 'r') as c:
    params = json.load(c)["params"]

#### Defining Flask App
app = Flask(__name__)

# configure the SQLite database, relative to the app instance folder
if(local_server):
    app.config["SQLALCHEMY_DATABASE_URI"] = params['local_uri']
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = params['prod_uri']

db = SQLAlchemy(app)

class Add_patient(db.Model):
    pat_id = db.Column(db.Integer, primary_key=True)
    pat_fname = db.Column(db.String(200),  nullable=False)
    pat_lname = db.Column(db.String(200),  nullable=False)
    pat_dob = db.Column(db.String(200),  nullable=False)
    pat_age = db.Column(db.Integer, nullable=False)
    pat_addr = db.Column(db.String(200),  nullable=False)
    pat_phone = db.Column(db.String(10),  nullable=False)
    pat_number = db.Column(db.String(200),  nullable=False)
    pat_date_join = db.Column(db.String(200),  nullable=False)
    pat_type = db.Column(db.String(200),  nullable=False)
    pat_ailment = db.Column(db.String(200),  nullable=False)
    pat_gender = db.Column(db.String(200),  nullable=False)

class Add_emp(db.Model):
    doc_id = db.Column(db.Integer, primary_key=True)
    doc_fname = db.Column(db.String(200),  nullable=False)
    doc_lname = db.Column(db.String(200),  nullable=False)
    doc_number = db.Column(db.String(200),  nullable=False)
    doc_email = db.Column(db.String(200), nullable=False)
    doc_phone = db.Column(db.String(200),  nullable=False)
    doc_dept = db.Column(db.String(200),  nullable=False)

class Medical_records(db.Model):
    mdr_id = db.Column(db.Integer, primary_key=True)
    mdr_number = db.Column(db.String(200),  nullable=False)
    mdr_pat_name = db.Column(db.String(200),  nullable=False)
    mdr_pat_gender = db.Column(db.String(200),  nullable=False)
    mdr_pat_age = db.Column(db.String(200), nullable=False)
    mdr_pat_adr = db.Column(db.String(200),  nullable=False)
    mdr_pat_phone = db.Column(db.String(200),  nullable=False)
    mdr_pat_number = db.Column(db.String(200),  nullable=False)
    mdr_pat_type = db.Column(db.String(200),  nullable=False)
    mdr_pat_ailment	 = db.Column(db.String(200),  nullable=False)
    mdr_pat_prescr = db.Column(db.String(200),  nullable=False)
    file_name = db.Column(db.String(500),  nullable=False)
    mdr_date_rec = db.Column(db.String(200),  nullable=False)

class Vitals(db.Model):
    pat_v_id = db.Column(db.Integer, primary_key=True)
    pat_v_fname = db.Column(db.String(200),  nullable=False)
    pat_v_lname = db.Column(db.String(200),  nullable=False)
    pat_v_number = db.Column(db.String(200),  nullable=False)
    body_temp = db.Column(db.String(200), nullable=False)
    heart_rate = db.Column(db.String(200),  nullable=False)
    resp_rate = db.Column(db.String(200),  nullable=False)
    blood_pres = db.Column(db.String(200),  nullable=False)
    date = db.Column(db.String(200),  nullable=False)



#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name, Roll, Phone, Type')

#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')

def extract_attendance():
    attendance_file_path = f'Attendance/Attendance-{datetoday}.csv'
    if not os.path.isfile(attendance_file_path):
        return [], [], [], [], 0

    df = pd.read_csv(attendance_file_path)
    if 'Pat_Number' not in df.columns:
        return [], [], [], [], 0

    pat_fname = df['Name']
    pat_number = df['Roll']
    pat_phone = df['Phone']
    pat_type = df['Type']
    l = len(df)
    return pat_fname, pat_number, pat_phone, pat_type, l

#### Our main page
@app.route('/', methods = ['GET', 'POST'])
def home():
    outpatient_count = Add_patient.query.filter_by(pat_type='OutPatient').count()
    inpatient_count = Add_patient.query.filter_by(pat_type='InPatient').count()
    employee_count = Add_emp.query.count()
    employees = Add_emp.query.all()
    return render_template('dashboard.html', params=params,outpatient_count=outpatient_count, inpatient_count=inpatient_count, employee_count=employee_count, employees=employees) 

@app.route('/add_emp', methods = ['GET', 'POST'])
def add_employee():
    if(request.method == 'POST'):
        doc_fname = request.form.get('doc_fname')
        doc_lname = request.form.get('doc_lname')
        doc_number = request.form.get('doc_number')
        doc_email = request.form.get('doc_email')
        doc_phone = request.form.get('doc_phone')
        doc_dept = request.form.get('doc_dept')
        entry = Add_emp(doc_fname=doc_fname, doc_lname=doc_lname, doc_number=doc_number, doc_email=doc_email, doc_phone=doc_phone, doc_dept=doc_dept)
        db.session.add(entry)
        db.session.commit()  
    return render_template('add_employee.html', params=params) 

@app.route('/add_medical_record', methods = ['GET', 'POST'])
def add_medical_record():
    rows = Add_patient.query.all()  
    return render_template('add_medical_record.html', rows=rows, params=params) 

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/add_single_patient_medical_record/<string:pat_number>')
def add_single_patient_medical_record(pat_number):
    patient = Add_patient.query.filter_by(pat_number=pat_number).first()

    if patient:
        # If a patient is found, return a template with the patient details
        return render_template('add_single_patient_medical_record.html', patient=patient)
    else:
        # If no patient is found, return a 404 response or a custom error page
        return "Patient not found", 404
    # return render_template('view_single_patient.html', params=params)

UPLOAD_FOLDER = r'C:\Users\arssh\OneDrive\Desktop\MedVision\static\patients'

@app.route('/medical_records', methods = ['GET', 'POST'])
def medical_records():
    if(request.method == 'POST'):
        mdr_number = request.form.get('mdr_number')
        mdr_pat_name = request.form.get('mdr_pat_name')
        mdr_pat_gender = request.form.get('mdr_pat_gender')
        mdr_pat_age = request.form.get('mdr_pat_age')
        mdr_pat_adr = request.form.get('mdr_pat_adr')
        mdr_pat_phone = request.form.get('mdr_pat_phone')
        mdr_pat_number = request.form.get('mdr_pat_number')
        mdr_pat_type = request.form.get('mdr_pat_type')
        mdr_pat_ailment = request.form.get('mdr_pat_ailment')
        mdr_pat_prescr = request.form.get('mdr_pat_prescr')

        prescr_filename = None
        if 'mdr_pat_prescr' in request.files:
            mdr_pat_prescr = request.files['mdr_pat_prescr']
            
            # Create a folder with patient number if it doesn't exist
            folder_path = os.path.join(UPLOAD_FOLDER, mdr_pat_number)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            current_datetime = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

            # Save the prescription file in the folder
            prescr_filename = secure_filename(f"{current_datetime}_{mdr_pat_prescr.filename}")
            prescr_filepath = os.path.join(folder_path, prescr_filename)
            mdr_pat_prescr.save(prescr_filepath)

            # Save filename to the database record (this is crucial for deletion later)
            record = Medical_records.query.filter_by(mdr_number=mdr_pat_number).first()
            if record:
                record.filename = prescr_filename
                db.session.commit()

        entry = Medical_records(mdr_number=mdr_number, mdr_pat_name=mdr_pat_name,mdr_pat_gender=mdr_pat_gender, mdr_pat_age=mdr_pat_age, mdr_pat_adr=mdr_pat_adr,mdr_pat_phone=mdr_pat_phone, mdr_pat_number=mdr_pat_number, mdr_pat_type=mdr_pat_type, mdr_pat_ailment=mdr_pat_ailment, mdr_pat_prescr=mdr_pat_prescr, file_name=prescr_filename, mdr_date_rec=datetime.now())
        db.session.add(entry)
        db.session.commit()  
    rows = Medical_records.query.all()
    return render_template('manage_medical_record.html', params=params, rows=rows) 

@app.route('/patient/<pat_number>')
def patient_prescriptions(pat_number):
    folder_path = os.path.join(UPLOAD_FOLDER, pat_number)
    prescription_files = os.listdir(folder_path) if os.path.exists(folder_path) else []
    return render_template('patient_prescriptions.html', pat_number=pat_number, prescription_files=prescription_files)

@app.route('/add_patient', methods = ['GET', 'POST'])
def register_patient():  
    if(request.method == 'POST'):
        pat_fname = request.form.get('pat_fname')
        pat_lname = request.form.get('pat_lname')
        pat_dob = request.form.get('pat_dob')
        pat_age = request.form.get('pat_age')
        pat_addr = request.form.get('pat_addr')
        pat_phone = request.form.get('pat_phone')
        pat_number = request.form.get('pat_number')
        pat_type = request.form.get('pat_type')
        pat_ailment = request.form.get('pat_ailment')
        pat_gender = request.form.get('pat_gender')

        entry = Add_patient(pat_fname=pat_fname, pat_lname=pat_lname, pat_dob=pat_dob, pat_age=pat_age, pat_addr=pat_addr, pat_phone=pat_phone, pat_number=pat_number, pat_date_join=datetime.now(),pat_type=pat_type, pat_ailment=pat_ailment, pat_gender=pat_gender)
        db.session.add(entry)
        db.session.commit()

    return render_template('register_patient.html', params=params) 

@app.route('/manage_patient', methods = ['GET', 'POST'])
def manage_patient():  
    rows = Add_patient.query.all()
    return render_template('manage_patient.html', rows=rows, params=params) 

@app.route('/delete/<string:pat_number>', methods=['GET', 'POST'])
def delete_patient(pat_number):
    patient = Add_patient.query.filter_by(pat_number=pat_number).first()
    records = Medical_records.query.filter_by(mdr_pat_number=pat_number).all()
    vitals = Vitals.query.filter_by(pat_v_number=pat_number).all()

    if patient:
        # Construct the folder path
        userimagefolder = 'static/faces/' + patient.pat_fname + '_' + str(patient.pat_number)

        db.session.delete(patient)
        db.session.commit()
        
        # Delete the folder and its contents
        if os.path.isdir(userimagefolder):
            shutil.rmtree(userimagefolder)

    if records:
        for record in records:
            # Construct the medical records folder path
            medical_record_folder = os.path.join(UPLOAD_FOLDER, record.mdr_pat_number)
            db.session.delete(record)
            db.session.commit()
            
            # Delete the medical records folder and its contents if it exists
            if os.path.isdir(medical_record_folder):
                shutil.rmtree(medical_record_folder)

    if vitals:
        for vital in records:
            db.session.delete(vital)
        db.session.commit()

    rows = Add_patient.query.all()
    return render_template('manage_patient.html', rows=rows, params=params) 

@app.route('/view/<string:pat_number>')
def view_patient(pat_number):
    patient = Add_patient.query.filter_by(pat_number=pat_number).first()
    vitals = Vitals.query.filter_by(pat_v_number=pat_number).all()  # Get all vital records for the patient

    if patient:
        # If a patient is found, return a template with the patient details and vitals
        folder_path = os.path.join(UPLOAD_FOLDER, pat_number)
        prescription_files = os.listdir(folder_path) if os.path.exists(folder_path) else []

        return render_template('view_single_patient.html', patient=patient, prescription_files=prescription_files, vitals=vitals)
    
    else:
        # If no patient is found, return a 404 response or a custom error page
        return "Patient not found", 404

@app.route('/vitals_patient/<string:pat_number>')
def vitals_patient(pat_number):
    patient = Add_patient.query.filter_by(pat_number=pat_number).first()

    if patient:
        # If a patient is found, return a template with the patient details
        return render_template('vitals_patient.html', patient=patient)
    else:
        # If no patient is found, return a 404 response or a custom error page
        return "Patient not found", 404
    # return render_template('view_single_patient.html', params=params)

@app.route('/vitals', methods = ['GET', 'POST'])
def vitals():
    if(request.method == 'POST'):
        pat_v_fname = request.form.get('pat_v_fname')
        pat_v_lname = request.form.get('pat_v_lname')
        pat_v_number = request.form.get('pat_v_number')
        body_temp = request.form.get('body_temp')
        heart_rate = request.form.get('heart_rate')
        resp_rate = request.form.get('resp_rate')
        blood_pres = request.form.get('blood_pres')

        entry = Vitals(pat_v_fname=pat_v_fname, pat_v_lname=pat_v_lname,pat_v_number=pat_v_number, body_temp=body_temp, heart_rate=heart_rate,resp_rate=resp_rate, blood_pres=blood_pres, date=datetime.now())
        db.session.add(entry)
        db.session.commit()  
    outpatient_count = Add_patient.query.filter_by(pat_type='OutPatient').count()
    inpatient_count = Add_patient.query.filter_by(pat_type='InPatient').count()
    employee_count = Add_emp.query.count()
    employees = Add_emp.query.all()
    return render_template('dashboard.html', params=params,outpatient_count=outpatient_count, inpatient_count=inpatient_count, employee_count=employee_count, employees=employees) 

@app.route('/update/<string:pat_number>', methods=['GET', 'POST'])
def update_patient(pat_number):
    patient = Add_patient.query.filter_by(pat_number=pat_number).first()

    if request.method == 'POST':
        if patient:
            patient.pat_fname = request.form['pat_fname']
            patient.pat_lname = request.form['pat_lname']
            patient.pat_dob = request.form['pat_dob']
            patient.pat_age = request.form['pat_age']
            patient.pat_addr = request.form['pat_addr']
            patient.pat_phone = request.form['pat_phone']
            patient.pat_ailment = request.form['pat_ailment']
            patient.pat_type = request.form['pat_type']
            
            db.session.commit()
            return redirect(url_for('home'))
        else:
            return "Patient not found", 404

    if patient:
        return render_template('update_single_patient.html', patient=patient)
    else:
        return "Patient not found", 404

# Manage Medical Record
@app.route('/manage_medical_record', methods = ['GET', 'POST'])
def manage_medical_record():  
    rows = Medical_records.query.all()
    return render_template('manage_medical_record.html', rows=rows, params=params) 

@app.route('/delete_record/<string:mdr_number>', methods=['GET', 'POST'])
def delete_record(mdr_number):
    record = Medical_records.query.filter_by(mdr_number=mdr_number).first()

    if record:
        # Construct the folder path
        folder_path = os.path.join(UPLOAD_FOLDER, record.mdr_pat_number)
        
        # Assuming the filename is stored in the record and there is only one file per record
        # Adjust according to your actual database schema
        if record.file_name:
            file_path = os.path.join(folder_path, record.file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Delete the record from the database
        db.session.delete(record)
        db.session.commit()

    rows = Medical_records.query.all()
    return render_template('manage_medical_record.html', rows=rows, params=params)

# In patient
@app.route('/inpatient', methods = ['GET', 'POST'])
def inpatient():  
    rows = Add_patient.query.filter_by(pat_type='InPatient').all()
    return render_template('inpatient.html', rows=rows, params=params) 

# Out patient
@app.route('/outpatient', methods = ['GET', 'POST'])
def outpatient():  
    rows = Add_patient.query.filter_by(pat_type='OutPatient').all()
    return render_template('outpatient.html', rows=rows, params=params) 

@app.route('/employee_record', methods = ['GET', 'POST'])
def employee_record():  
    rows = Add_emp.query.all()
    return render_template('employee_record.html', rows=rows, params=params) 

# Manage Employee
@app.route('/manage_employee', methods = ['GET', 'POST'])
def manage_employee():  
    rows = Add_emp.query.all()
    return render_template('manage_employee.html', rows=rows, params=params) 

@app.route('/delete_employee/<string:doc_number>', methods=['GET', 'POST'])
def delete_employee(doc_number):
    # employee = Add_emp.query.get(doc_number)
    employee = Add_emp.query.filter_by(doc_number=doc_number).first()

    if employee:
        db.session.delete(employee)
        db.session.commit()
        return redirect(url_for('home'))
    else:
        return "Employee with ID {} not found".format(doc_number), 404

@app.route('/view_employee/<string:doc_number>')
def view_employee(doc_number):
    employee = Add_emp.query.filter_by(doc_number=doc_number).first()


    if employee:
        # If a patient is found, return a template with the patient details
        return render_template('view_single_employee.html', employee=employee)
    else:
        # If no patient is found, return a 404 response or a custom error page
        return "Employee not found", 404


@app.route('/update_employee/<string:doc_number>', methods=['GET', 'POST'])
def update_employee(doc_number):
    employee = Add_emp.query.filter_by(doc_number=doc_number).first()

    if request.method == 'POST':
        if employee:
            employee.doc_fname = request.form['doc_fname']
            employee.doc_lname = request.form['doc_lname']
            employee.doc_number = request.form['doc_number']
            employee.doc_email = request.form['doc_email']
            employee.doc_phone = request.form['doc_phone']
            employee.doc_dept = request.form['doc_dept']
            
            db.session.commit()
            return redirect(url_for('home'))
        else:
            return "Employee not found", 404

    if employee:
        return render_template('update_single_employee.html', employee=employee)
    else:
        return "Employee not found", 404

#### This function will run when we click on Take Attendance Button
# @app.route('/start', methods=['GET'])
# def start():
#     if 'face_recognition_model.pkl' not in os.listdir('static'):
#         outpatient_count = Add_patient.query.filter_by(pat_type='OutPatient').count()
#         inpatient_count = Add_patient.query.filter_by(pat_type='InPatient').count()
#         employee_count = Add_emp.query.count()
#         return render_template('dashboard.html',outpatient_count=outpatient_count, inpatient_count=inpatient_count, employee_count=employee_count) 

#     cap = cv2.VideoCapture(0)
#     ret = True
#     identified_person = None
#     patient_info = None
    
#     while ret:
#         ret, frame = cap.read()
#         faces = extract_faces(frame)
#         if len(faces) > 0:  # Check if any face is detected
#             (x, y, w, h) = faces[0]  # Assuming only one face is detected
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
#             face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))  # Corrected indexing
#             identified_person = identify_face(face.reshape(1, -1))[0]
#             add_attendance(identified_person)
#             print(identified_person)
#             cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
#             break  # Exit the loop once a face is identified
#         cv2.imshow('Attendance', frame)
#         if cv2.waitKey(1) == 27:
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

#     if identified_person:
#         pat_number = identified_person.split('_')[1]
#         folder_path = os.path.join(UPLOAD_FOLDER, pat_number)
#         prescription_files = os.listdir(folder_path) if os.path.exists(folder_path) else []
#         vitals = Vitals.query.filter_by(pat_v_number=pat_number).all()  # Get all vital records for the patient
#         return redirect(url_for('view_single_patient', pat_number=pat_number, prescription_files=prescription_files, vitals=vitals))
#     outpatient_count = Add_patient.query.filter_by(pat_type='OutPatient').count()
#     inpatient_count = Add_patient.query.filter_by(pat_type='InPatient').count()
#     employee_count = Add_emp.query.count()
#     return render_template('dashboard.html',outpatient_count=outpatient_count, inpatient_count=inpatient_count, employee_count=employee_count)
app.secret_key = os.urandom(24)  # Set a random secret key

@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        outpatient_count = Add_patient.query.filter_by(pat_type='OutPatient').count()
        inpatient_count = Add_patient.query.filter_by(pat_type='InPatient').count()
        employee_count = Add_emp.query.count()
        employees = Add_emp.query.all()
        return render_template('dashboard.html', outpatient_count=outpatient_count, inpatient_count=inpatient_count, employee_count=employee_count, employees=employees) 

    cap = cv2.VideoCapture(0)
    ret = True
    identified_person = None
    
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        if len(faces) > 0:  # Check if any face is detected
            (x, y, w, h) = faces[0]  # Assuming only one face is detected
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))  # Corrected indexing
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            print(identified_person)
            cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            break  # Exit the loop once a face is identified
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

    if identified_person:
        pat_number = identified_person.split('_')[1]
        folder_path = os.path.join(UPLOAD_FOLDER, pat_number)
        prescription_files = os.listdir(folder_path) if os.path.exists(folder_path) else []
        vitals = Vitals.query.filter_by(pat_v_number=pat_number).all()  # Get all vital records for the patient

        # Store patient info in the session
        patient = Add_patient.query.filter_by(pat_number=pat_number).first()
        session['recent_patient'] = {
            'name': patient.pat_fname,
            'number': patient.pat_number,
            'phone': patient.pat_phone,
            'type': patient.pat_type
        }

        return redirect(url_for('view_single_patient', pat_number=pat_number, prescription_files=prescription_files, vitals=vitals))
    
    outpatient_count = Add_patient.query.filter_by(pat_type='OutPatient').count()
    inpatient_count = Add_patient.query.filter_by(pat_type='InPatient').count()
    employee_count = Add_emp.query.count()
    recent_patient = session.get('recent_patient', None)
    employees = Add_emp.query.all()
    return render_template('dashboard.html', outpatient_count=outpatient_count, inpatient_count=inpatient_count, employee_count=employee_count, recent_patient=recent_patient, employees=employees)

@app.route('/dashboard', methods=['GET'])
def dashboard():
    outpatient_count = Add_patient.query.filter_by(pat_type='OutPatient').count()
    inpatient_count = Add_patient.query.filter_by(pat_type='InPatient').count()
    employee_count = Add_emp.query.count()
    employees = Add_emp.query.all()
    
    recent_patient = session.get('recent_patient', None)
    
    return render_template('dashboard.html', outpatient_count=outpatient_count, inpatient_count=inpatient_count, employee_count=employee_count, recent_patient=recent_patient, employees=employees)


#### Add Attendance of a specific user
def add_attendance(name):
    pat_fname = name.split('_')[0]
    pat_number = name.split('_')[1]
    
    patient = Add_patient.query.filter_by(pat_fname=pat_fname, pat_number=pat_number).first()
    
    datetoday = pd.to_datetime('today').strftime('%Y-%m-%d')
    attendance_file = f'Attendance/Attendance-{datetoday}.csv'

    # Check if the attendance file exists
    if not os.path.isfile(attendance_file):
        df = pd.DataFrame(columns=['Pat_Fname', 'Pat_Number', 'Pat_Type', 'Time'])
        df.to_csv(attendance_file, index=False)
    else:
        df = pd.read_csv(attendance_file)

    if str(pat_number) not in df['Pat_Number'].astype(str).tolist():
        current_time = datetime.now().strftime("%H:%M:%S")
        with open(attendance_file, 'a') as f:
            f.write(f'\n{patient.pat_fname},{patient.pat_number},{patient.pat_type},{current_time}')
    # Return patient info for rendering on the dashboard
    return {
        'pat_fname': patient.pat_fname,
        'pat_number': patient.pat_number,
        'pat_phone': patient.pat_phone,
        'pat_type': patient.pat_type,
    }

@app.route('/view_single_patient/<pat_number>')
def view_single_patient(pat_number):
    patient = Add_patient.query.filter_by(pat_number=pat_number).first()
    vitals = Vitals.query.filter_by(pat_v_number=pat_number).all()  # Get all vital records for the patient

    if patient:
        folder_path = os.path.join(UPLOAD_FOLDER, pat_number)
        prescription_files = os.listdir(folder_path) if os.path.exists(folder_path) else []
        return render_template('view_single_patient.html', patient=patient, prescription_files=prescription_files, vitals=vitals)
    outpatient_count = Add_patient.query.filter_by(pat_type='OutPatient').count()
    inpatient_count = Add_patient.query.filter_by(pat_type='InPatient').count()
    employee_count = Add_emp.query.count()
    employees = Add_emp.query.all()
    return render_template('dashboard.html',outpatient_count=outpatient_count, inpatient_count=inpatient_count, employee_count=employee_count, employees=employees)

#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    pat_fname = request.form['pat_fname']
    pat_number = request.form['pat_number']
    userimagefolder = 'static/faces/'+pat_fname+'_'+str(pat_number)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/20',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = pat_fname+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==200:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break



    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    if(request.method == 'POST'):
        pat_fname = request.form.get('pat_fname')
        pat_lname = request.form.get('pat_lname')
        pat_dob = request.form.get('pat_dob')
        pat_age = request.form.get('pat_age')
        pat_addr = request.form.get('pat_addr')
        pat_phone = request.form.get('pat_phone')
        pat_number = request.form.get('pat_number')
        pat_type = request.form.get('pat_type')
        pat_ailment = request.form.get('pat_ailment')
        pat_gender = request.form.get('pat_gender')

        entry = Add_patient(pat_fname=pat_fname, pat_lname=pat_lname, pat_dob=pat_dob, pat_age=pat_age, pat_addr=pat_addr, pat_phone=pat_phone, pat_number=pat_number, pat_date_join=datetime.now(),pat_type=pat_type, pat_ailment=pat_ailment, pat_gender=pat_gender)
        db.session.add(entry)
        db.session.commit()

    pat_fname,pat_number,pat_phone, pat_type,l = extract_attendance()
    outpatient_count = Add_patient.query.filter_by(pat_type='OutPatient').count()
    inpatient_count = Add_patient.query.filter_by(pat_type='InPatient').count()
    employee_count = Add_emp.query.count()   
    employees = Add_emp.query.all() 
    return render_template('dashboard.html', pat_fname=pat_fname, pat_number=pat_number, pat_phone=pat_phone, pat_type=pat_type, l=l,outpatient_count=outpatient_count, inpatient_count=inpatient_count, employee_count=employee_count, employees=employees) 


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)