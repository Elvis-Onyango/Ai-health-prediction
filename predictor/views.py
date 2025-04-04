import os
import joblib
import pandas as pd
from django.contrib.auth.decorators import login_required, user_passes_test
from django.conf import settings
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.db.models import Avg
from .forms import HeartDiseaseForm, KidneyDiseaseForm, DiabetesForm, PatientProfileForm, DoctorProfileForm
from .models import Doctor, Patient, KidneyDiseasePrediction, HeartDiseasePrediction, DiabetesPrediction
from django.contrib.auth import get_user_model
from django.db.models.functions import TruncDate
from django.db.models import Count,Avg
import json
from django.utils import timezone
from datetime import timedelta
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  


User = get_user_model()


heart_model_path = os.path.join(settings.BASE_DIR, "predictor", "stacking_model.pkl")
heart_scaler_path = os.path.join(settings.BASE_DIR, "predictor", "heart_scaler.pkl")

kidney_model_path = os.path.join(settings.BASE_DIR, "predictor", "kidney_model.pkl")
kidney_scaler_path = os.path.join(settings.BASE_DIR, "predictor", "kidney_scaler.pkl")

diabetes_model_path = os.path.join(settings.BASE_DIR, "predictor", "diabetes_model.pkl")
diabetes_scaler_path = os.path.join(settings.BASE_DIR, "predictor", "scaler_diabetes.pkl")

# Load models and scalers
heart_model = joblib.load(heart_model_path)
heart_scaler = joblib.load(heart_scaler_path)

kidney_model = joblib.load(kidney_model_path)
kidney_scaler = joblib.load(kidney_scaler_path)

diabetes_model = joblib.load(diabetes_model_path)
diabetes_scaler = joblib.load(diabetes_scaler_path)

# Feature sets for each prediction
HEART_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Load the selected features
selected_features = [
    'Age (yrs)', 
    'Blood Pressure (mm/Hg)', 
    'Specific Gravity', 
    'Albumin', 
    'Sugar', 
    'Blood Glucose Random (mgs/dL)', 
    'Blood Urea (mgs/dL)', 
    'Serum Creatinine (mgs/dL)', 
    'Sodium (mEq/L)', 
    'Potassium (mEq/L)', 
    'Hemoglobin (gms)', 
    'Packed Cell Volume', 
    'White Blood Cells (cells/cmm)', 
    'Red Blood Cells (millions/cmm)', 
    'Red Blood Cells: normal', 
    'Pus Cells: normal', 
    'Pus Cell Clumps: present', 
    'Bacteria: present', 
    'Hypertension: yes', 
    'Diabetes Mellitus: yes', 
    'Coronary Artery Disease: yes', 
    'Appetite: poor', 
    'Pedal Edema: yes', 
    'Anemia: yes'
]




DIABETES_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]




@login_required
def predict_heart_view(request):
    result = None
    if request.method == "POST":
        form = HeartDiseaseForm(request.POST)
        if form.is_valid():
            try:
                input_data = [
                    int(form.cleaned_data['age']),
                    int(form.cleaned_data['sex']),
                    int(form.cleaned_data['cp']),
                    int(form.cleaned_data['trestbps']),
                    int(form.cleaned_data['chol']),
                    int(form.cleaned_data['fbs']),
                    int(form.cleaned_data['restecg']),
                    int(form.cleaned_data['thalach']),
                    int(form.cleaned_data['exang']),
                    float(form.cleaned_data['oldpeak']),
                    int(form.cleaned_data['slope']),
                    int(form.cleaned_data['ca']),
                    int(form.cleaned_data['thal'])
                ]

                input_df = pd.DataFrame([input_data], columns=HEART_FEATURES)
                input_scaled = heart_scaler.transform(input_df)
                prediction = heart_model.predict(input_scaled)[0]
                probability = heart_model.predict_proba(input_scaled)[0][1] * 100  

                # Save result in HeartDiseasePrediction model
                HeartDiseasePrediction.objects.create(
                    patient=request.user.patient_profile,
                    doctor=request.user.doctor_profile if hasattr(request.user, 'doctor_profile') else None,
                    age=input_data[0],
                    sex=input_data[1],
                    cp=input_data[2],
                    trestbps=input_data[3],
                    chol=input_data[4],
                    fbs=input_data[5],
                    restecg=input_data[6],
                    thalach=input_data[7],
                    exang=input_data[8],
                    oldpeak=input_data[9],
                    slope=input_data[10],
                    ca=input_data[11],
                    thal=input_data[12],
                    prediction_result=prediction,
                    probability=probability
                )

                result = {'prediction': int(prediction), 'probability': round(probability, 2)}

            except Exception as e:
                print(f"Error during heart prediction: {e}")
                result = {'error': str(e)}

        else:
            print("Heart Disease Form is not valid")
            print(form.errors.as_json())
    else:
        form = HeartDiseaseForm()

    return render(request, "heart_predict.html", {"form": form, "result": result})





def predict_kidney_view(request):
    result = None
    form = KidneyDiseaseForm(request.POST or None)
    
    if request.method == "POST" and form.is_valid():
        try:
            # Convert boolean fields to numerical (1/0) before scaling
            input_data = {
                'Age (yrs)': float(form.cleaned_data['age']),
                'Blood Pressure (mm/Hg)': float(form.cleaned_data['blood_pressure']),
                'Specific Gravity': float(form.cleaned_data['specific_gravity']),
                'Albumin': float(form.cleaned_data['albumin']),
                'Sugar': float(form.cleaned_data['sugar']),
                'Blood Glucose Random (mgs/dL)': float(form.cleaned_data['blood_glucose_random']),
                'Blood Urea (mgs/dL)': float(form.cleaned_data['blood_urea']),
                'Serum Creatinine (mgs/dL)': float(form.cleaned_data['serum_creatinine']),
                'Sodium (mEq/L)': float(form.cleaned_data['sodium']),
                'Potassium (mEq/L)': float(form.cleaned_data['potassium']),
                'Hemoglobin (gms)': float(form.cleaned_data['hemoglobin']),
                'Packed Cell Volume': float(form.cleaned_data['packed_cell_volume']),
                'White Blood Cells (cells/cmm)': float(form.cleaned_data['white_blood_cells']),
                'Red Blood Cells (millions/cmm)': float(form.cleaned_data['red_blood_cells']),
                'Red Blood Cells: normal': 1.0 if form.cleaned_data['red_blood_cells_normal'] else 0.0,
                'Pus Cells: normal': 1.0 if form.cleaned_data['pus_cells_normal'] else 0.0,
                'Pus Cell Clumps: present': 1.0 if form.cleaned_data['pus_cell_clumps_present'] else 0.0,
                'Bacteria: present': 1.0 if form.cleaned_data['bacteria_present'] else 0.0,
                'Hypertension: yes': 1.0 if form.cleaned_data['hypertension'] else 0.0,
                'Diabetes Mellitus: yes': 1.0 if form.cleaned_data['diabetes_mellitus'] else 0.0,
                'Coronary Artery Disease: yes': 1.0 if form.cleaned_data['coronary_artery_disease'] else 0.0,
                'Appetite: poor': 1.0 if not form.cleaned_data['appetite'] else 0.0,  # Note the inversion
                'Pedal Edema: yes': 1.0 if form.cleaned_data['pedal_edema'] else 0.0,
                'Anemia: yes': 1.0 if form.cleaned_data['anemia'] else 0.0
            }

            # Create DataFrame with consistent feature order
            input_df = pd.DataFrame([input_data], columns=selected_features)
            
            # Scale features
            input_scaled = kidney_scaler.transform(input_df)
            print(input_scaled)
            # Make prediction
            prediction = kidney_model.predict(input_scaled)[0]
            print(prediction)
            probability = kidney_model.predict_proba(input_scaled)[0][1] * 100

            # Save to database (convert back to booleans for storage)
            prediction_record = KidneyDiseasePrediction(
                patient=request.user.patient_profile if hasattr(request.user, 'patient_profile') else None,
                doctor=request.user.doctor_profile if hasattr(request.user, 'doctor_profile') else None,
                prediction_result=prediction,
                probability=probability,
                age=int(input_data['Age (yrs)']),
                blood_pressure=int(input_data['Blood Pressure (mm/Hg)']),
                specific_gravity=float(input_data['Specific Gravity']),
                albumin=int(input_data['Albumin']),
                sugar=int(input_data['Sugar']),
                blood_glucose_random=int(input_data['Blood Glucose Random (mgs/dL)']),
                blood_urea=int(input_data['Blood Urea (mgs/dL)']),
                serum_creatinine=float(input_data['Serum Creatinine (mgs/dL)']),
                sodium=float(input_data['Sodium (mEq/L)']),
                potassium=float(input_data['Potassium (mEq/L)']),
                hemoglobin=float(input_data['Hemoglobin (gms)']),
                packed_cell_volume=int(input_data['Packed Cell Volume']),
                white_blood_cells=int(input_data['White Blood Cells (cells/cmm)']),
                red_blood_cells=float(input_data['Red Blood Cells (millions/cmm)']),
                red_blood_cells_normal=bool(input_data['Red Blood Cells: normal']),
                pus_cells_normal=bool(input_data['Pus Cells: normal']),
                pus_cell_clumps_present=bool(input_data['Pus Cell Clumps: present']),
                bacteria_present=bool(input_data['Bacteria: present']),
                hypertension=bool(input_data['Hypertension: yes']),
                diabetes_mellitus=bool(input_data['Diabetes Mellitus: yes']),
                coronary_artery_disease=bool(input_data['Coronary Artery Disease: yes']),
                appetite=not bool(input_data['Appetite: poor']),  # Invert back for storage
                pedal_edema=bool(input_data['Pedal Edema: yes']),
                anemia=bool(input_data['Anemia: yes'])
            )
            prediction_record.save()

            result = {
                'prediction': int(prediction),
                'probability': round(probability, 2)
            }

        except Exception as e:
            logger.error(f"Error during kidney prediction: {str(e)}", exc_info=True)
            messages.error(request, "An error occurred during prediction. Please try again.")
            result = {'error': "Prediction failed. Please check your inputs."}

    elif request.method == "POST":
        logger.warning(f"Form errors: {form.errors.as_json()}")
        messages.warning(request, "Please correct the errors in the form.")

    return render(request, "kidney_predict.html", {
        "form": form,
        "result": result
    })


from django.template.defaulttags import register
@login_required
def predict_diabetes_view(request):
    result = None
    if request.method == "POST":
        form = DiabetesForm(request.POST)
        if form.is_valid():
            try:
                # Collect input data from the form in the correct order
                input_data = [
                    int(form.cleaned_data['HighBP']),
                    int(form.cleaned_data['HighChol']),
                    int(form.cleaned_data['CholCheck']),
                    float(form.cleaned_data['BMI']),
                    int(form.cleaned_data['Smoker']),
                    int(form.cleaned_data['Stroke']),
                    int(form.cleaned_data['HeartDiseaseorAttack']),
                    int(form.cleaned_data['PhysActivity']),
                    int(form.cleaned_data['Fruits']),
                    int(form.cleaned_data['Veggies']),
                    int(form.cleaned_data['HvyAlcoholConsump']),
                    int(form.cleaned_data['AnyHealthcare']),
                    int(form.cleaned_data['NoDocbcCost']),
                    int(form.cleaned_data['GenHlth']),
                    int(form.cleaned_data['MentHlth']),
                    int(form.cleaned_data['PhysHlth']),
                    int(form.cleaned_data['DiffWalk']),
                    int(form.cleaned_data['Sex']),
                    int(form.cleaned_data['Age']),
                    int(form.cleaned_data['Education']),
                    int(form.cleaned_data['Income']),
                ]

                # Convert input data into a DataFrame for prediction
                input_df = pd.DataFrame([input_data], columns=DIABETES_FEATURES)
                input_scaled = diabetes_scaler.transform(input_df)  
                prediction = diabetes_model.predict(input_scaled)[0]
                probability = diabetes_model.predict_proba(input_scaled)[0][1] * 100
                
                # Save result to DiabetesPrediction model
                DiabetesPrediction.objects.create(
                    patient=request.user.patient_profile,
                    doctor=request.user.doctor_profile if hasattr(request.user, 'doctor_profile') else None,
                    HighBP=input_data[0],
                    HighChol=input_data[1],
                    CholCheck=input_data[2],
                    BMI=input_data[3],
                    Smoker=input_data[4],
                    Stroke=input_data[5],
                    HeartDiseaseorAttack=input_data[6],
                    PhysActivity=input_data[7],
                    Fruits=input_data[8],
                    Veggies=input_data[9],
                    HvyAlcoholConsump=input_data[10],
                    AnyHealthcare=input_data[11],
                    NoDocbcCost=input_data[12],
                    GenHlth=input_data[13],
                    MentHlth=input_data[14],
                    PhysHlth=input_data[15],
                    DiffWalk=input_data[16],
                    Sex=input_data[17],
                    Age=input_data[18],
                    Education=input_data[19],
                    Income=input_data[20],
                    prediction_result=prediction,
                    probability=probability
                )

                result = {'prediction': int(prediction), 'probability': round(probability, 2)}

            except Exception as e:
                print(f"Error during diabetes prediction: {e}")
                result = {'error': str(e)}
        else:
            print("Diabetes Form is not valid")
            print(form.errors.as_json())
    else:
        form = DiabetesForm()

    # Define field groups for template organization
    field_groups = {
        'personal_info': ['Sex', 'Age', 'Education', 'Income'],
        'health_indicators': ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk'],
        'medical_history': [
            'HighBP', 'HighChol', 'CholCheck', 'Smoker', 
            'Stroke', 'HeartDiseaseorAttack'
        ],
        'lifestyle': [
            'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',
            'AnyHealthcare', 'NoDocbcCost'
        ]
    }

    return render(request, "diabetes_predict.html", {
        "form": form,
        "result": result,
        "field_groups": field_groups
    })



def signup_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")
        role = request.POST.get("role")  # Get the selected role

        # Check if passwords match
        if password1 != password2:
            messages.error(request, "Passwords do not match!")
            return redirect("predictor:signup")

        # Check if username or email already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists!")
            return redirect("predictor:signup")

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists!")
            return redirect("predictor:signup")

        # Create user
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1,
            first_name=first_name,
            last_name=last_name,
            role=role
        )

        # Auto-login after signup
        login(request, user)

        # Check if profile already exists before redirecting
        if role == 'patient':
            if not hasattr(user, 'patient'):  # If the patient profile doesn't exist
                return redirect("predictor:create_patient_profile")
        elif role == 'doctor':
            if not hasattr(user, 'doctor'):  # If the doctor profile doesn't exist
                return redirect("predictor:create_doctor_profile")

        # Redirect to the dashboard if profile already exists
        return redirect("predictor:dashboard")

    return render(request, "signup.html")


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        # Authenticate the user
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)

            # Check if the user is active
            if not user.is_active:
                messages.error(request, "Your account is inactive. Please contact support.")
                return redirect("predictor:login")

            # Redirect based on user role
            if user.role == 'patient':
                return redirect("predictor:dashboard")
            elif user.role == 'doctor':
                return redirect("predictor:doctor-dashboard")
            elif user.role == 'admin':
                return redirect("predictor:admin_dashboard")
            else:
                messages.error(request, "Invalid user role. Please contact support.")
                return redirect("predictor:login")

        else:
            messages.error(request, "Invalid username or password!")

    return render(request, "login.html")




def create_patient_profile(request):
    # Check if the patient profile already exists for this user
    try:
        patient = request.user.patient_profile  # Access patient profile using related_name
    except Patient.DoesNotExist:
        patient = None

    if patient:  # If the user already has a patient profile
        if request.method == "POST":
            form = PatientProfileForm(request.POST, request.FILES, instance=patient)
            if form.is_valid():
                # Save only the missing data
                updated_patient = form.save(commit=False)
                updated_patient.user = request.user
                updated_patient.save()
                messages.success(request, "Patient profile updated successfully!")
                return redirect("predictor:dashboard")
        else:
            form = PatientProfileForm(instance=patient)
        return render(request, "create_patient_profile.html", {"form": form})
    else:
        if request.method == "POST":
            form = PatientProfileForm(request.POST, request.FILES)
            if form.is_valid():
                # Create a new patient profile
                patient = form.save(commit=False)
                patient.user = request.user
                patient.save()
                messages.success(request, "Patient profile created successfully!")
                return redirect("predictor:dashboard")
        else:
            form = PatientProfileForm()

        return render(request, "create_patient_profile.html", {"form": form})



def create_doctor_profile(request):
    # Check if the doctor profile already exists for this user
    try:
        doctor = request.user.doctor_profile  # Access doctor profile using related_name
    except Doctor.DoesNotExist:
        doctor = None

    if doctor:  # If the user already has a doctor profile
        if request.method == "POST":
            form = DoctorProfileForm(request.POST, request.FILES, instance=doctor)
            if form.is_valid():
                # Save only the missing data
                updated_doctor = form.save(commit=False)
                updated_doctor.user = request.user
                updated_doctor.save()
                messages.success(request, "Doctor profile updated successfully!")
                return redirect("predictor:dashboard")
        else:
            form = DoctorProfileForm(instance=doctor)
        return render(request, "create_doctor_profile.html", {"form": form})
    else:
        if request.method == "POST":
            form = DoctorProfileForm(request.POST, request.FILES)
            if form.is_valid():
                # Create a new doctor profile
                doctor = form.save(commit=False)
                doctor.user = request.user
                doctor.save()
                messages.success(request, "Doctor profile created successfully!")
                return redirect("predictor:dashboard")
        else:
            form = DoctorProfileForm()

        return render(request, "create_doctor_profile.html", {"form": form})



def logout_view(request):
    logout(request)
    return redirect("predictor:login")

@login_required
def dashboard_view(request):
    try:
        # Check if the logged-in user has a patient profile
        if not hasattr(request.user, 'patient_profile'):
            messages.error(request, "Access restricted to patients only.")
            return redirect("predictor:landing-page")

        # Fetch the patient profile
        profile = request.user.patient_profile
        profile_picture = profile.profile_picture.url if profile.profile_picture else None


    except Exception as e:
        messages.error(request, f"An error occurred: {e}")
        return redirect("predictor:landing-page")

    # Fetch predictions for the logged-in patient
    heart_predictions = HeartDiseasePrediction.objects.filter(patient=profile)
    kidney_predictions = KidneyDiseasePrediction.objects.filter(patient=profile)
    diabetes_predictions = DiabetesPrediction.objects.filter(patient=profile)

    # Count total predictions
    total_heart_predictions = heart_predictions.count()
    total_kidney_predictions = kidney_predictions.count()
    total_diabetes_predictions = diabetes_predictions.count()
    total_predictions = total_heart_predictions + total_kidney_predictions + total_diabetes_predictions

    # High-risk predictions
    high_risk_heart = heart_predictions.filter(prediction_result=1).count()
    high_risk_kidney = kidney_predictions.filter(prediction_result=1).count()
    high_risk_diabetes = diabetes_predictions.filter(prediction_result=1).count()

    # Low-risk predictions
    low_risk_heart = total_heart_predictions - high_risk_heart
    low_risk_kidney = total_kidney_predictions - high_risk_kidney
    low_risk_diabetes = total_diabetes_predictions - high_risk_diabetes

    # Heart disease data for charts
    heart_probabilities = list(heart_predictions.values_list('probability', flat=True))
    heart_ages = list(heart_predictions.values_list('age', flat=True))
    chols = list(heart_predictions.values_list('chol', flat=True))
    trestbps = list(heart_predictions.values_list('trestbps', flat=True))
    thalach = list(heart_predictions.values_list('thalach', flat=True))

    # Kidney disease data for charts
    kidney_probabilities = list(kidney_predictions.values_list('probability', flat=True))
    blood_urea = list(kidney_predictions.values_list('blood_urea', flat=True))
    serum_creatinine = list(kidney_predictions.values_list('serum_creatinine', flat=True))
    hemoglobin = list(kidney_predictions.values_list('hemoglobin', flat=True))

    # Diabetes data for charts
    diabetes_probabilities = list(diabetes_predictions.values_list('probability', flat=True))
    hbp = list(diabetes_predictions.values_list('HighBP', flat=True))
    bmi_levels = list(diabetes_predictions.values_list('BMI', flat=True))

    # Latest predictions for personalized advice
    latest_heart = heart_predictions.order_by('-created_at').first()
    latest_kidney = kidney_predictions.order_by('-created_at').first()
    latest_diabetes = diabetes_predictions.order_by('-created_at').first()

    risk_levels = []
    advice_list = []

    if latest_heart:
        prob = latest_heart.probability
        if prob >= 70:
            risk_levels.append("High Risk (Heart)")
            advice_list.append("Consult a cardiologist. Maintain a heart-healthy diet and regular exercise.")
        elif prob >= 40:
            risk_levels.append("Moderate Risk (Heart)")
            advice_list.append("Monitor your heart health, reduce salt and fatty foods, and exercise regularly.")
        else:
            risk_levels.append("Low Risk (Heart)")
            advice_list.append("Maintain a healthy lifestyle. Keep up the good work!")

    if latest_kidney:
        prob = latest_kidney.probability
        if prob >= 70:
            risk_levels.append("High Risk (Kidney)")
            advice_list.append("High risk for kidney issues—limit protein and salt, stay hydrated, and consult a nephrologist.")
        elif prob >= 40:
            risk_levels.append("Moderate Risk (Kidney)")
            advice_list.append("Monitor your kidney function, avoid excessive painkillers, and eat a balanced diet.")
        else:
            risk_levels.append("Low Risk (Kidney)")
            advice_list.append("Your kidneys seem healthy—maintain hydration and avoid harmful substances.")

    if latest_diabetes:
        prob = latest_diabetes.probability
        if prob >= 70:
            risk_levels.append("High Risk (Diabetes)")
            advice_list.append("High risk of diabetes—control sugar intake, monitor blood sugar levels, and consult an endocrinologist.")
        elif prob >= 40:
            risk_levels.append("Moderate Risk (Diabetes)")
            advice_list.append("Watch your diet, reduce carbs and sugar, and stay active.")
        else:
            risk_levels.append("Low Risk (Diabetes)")
            advice_list.append("Keep maintaining a healthy lifestyle to prevent diabetes.")

    # Combine risk levels and advice
    risk_level = ", ".join(risk_levels) if risk_levels else "No risk level detected."
    advice = " ".join(advice_list) if advice_list else "No predictions available yet."

    context = {
        "profile_picture":profile_picture,
        "user": request.user,
        "total_predictions": total_predictions,
        "total_heart_predictions": total_heart_predictions,
        "total_kidney_predictions": total_kidney_predictions,
        "total_diabetes_predictions": total_diabetes_predictions,
        "high_risk_heart": high_risk_heart,
        "high_risk_kidney": high_risk_kidney,
        "high_risk_diabetes": high_risk_diabetes,
        "low_risk_heart": low_risk_heart,
        "low_risk_kidney": low_risk_kidney,
        "low_risk_diabetes": low_risk_diabetes,
        "heart_probabilities": heart_probabilities,
        "heart_ages": heart_ages,
        "chols": chols,
        "trestbps": trestbps,
        "thalach": thalach,
        "kidney_probabilities": kidney_probabilities,
        "blood_urea": blood_urea,
        "serum_creatinine": serum_creatinine,
        "hemoglobin": hemoglobin,
        "diabetes_probabilities": diabetes_probabilities,
        "hbp": hbp,
        "bmi_levels": bmi_levels,
        "risk_level": risk_level,
        "advice": advice,
    }

    return render(request, "patient_dashboard.html", context)



def is_staff_or_superuser(user):
    return user.is_staff or user.is_superuser


@user_passes_test(is_staff_or_superuser, login_url='/admin/login/')
def admin_dashboard(request):
    # Fetch all users
    users = User.objects.all()

    # Fetch all predictions
    heart_predictions = HeartDiseasePrediction.objects.all()
    kidney_predictions = KidneyDiseasePrediction.objects.all()
    diabetes_predictions = DiabetesPrediction.objects.all()

    # Calculate high-risk and low-risk counts
    high_risk_heart = heart_predictions.filter(prediction_result=1).count()
    low_risk_heart = heart_predictions.filter(prediction_result=0).count()
    high_risk_kidney = kidney_predictions.filter(prediction_result=1).count()
    low_risk_kidney = kidney_predictions.filter(prediction_result=0).count()
    high_risk_diabetes = diabetes_predictions.filter(prediction_result=1).count()
    low_risk_diabetes = diabetes_predictions.filter(prediction_result=0).count()

    # Risk by gender
    male_high_risk_heart = heart_predictions.filter(sex=1, prediction_result=1).count()
    male_low_risk_heart = heart_predictions.filter(sex=1, prediction_result=0).count()
    female_high_risk_heart = heart_predictions.filter(sex=0, prediction_result=1).count()
    female_low_risk_heart = heart_predictions.filter(sex=0, prediction_result=0).count()

    # Most common risk factors
    avg_age = heart_predictions.aggregate(avg_age=Avg('age'))['avg_age']
    avg_chol = heart_predictions.aggregate(avg_chol=Avg('chol'))['avg_chol']
    avg_trestbps = heart_predictions.aggregate(avg_trestbps=Avg('trestbps'))['avg_trestbps']
    avg_thalach = heart_predictions.aggregate(avg_thalach=Avg('thalach'))['avg_thalach']
    avg_blood_urea = kidney_predictions.aggregate(avg_blood_urea=Avg('blood_urea'))['avg_blood_urea']
    avg_serum_creatinine = kidney_predictions.aggregate(avg_serum_creatinine=Avg('serum_creatinine'))['avg_serum_creatinine']
    avg_hemoglobin = kidney_predictions.aggregate(avg_hemoglobin=Avg('hemoglobin'))['avg_hemoglobin']
    avg_bmi = diabetes_predictions.aggregate(avg_bmi=Avg('BMI'))['avg_bmi']
    avg_glucose = diabetes_predictions.aggregate(avg_glucose=Avg('HighBP'))['avg_glucose']

    # Count of predictions over time
    prediction_counts_by_date = heart_predictions.annotate(
        date=TruncDate('created_at')
    ).values('date').annotate(
        count=Count('id')
    ).order_by('date')

    # Metric values for charts
    chol_values = [p.chol for p in heart_predictions]
    trestbps_values = [p.trestbps for p in heart_predictions]
    thalach_values = [p.thalach for p in heart_predictions]
    blood_urea_values = [p.blood_urea for p in kidney_predictions]
    serum_creatinine_values = [p.serum_creatinine for p in kidney_predictions]
    hemoglobin_values = [p.hemoglobin for p in kidney_predictions]
    bmi_values = [p.BMI for p in diabetes_predictions]
    glucose_values = [p.HighBP for p in diabetes_predictions]


    # Age distribution
    age_distribution = {
        'under_30': heart_predictions.filter(age__lt=30).count(),
        '30_40': heart_predictions.filter(age__gte=30, age__lt=40).count(),
        '40_50': heart_predictions.filter(age__gte=40, age__lt=50).count(),
        '50_60': heart_predictions.filter(age__gte=50, age__lt=60).count(),
        '60_70': heart_predictions.filter(age__gte=60, age__lt=70).count(),
        'over_70': heart_predictions.filter(age__gte=70).count()
    }

    context = {
        "users": users,
        "high_risk_heart": high_risk_heart,
        "low_risk_heart": low_risk_heart,
        "high_risk_kidney": high_risk_kidney,
        "low_risk_kidney": low_risk_kidney,
        "high_risk_diabetes": high_risk_diabetes,
        "low_risk_diabetes": low_risk_diabetes,
        "male_high_risk_heart": male_high_risk_heart,
        "male_low_risk_heart": male_low_risk_heart,
        "female_high_risk_heart": female_high_risk_heart,
        "female_low_risk_heart": female_low_risk_heart,
        "avg_age": avg_age,
        "avg_chol": avg_chol,
        "avg_trestbps": avg_trestbps,
        "avg_thalach": avg_thalach,
        "avg_blood_urea": avg_blood_urea,
        "avg_serum_creatinine": avg_serum_creatinine,
        "avg_hemoglobin": avg_hemoglobin,
        "avg_bmi": avg_bmi,
        "avg_glucose": avg_glucose,
        "prediction_counts_by_date": list(prediction_counts_by_date),
        "chol_values": chol_values,
        "trestbps_values": trestbps_values,
        "thalach_values": thalach_values,
        "blood_urea_values": blood_urea_values,
        "serum_creatinine_values": serum_creatinine_values,
        "hemoglobin_values": hemoglobin_values,
        "bmi_values": bmi_values,
        "glucose_values": glucose_values,
        "age_distribution": age_distribution,
    }
    return render(request, "adminDashboard.html", context)



@user_passes_test(lambda u: u.is_staff or u.is_superuser, login_url='/admin/login/')
def user_detail(request, user_id):
    # Fetch the user
    User = get_user_model()
    user = get_object_or_404(User, id=user_id)

    # Fetch the user's predictions with optimizations (use select_related or prefetch_related for related models if applicable)
    heart_predictions = HeartDiseasePrediction.objects.filter(patient__user=user)
    kidney_predictions = KidneyDiseasePrediction.objects.filter(patient__user=user)
    diabetes_predictions = DiabetesPrediction.objects.filter(patient__user=user)

    # Prepare data for charts - simplified date format
    heart_dates = [prediction.created_at.strftime("%Y-%m-%d") for prediction in heart_predictions]
    heart_probabilities = [float(prediction.probability) for prediction in heart_predictions]
    heart_ages = [prediction.age for prediction in heart_predictions]
    heart_chols = [prediction.chol for prediction in heart_predictions]

    kidney_dates = [prediction.created_at.strftime("%Y-%m-%d") for prediction in kidney_predictions]
    kidney_probabilities = [float(prediction.probability) for prediction in kidney_predictions]
    kidney_blood_urea = [prediction.blood_urea for prediction in kidney_predictions]
    kidney_serum_creatinine = [prediction.serum_creatinine for prediction in kidney_predictions]

    diabetes_dates = [prediction.created_at.strftime("%Y-%m-%d") for prediction in diabetes_predictions]
    diabetes_probabilities = [float(prediction.probability) for prediction in diabetes_predictions]
    diabetes_hbp = [prediction.HighBP for prediction in diabetes_predictions]
    diabetes_bmi = [prediction.BMI for prediction in diabetes_predictions]

    # Consolidated chart data
    chart_data = {
        "heart": {
            "dates": heart_dates,
            "probabilities": heart_probabilities,
            "ages": heart_ages,
            "chols": heart_chols,
        },
        "kidney": {
            "dates": kidney_dates,
            "probabilities": kidney_probabilities,
            "blood_urea": kidney_blood_urea,
            "serum_creatinine": kidney_serum_creatinine,
        },
        "diabetes": {
            "dates": diabetes_dates,
            "probabilities": diabetes_probabilities,
            "hbp": diabetes_hbp,
            "bmi": diabetes_bmi,
        }
    }

    context = {
        "user": user,
        "heart_predictions": heart_predictions,
        "kidney_predictions": kidney_predictions,
        "diabetes_predictions": diabetes_predictions,
        "chart_data": json.dumps(chart_data),  # Ensure the data is serialized correctly
    }

    return render(request, "user_detail.html", context)




from django.contrib.auth.forms import UserChangeForm

@user_passes_test(is_staff_or_superuser, login_url='/admin/login/')
def edit_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if request.method == 'POST':
        form = UserChangeForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            return redirect('predictor:admin-dashboard')
    else:
        form = UserChangeForm(instance=user)
    return render(request, 'edit_user.html', {'form': form, 'user': user})


@user_passes_test(is_staff_or_superuser, login_url='/admin/login/')
def delete_user(request, user_id):
    user = get_object_or_404(User, id=user_id)
    if request.method == 'POST':
        user.delete()
        return redirect('predictor:admin-dashboard')
    return render(request, 'confirm_delete.html', {'user': user})


from .forms import CustomUserForm

@user_passes_test(is_staff_or_superuser, login_url='/admin/login/')
def create_user(request):
    if request.method == 'POST':
        form = CustomUserForm(request.POST)  # Use your custom form
        if form.is_valid():
            form.save()
            return redirect('predictor:admin-dashboard')
    else:
        form = CustomUserForm()  # Use your custom form
    return render(request, 'create_user.html', {'form': form})




@login_required
def doctor_dashboard(request):
    try:
        # Check if the logged-in user has a doctor profile
        if not hasattr(request.user, 'doctor_profile'):
            messages.error(request, "Access restricted to doctors only.")
            return redirect("predictor:login")

        doctor = request.user.doctor_profile

        # Fetch all patients assigned to this doctor with prefetch
        patients = Patient.objects.filter(doctor=doctor).prefetch_related(
            'heartdiseaseprediction_set',
            'kidneydiseaseprediction_set',
            'diabetesprediction_set'
        )

        patient_data = []
        high_risk_count = 0
        recent_count = 0

        for patient in patients:
            # Get predictions
            heart_predictions = patient.heartdiseaseprediction_set.all()
            kidney_predictions = patient.kidneydiseaseprediction_set.all()
            diabetes_predictions = patient.diabetesprediction_set.all()

            # Check if any predictions exist
            has_heart = heart_predictions.exists()
            has_kidney = kidney_predictions.exists()
            has_diabetes = diabetes_predictions.exists()

            # Calculate high risk
            if (has_heart and heart_predictions.last().prediction_result == 1) or \
               (has_kidney and kidney_predictions.last().prediction_result == 1) or \
               (has_diabetes and diabetes_predictions.last().prediction_result == 1):
                high_risk_count += 1

            # Calculate recent predictions (within 24 hours)
            now = timezone.now()
            if (has_heart and (now - heart_predictions.last().created_at) <= timedelta(hours=24)) or \
               (has_kidney and (now - kidney_predictions.last().created_at) <= timedelta(hours=24)) or \
               (has_diabetes and (now - diabetes_predictions.last().created_at) <= timedelta(hours=24)):
                recent_count += 1

            # Prepare chart data
            def prepare_chart_data(predictions, fields):
                data = {field: [] for field in fields}
                dates = []
                for prediction in predictions:
                    dates.append(prediction.created_at.isoformat())
                    for field in fields:
                        data[field].append(getattr(prediction, field))
                return {'dates': dates, **data}

            heart_data = prepare_chart_data(heart_predictions, ['probability', 'age', 'chol']) if has_heart else {'dates': [], 'probability': [], 'age': [], 'chol': []}
            kidney_data = prepare_chart_data(kidney_predictions, ['probability', 'blood_urea', 'serum_creatinine']) if has_kidney else {'dates': [], 'probability': [], 'blood_urea': [], 'serum_creatinine': []}
            diabetes_data = prepare_chart_data(diabetes_predictions, ['probability', 'HighBP', 'BMI']) if has_diabetes else {'dates': [], 'probability': [], 'HighBP': [], 'BMI': []}

            # AI suggestions
            ai_suggestions = []
            if has_heart:
                latest = heart_predictions.last()
                ai_suggestions.append(
                    "High risk of heart disease. Recommend lifestyle changes and regular checkups." 
                    if latest.prediction_result == 1 else
                    "Low risk of heart disease. Maintain a healthy lifestyle."
                )
            
            if has_kidney:
                latest = kidney_predictions.last()
                ai_suggestions.append(
                    "High risk of kidney disease. Monitor kidney function and avoid nephrotoxic drugs."
                    if latest.prediction_result == 1 else
                    "Low risk of kidney disease. Stay hydrated and avoid excessive protein intake."
                )
            
            if has_diabetes:
                latest = diabetes_predictions.last()
                ai_suggestions.append(
                    "High risk of diabetes. Monitor blood sugar levels and follow a diabetic diet."
                    if latest.prediction_result == 1 else
                    "Low risk of diabetes. Maintain a balanced diet and exercise regularly."
                )

            patient_data.append({
                "patient": patient,
                "heart_predictions": heart_predictions,
                "kidney_predictions": kidney_predictions,
                "diabetes_predictions": diabetes_predictions,
                "heart_dates": json.dumps(heart_data['dates']),
                "heart_probabilities": json.dumps(heart_data['probability']),
                "heart_ages": json.dumps(heart_data['age']),
                "heart_chols": json.dumps(heart_data['chol']),
                "kidney_dates": json.dumps(kidney_data['dates']),
                "kidney_probabilities": json.dumps(kidney_data['probability']),
                "kidney_blood_urea": json.dumps(kidney_data['blood_urea']),
                "kidney_serum_creatinine": json.dumps(kidney_data['serum_creatinine']),
                "diabetes_dates": json.dumps(diabetes_data['dates']),
                "diabetes_probabilities": json.dumps(diabetes_data['probability']),
                "diabetes_hbp": json.dumps(diabetes_data['HighBP']),
                "diabetes_bmi": json.dumps(diabetes_data['BMI']),
                "ai_suggestions": ai_suggestions,
            })

        context = {
            "doctor": doctor,
            "patient_data": patient_data,
            "high_risk_count": high_risk_count,
            "recent_count": recent_count,
            "total_patients": len(patient_data),
        }
        return render(request, "doctor_dashboard.html", context)

    except Exception as e:
        messages.error(request, f"An error occurred: {str(e)}")
        return redirect("predictor:login")



def index(request):
    return render(request, 'index.html')