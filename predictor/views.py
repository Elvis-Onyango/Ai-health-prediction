import os
import joblib
import pandas as pd
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.shortcuts import render, redirect
from .forms import HeartDiseaseForm
from .models import Prediction
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
from django.contrib import messages
from django.contrib.auth.decorators import user_passes_test
from django.db.models import Avg
from django.contrib.auth import authenticate, login

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Get the absolute path to the model
model_path = os.path.join(settings.BASE_DIR, "predictor", "heart_disease_model.pkl")
scaler_path = os.path.join(settings.BASE_DIR, "predictor", "scaler.pkl")

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Feature names in correct order
FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

@login_required  # Ensure user is logged in before making predictions
def predict_view(request):
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

                # Convert to DataFrame with feature names
                input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

                # Scale input data
                input_scaled = scaler.transform(input_df)

                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1] * 100  

                # Save result in DB
                if request.user.is_authenticated:
                    Prediction.objects.create(
                        user=request.user,
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

                result = {
                    'prediction': int(prediction),
                    'probability': round(probability, 2)
                }

                print("\n Prediction Result ")
                print(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
                print(f"Confidence: {result['probability']}%\n")

            except Exception as e:
                result = {'error': str(e)}
                print(f"\n Error: {str(e)}\n")

    else:
        form = HeartDiseaseForm()

    return render(request, "predict.html", {"form": form, "result": result})





def signup_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        first_name = request.POST["first_name"]
        last_name = request.POST["last_name"]
        email = request.POST["email"]
        password1 = request.POST["password1"]
        password2 = request.POST["password2"]

        # Check if passwords match
        if password1 != password2:
            messages.error(request, "Passwords do not match!")
            return redirect("signup")

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists!")
            return redirect("signup")

        # Create user and explicitly save first and last name
        user = User.objects.create_user(username=username, email=email, password=password1)
        user.first_name = first_name  #Add first name
        user.last_name = last_name    # Add last name
        user.save()  # Save user again to update first_name & last_name

        login(request, user)  # Auto-login after signup
        return redirect("dashboard")

    return render(request, "signup.html")




def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid username or password!")

    return render(request, "login.html")



from django.contrib.auth import logout

def logout_view(request):
    logout(request)
    return redirect("login")





@login_required
def dashboard_view(request):
    user_predictions = Prediction.objects.filter(user=request.user)
    total_predictions = user_predictions.count()
    high_risk_count = user_predictions.filter(prediction_result=1).count()
    low_risk_count = total_predictions - high_risk_count

    # Data for charts
    probabilities = [p.probability for p in user_predictions]
    ages = [p.age for p in user_predictions]
    chols = [p.chol for p in user_predictions]
    trestbps = [p.trestbps for p in user_predictions]
    thalach = [p.thalach for p in user_predictions]

    # Latest Prediction for Personalized Advice
    latest_prediction = user_predictions.order_by('-created_at').first()
    risk_level = None
    advice = "No prediction available yet."

    if latest_prediction:
        prob = latest_prediction.probability

        if prob < 40:
            risk_level = "Low Risk"
            advice = "Great job! Maintain a balanced diet, exercise regularly, and avoid smoking."
        elif prob < 70:
            risk_level = "Moderate Risk"
            advice = "You're at moderate risk. Reduce sugar and fat intake, increase physical activity, and monitor your blood pressure."
        else:
            risk_level = "High Risk"
            advice = "High risk detected! Consult a doctor immediately, follow a heart-healthy diet, and consider medication if necessary."

        # Additional suggestions based on user health data
        if latest_prediction.chol > 240:
            advice += " Your cholesterol is high—reduce fatty foods and eat more fiber."
        if latest_prediction.trestbps > 130:
            advice += " Your blood pressure is elevated—reduce salt intake and manage stress."
        if not latest_prediction.exang:  # If no exercise-induced angina
            advice += " Consider increasing your daily physical activity."

    context = {
        "user": request.user,
        "total_predictions": total_predictions,
        "high_risk_count": high_risk_count,
        "low_risk_count": low_risk_count,
        "probabilities": probabilities,
        "ages": ages,
        "chols": chols,
        "trestbps": trestbps,
        "thalach": thalach,
        "risk_level": risk_level,
        "advice": advice,
    }
    return render(request, "dashboard.html", context)





def is_staff_or_superuser(user):
    return user.is_staff or user.is_superuser

@user_passes_test(is_staff_or_superuser, login_url='/admin/login/')
def admin_dashboard(request):
    # Fetch all users
    users = User.objects.all()

    # Fetch all predictions
    predictions = Prediction.objects.all()

    # Calculate high-risk and low-risk counts
    high_risk_count = predictions.filter(prediction_result=1).count()
    low_risk_count = predictions.filter(prediction_result=0).count()

    # Risk by gender
    male_high_risk = predictions.filter(sex=1, prediction_result=1).count()
    male_low_risk = predictions.filter(sex=1, prediction_result=0).count()
    female_high_risk = predictions.filter(sex=0, prediction_result=1).count()
    female_low_risk = predictions.filter(sex=0, prediction_result=0).count()

    # Most common risk factors
    avg_age = predictions.aggregate(avg_age=Avg('age'))['avg_age']
    avg_chol = predictions.aggregate(avg_chol=Avg('chol'))['avg_chol']
    avg_trestbps = predictions.aggregate(avg_trestbps=Avg('trestbps'))['avg_trestbps']

    context = {
        "users": users,
        "predictions": predictions,
        "high_risk_count": high_risk_count,
        "low_risk_count": low_risk_count,
        "male_high_risk": male_high_risk,
        "male_low_risk": male_low_risk,
        "female_high_risk": female_high_risk,
        "female_low_risk": female_low_risk,
        "avg_age": avg_age,
        "avg_chol": avg_chol,
        "avg_trestbps": avg_trestbps,
    }
    return render(request, "adminDashboard.html", context)


def index(request):
    return render(request,'index.html')