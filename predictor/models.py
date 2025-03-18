from django.contrib.auth.models import User  
from django.db import models  

class Prediction(models.Model):  
    user = models.ForeignKey(User, on_delete=models.CASCADE)  

    # Features with default values
    age = models.IntegerField(default=30)  
    sex = models.IntegerField(choices=[(0, 'Female'), (1, 'Male')], default=1)  
    cp = models.IntegerField(default=0)  # Chest Pain Type  
    trestbps = models.IntegerField(default=120)  # Resting Blood Pressure  
    chol = models.IntegerField(default=200)  # Serum Cholesterol  
    fbs = models.BooleanField(default=False)  # Fasting Blood Sugar  
    restecg = models.IntegerField(default=0)  # Resting ECG Results  
    thalach = models.IntegerField(default=150)  # Maximum Heart Rate  
    exang = models.BooleanField(default=False)  # Exercise Induced Angina  
    oldpeak = models.FloatField(default=0.0)  # ST Depression  
    slope = models.IntegerField(default=1)  # Slope of ST Segment  
    ca = models.IntegerField(default=0)  # Number of Major Vessels  
    thal = models.IntegerField(default=2)  # Thalassemia  

    # Prediction result
    prediction_result = models.IntegerField(choices=[(0, 'No Heart Disease'), (1, 'Heart Disease')], default=0)  
    probability = models.FloatField(default=50.0)  # Confidence Score  
    created_at = models.DateTimeField(auto_now_add=True)  

    def __str__(self):  
        return f"{self.user.username} - {self.get_prediction_result_display()} ({self.probability}%)"
