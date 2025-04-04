from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.core.validators import MinValueValidator, MaxValueValidator

class RoleChoices(models.TextChoices):
    DOCTOR = 'doctor', 'Doctor'
    PATIENT = 'patient', 'Patient'

class CustomUser(AbstractUser):
    role = models.CharField(
        max_length=10,
        choices=RoleChoices.choices,
        default=RoleChoices.PATIENT
    )

    def __str__(self):
        return self.username


class Doctor(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='doctor_profile')
    profile_picture = models.ImageField(upload_to='doctor_profiles/', blank=True, null=True)
    specialization = models.CharField(max_length=100, default='General Practitioner')
    experience_years = models.PositiveIntegerField(default=1)
    hospital = models.CharField(max_length=100, default='Unknown Hospital')
    phone = models.CharField(max_length=15, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)

    def __str__(self):
        return f"Dr. {self.user.first_name} {self.user.last_name}"


class Patient(models.Model):
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='patient_profile')
    profile_picture = models.ImageField(upload_to='patient_profiles/', blank=True, null=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)
    address = models.CharField(max_length=255, blank=True, null=True)
    medical_history = models.TextField(blank=True, null=True)
    doctor = models.ForeignKey(Doctor, on_delete=models.SET_NULL, null=True, blank=True, related_name='patients') 

    def __str__(self):
        return f"{self.user.first_name} {self.user.last_name} - Patient"


@receiver(post_save, sender=CustomUser)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        try:
            if instance.role == RoleChoices.DOCTOR:
                Doctor.objects.create(user=instance)
            elif instance.role == RoleChoices.PATIENT:
                Patient.objects.create(user=instance)
        except Exception as e:
            print(f"Error creating profile: {e}")




class HealthPredictionBase(models.Model):
    patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
    doctor = models.ForeignKey('Doctor', on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    prediction_result = models.IntegerField(choices=[(0, 'Negative'), (1, 'Positive')], default=0)
    probability = models.FloatField(
        default=50.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)]
    )  # Confidence Score

    class Meta:
        abstract = True

    def __str__(self):
        return f"{self.patient.user.username} - {self.__class__.__name__} Prediction ({self.probability}%)"


class KidneyDiseasePrediction(HealthPredictionBase):
    # Numerical features
    age = models.IntegerField(
        verbose_name="Age (yrs)",
        validators=[MinValueValidator(0)]
    )
    blood_pressure = models.IntegerField(
        verbose_name="Blood Pressure (mm/Hg)",
        validators=[MinValueValidator(0)]
    )
    specific_gravity = models.FloatField(
        validators=[MinValueValidator(0.0)]
    )
    albumin = models.IntegerField(
        validators=[MinValueValidator(0)]
    )
    sugar = models.IntegerField(
        validators=[MinValueValidator(0)]
    )
    blood_glucose_random = models.IntegerField(
        verbose_name="Blood Glucose Random (mgs/dL)",
        validators=[MinValueValidator(0)]
    )
    blood_urea = models.IntegerField(
        verbose_name="Blood Urea (mgs/dL)",
        validators=[MinValueValidator(0)]
    )
    serum_creatinine = models.FloatField(
        verbose_name="Serum Creatinine (mgs/dL)",
        validators=[MinValueValidator(0.0)]
    )
    sodium = models.FloatField(
        verbose_name="Sodium (mEq/L)",
        validators=[MinValueValidator(0.0)]
    )
    potassium = models.FloatField(
        verbose_name="Potassium (mEq/L)",
        validators=[MinValueValidator(0.0)]
    )
    hemoglobin = models.FloatField(
        verbose_name="Hemoglobin (gms)",
        validators=[MinValueValidator(0.0)]
    )
    packed_cell_volume = models.IntegerField(
        validators=[MinValueValidator(0)]
    )
    white_blood_cells = models.IntegerField(
        verbose_name="White Blood Cells (cells/cmm)",
        validators=[MinValueValidator(0)]
    )
    red_blood_cells = models.FloatField(
        verbose_name="Red Blood Cells (millions/cmm)",
        validators=[MinValueValidator(0.0)]
    )
    
    # Binary features
    red_blood_cells_normal = models.BooleanField(
        verbose_name="Red Blood Cells: normal",
        default=True
    )
    pus_cells_normal = models.BooleanField(
        verbose_name="Pus Cells: normal",
        default=True
    )
    pus_cell_clumps_present = models.BooleanField(
        verbose_name="Pus Cell Clumps: present",
        default=False
    )
    bacteria_present = models.BooleanField(
        verbose_name="Bacteria: present",
        default=False
    )
    hypertension = models.BooleanField(
        verbose_name="Hypertension: yes",
        default=False
    )
    diabetes_mellitus = models.BooleanField(
        verbose_name="Diabetes Mellitus: yes",
        default=False
    )
    coronary_artery_disease = models.BooleanField(
        verbose_name="Coronary Artery Disease: yes",
        default=False
    )
    appetite = models.BooleanField(
        verbose_name="Appetite: poor",
        default=False  # False means good appetite
    )
    pedal_edema = models.BooleanField(
        verbose_name="Pedal Edema: yes",
        default=False
    )
    anemia = models.BooleanField(
        verbose_name="Anemia: yes",
        default=False
    )

    class Meta:
        verbose_name = "Kidney Disease Prediction"
        verbose_name_plural = "Kidney Disease Predictions"



class HeartDiseasePrediction(HealthPredictionBase):
    class ChestPainType(models.TextChoices):
        TYPICAL_ANGINA = '0', 'Typical Angina'
        ATYPICAL_ANGINA = '1', 'Atypical Angina'
        NON_ANGINAL_PAIN = '2', 'Non-Anginal Pain'
        ASYMPTOMATIC = '3', 'Asymptomatic'

    age = models.IntegerField(default=30, validators=[MinValueValidator(0)])
    sex = models.IntegerField(choices=[(0, 'Female'), (1, 'Male')], default=1)
    cp = models.CharField(max_length=1, choices=ChestPainType.choices, default=ChestPainType.TYPICAL_ANGINA)
    trestbps = models.IntegerField(default=120, validators=[MinValueValidator(0)])  # Resting Blood Pressure
    chol = models.IntegerField(default=200, validators=[MinValueValidator(0)])  # Cholesterol
    fbs = models.BooleanField(default=False)  # Fasting Blood Sugar
    restecg = models.IntegerField(default=0)  # Resting ECG
    thalach = models.IntegerField(default=150, validators=[MinValueValidator(0)])  # Maximum Heart Rate Achieved
    exang = models.BooleanField(default=False)  # Exercise-Induced Angina
    oldpeak = models.FloatField(default=0.0, validators=[MinValueValidator(0.0)])  # ST Depression
    slope = models.IntegerField(default=1)  # Slope of Peak Exercise ST Segment
    ca = models.IntegerField(default=0)  # Major Vessels
    thal = models.IntegerField(default=2)  # Thalassemia Indicator


class DiabetesPrediction(HealthPredictionBase):
    HighBP = models.IntegerField(default=0)
    HighChol = models.IntegerField(default=0)
    CholCheck = models.IntegerField(default=1)
    BMI = models.FloatField(default=25.0, validators=[MinValueValidator(0.0)])
    Smoker = models.IntegerField(default=0)
    Stroke = models.IntegerField(default=0)
    HeartDiseaseorAttack = models.IntegerField(default=0)
    PhysActivity = models.IntegerField(default=1)
    Fruits = models.IntegerField(default=1)
    Veggies = models.IntegerField(default=1)
    HvyAlcoholConsump = models.IntegerField(default=0)
    AnyHealthcare = models.IntegerField(default=1)
    NoDocbcCost = models.IntegerField(default=0)
    GenHlth = models.IntegerField(default=3)
    MentHlth = models.IntegerField(default=0)
    PhysHlth = models.IntegerField(default=0)
    DiffWalk = models.IntegerField(default=0)
    Sex = models.IntegerField(default=0)  # Female = 0, Male = 1
    Age = models.IntegerField(default=30, validators=[MinValueValidator(0)])
    Education = models.IntegerField(default=4)
    Income = models.IntegerField(default=3)

