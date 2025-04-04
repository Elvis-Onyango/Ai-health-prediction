from django import forms
from django.core.validators import MinValueValidator
from django.contrib.auth.forms import UserCreationForm
from .models import (
    CustomUser, RoleChoices, Doctor, Patient,
    KidneyDiseasePrediction, HeartDiseasePrediction, DiabetesPrediction
)

# Custom User Form
class CustomUserForm(UserCreationForm):
    role = forms.ChoiceField(choices=RoleChoices.choices, initial=RoleChoices.PATIENT)

    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2', 'role']

# Doctor Profile Form
class DoctorProfileForm(forms.ModelForm):
    class Meta:
        model = Doctor
        fields = ['profile_picture', 'specialization', 'experience_years', 'hospital', 'phone', 'email']
        widgets = {
            'profile_picture': forms.FileInput(attrs={'class': 'form-control'}),
            'specialization': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter specialization'}),
            'experience_years': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter years of experience'}),
            'hospital': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter hospital name'}),
            'phone': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter phone number'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter email'}),
        }

# Patient Profile Form
class PatientProfileForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ['profile_picture', 'phone', 'email', 'address', 'medical_history']
        widgets = {
            'profile_picture': forms.FileInput(attrs={'class': 'form-control'}),
            'phone': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter phone number'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter email'}),
            'address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Enter address'}),
            'medical_history': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Enter medical history'}),
        }




from django import forms
from .models import KidneyDiseasePrediction

class KidneyDiseaseForm(forms.ModelForm):
    # Numerical Features
    age = forms.IntegerField(
        min_value=0,
        max_value=120,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Age (yrs)',
        required=True
    )
    
    blood_pressure = forms.IntegerField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Blood Pressure (mm/Hg)',
        required=True
    )
    
    specific_gravity = forms.FloatField(
        min_value=1.000,
        max_value=1.050,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.001'}),
        label='Specific Gravity',
        required=True
    )
    
    albumin = forms.IntegerField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Albumin',
        required=True
    )
    
    sugar = forms.IntegerField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Sugar',
        required=True
    )
    
    blood_glucose_random = forms.IntegerField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Blood Glucose Random (mgs/dL)',
        required=True
    )
    
    blood_urea = forms.IntegerField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Blood Urea (mgs/dL)',
        required=True
    )
    
    serum_creatinine = forms.FloatField(
        min_value=0.0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Serum Creatinine (mgs/dL)',
        required=True
    )
    
    sodium = forms.FloatField(
        min_value=0.0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Sodium (mEq/L)',
        required=True
    )
    
    potassium = forms.FloatField(
        min_value=0.0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Potassium (mEq/L)',
        required=True
    )
    
    hemoglobin = forms.FloatField(
        min_value=0.0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Hemoglobin (gms)',
        required=True
    )
    
    packed_cell_volume = forms.IntegerField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Packed Cell Volume',
        required=True
    )
    
    white_blood_cells = forms.IntegerField(
        min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='White Blood Cells (cells/cmm)',
        required=True
    )
    
    red_blood_cells = forms.FloatField(
        min_value=0.0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Red Blood Cells (millions/cmm)',
        required=True
    )
    
    # Binary Features (using RadioSelect for better UX)
    red_blood_cells_normal = forms.ChoiceField(
        choices=[(True, 'Normal'), (False, 'Abnormal')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Red Blood Cells: normal',
        initial=True,
        required=True
    )
    
    pus_cells_normal = forms.ChoiceField(
        choices=[(True, 'Normal'), (False, 'Abnormal')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Pus Cells: normal',
        initial=True,
        required=True
    )
    
    pus_cell_clumps_present = forms.ChoiceField(
        choices=[(True, 'Present'), (False, 'Not Present')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Pus Cell Clumps: present',
        initial=False,
        required=True
    )
    
    bacteria_present = forms.ChoiceField(
        choices=[(True, 'Present'), (False, 'Not Present')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Bacteria: present',
        initial=False,
        required=True
    )
    
    hypertension = forms.ChoiceField(
        choices=[(True, 'Yes'), (False, 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Hypertension: yes',
        initial=False,
        required=True
    )
    
    diabetes_mellitus = forms.ChoiceField(
        choices=[(True, 'Yes'), (False, 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Diabetes Mellitus: yes',
        initial=False,
        required=True
    )
    
    coronary_artery_disease = forms.ChoiceField(
        choices=[(True, 'Yes'), (False, 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Coronary Artery Disease: yes',
        initial=False,
        required=True
    )
    
    appetite = forms.ChoiceField(
        choices=[(True, 'Poor'), (False, 'Good')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Appetite: poor',
        initial=False,
        required=True
    )
    
    pedal_edema = forms.ChoiceField(
        choices=[(True, 'Yes'), (False, 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Pedal Edema: yes',
        initial=False,
        required=True
    )
    
    anemia = forms.ChoiceField(
        choices=[(True, 'Yes'), (False, 'No')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        label='Anemia: yes',
        initial=False,
        required=True
    )

    class Meta:
        model = KidneyDiseasePrediction
        exclude = ['patient', 'doctor', 'created_at', 'prediction_result', 'probability']
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set all fields as required (already handled in field definitions)
        for field_name, field in self.fields.items():
            field.widget.attrs['required'] = 'required'
            if 'class' in field.widget.attrs:
                field.widget.attrs['class'] += ' required-field'
            else:
                field.widget.attrs['class'] = 'required-field'


# Heart Disease Form
class HeartDiseaseForm(forms.ModelForm):
    # Dropdown choices
    SEX_CHOICES = [
        (0, 'Female'),
        (1, 'Male'),
    ]
    CP_CHOICES = [
        (0, 'Typical Angina'),
        (1, 'Atypical Angina'),
        (2, 'Non-Anginal Pain'),
        (3, 'Asymptomatic'),
    ]
    RESTECG_CHOICES = [
        (0, 'Normal'),
        (1, 'ST-T Wave Abnormality'),
        (2, 'Left Ventricular Hypertrophy'),
    ]
    SLOPE_CHOICES = [
        (0, 'Upsloping'),
        (1, 'Flat'),
        (2, 'Downsloping'),
    ]
    THAL_CHOICES = [
        (1, 'Normal'),
        (2, 'Fixed Defect'),
        (3, 'Reversible Defect'),
    ]

    # Fields with dropdowns
    sex = forms.ChoiceField(choices=SEX_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}), label='Gender')
    cp = forms.ChoiceField(choices=CP_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}), label='Chest Pain Type')
    restecg = forms.ChoiceField(choices=RESTECG_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}), label='Resting ECG Results')
    slope = forms.ChoiceField(choices=SLOPE_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}), label='Slope of ST Segment')
    thal = forms.ChoiceField(choices=THAL_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}), label='Thalassemia')

    class Meta:
        model = HeartDiseasePrediction
        exclude = ['patient', 'doctor', 'created_at', 'prediction_result', 'probability']
        widgets = {
            'age': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter age', 'min': 0}),
            'trestbps': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter resting blood pressure', 'min': 0}),
            'chol': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter cholesterol level', 'min': 0}),
            'thalach': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter maximum heart rate', 'min': 0}),
            'oldpeak': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter ST depression', 'step': '0.1', 'min': 0}),
            'ca': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter number of major vessels (0-3)', 'min': 0, 'max': 3}),
            'fbs': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'exang': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
        labels = {
            'age': 'Age',
            'trestbps': 'Resting Blood Pressure',
            'chol': 'Serum Cholesterol',
            'thalach': 'Maximum Heart Rate Achieved',
            'oldpeak': 'ST Depression',
            'ca': 'Major Vessels Colored by Fluoroscopy',
            'fbs': 'Fasting Blood Sugar > 120 mg/dl',
            'exang': 'Exercise-Induced Angina',
        }

    def clean_age(self):
        age = self.cleaned_data.get('age')
        if age < 0:
            raise forms.ValidationError("Age cannot be negative.")
        return age




from django import forms
from .models import DiabetesPrediction

class DiabetesForm(forms.ModelForm):
    # Boolean fields with radio select
    YES_NO_CHOICES = [(1, 'Yes'), (0, 'No')]
    
    HighBP = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='High Blood Pressure',
        initial=0  # Matches model default
    )
    
    HighChol = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='High Cholesterol',
        initial=0
    )
    
    CholCheck = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Cholesterol Check in Last 5 Years',
        initial=1
    )
    
    Smoker = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Have you smoked at least 100 cigarettes in your life?',
        initial=0
    )
    
    Stroke = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Have you ever had a stroke?',
        initial=0
    )
    
    HeartDiseaseorAttack = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Have you had heart disease or a heart attack?',
        initial=0
    )
    
    PhysActivity = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Physical activity in past 30 days?',
        initial=1
    )
    
    Fruits = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Do you consume fruit daily?',
        initial=1
    )
    
    Veggies = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Do you consume vegetables daily?',
        initial=1
    )
    
    HvyAlcoholConsump = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Heavy alcohol consumption?',
        initial=0
    )
    
    AnyHealthcare = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Have any health care coverage?',
        initial=1
    )
    
    NoDocbcCost = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?',
        initial=0
    )
    
    # Numeric fields
    BMI = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'min': 10,
            'max': 50,
            'step': '0.1'
        }),
        label='Body Mass Index (BMI)',
        help_text='Normal range: 18.5-24.9',
        initial=25.0
    )
    
    # Health rating fields
    HEALTH_RATING = [(i, str(i)) for i in range(1, 6)]
    GenHlth = forms.ChoiceField(
        choices=HEALTH_RATING,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='General Health Rating (1-5)',
        help_text='1 = Excellent, 5 = Poor',
        initial=3
    )
    
    MentHlth = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'min': 0,
            'max': 30
        }),
        label='Days of poor mental health in past 30 days',
        initial=0
    )
    
    PhysHlth = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'min': 0,
            'max': 30
        }),
        label='Days of poor physical health in past 30 days',
        initial=0
    )
    
    DiffWalk = forms.ChoiceField(
        choices=YES_NO_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Do you have serious difficulty walking or climbing stairs?',
        initial=0
    )
    
    # Demographic fields
    SEX_CHOICES = [(0, 'Female'), (1, 'Male')]  # Corrected to match model
    Sex = forms.ChoiceField(
        choices=SEX_CHOICES,
        widget=forms.RadioSelect(attrs={'class': 'form-radio'}),
        label='Gender',
        initial=0
    )
    
    Age = forms.IntegerField(
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'min': 18,
            'max': 120
        }),
        label='Age (years)',
        initial=30,
        validators=[MinValueValidator(18)],
        help_text='Please enter your exact age'
    )
    
    EDUCATION_LEVELS = [(i, str(i)) for i in range(1, 7)]
    Education = forms.ChoiceField(
        choices=EDUCATION_LEVELS,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Education Level (1-6)',
        initial=4
    )
    
    INCOME_LEVELS = [(i, str(i)) for i in range(1, 9)]
    Income = forms.ChoiceField(
        choices=INCOME_LEVELS,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Income Level (1-8)',
        initial=3
    )

    class Meta:
        model = DiabetesPrediction
        exclude = ['patient', 'doctor', 'created_at', 'prediction_result', 'probability']
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set all fields as required by default
        for field in self.fields:
            self.fields[field].required = True
            if 'class' in self.fields[field].widget.attrs:
                self.fields[field].widget.attrs['class'] += ' w-full px-3 py-2 border rounded-md'
            else:
                self.fields[field].widget.attrs['class'] = 'w-full px-3 py-2 border rounded-md'
