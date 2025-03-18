from django import forms

class HeartDiseaseForm(forms.Form):
    age = forms.IntegerField(
        label="Age",
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500', 'placeholder': 'Enter your age'})
    )
    sex = forms.ChoiceField(
        label="Sex",
        choices=[(1, "Male"), (0, "Female")],
        widget=forms.Select(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500'})
    )
    cp = forms.ChoiceField(
        label="Chest Pain Type",
        choices=[(0, "Typical Angina"), (1, "Atypical Angina"), (2, "Non-Anginal Pain"), (3, "Asymptomatic")],
        widget=forms.Select(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500'})
    )
    trestbps = forms.IntegerField(
        label="Resting Blood Pressure",
        widget=forms.NumberInput(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500', 'placeholder': 'Enter resting blood pressure'})
    )
    chol = forms.IntegerField(
        label="Cholesterol Level",
        widget=forms.NumberInput(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500', 'placeholder': 'Enter cholesterol level'})
    )
    fbs = forms.ChoiceField(
        label="Fasting Blood Sugar > 120 mg/dl",
        choices=[(1, "True"), (0, "False")],
        widget=forms.Select(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500'})
    )
    restecg = forms.ChoiceField(
        label="Resting ECG Results",
        choices=[(0, "Normal"), (1, "ST-T Wave Abnormality"), (2, "Left Ventricular Hypertrophy")],
        widget=forms.Select(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500'})
    )
    thalach = forms.IntegerField(
        label="Maximum Heart Rate Achieved",
        widget=forms.NumberInput(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500', 'placeholder': 'Enter maximum heart rate'})
    )
    exang = forms.ChoiceField(
        label="Exercise-Induced Angina",
        choices=[(1, "Yes"), (0, "No")],
        widget=forms.Select(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500'})
    )
    oldpeak = forms.FloatField(
        label="ST Depression Induced by Exercise",
        widget=forms.NumberInput(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500', 'placeholder': 'Enter ST depression value'})
    )
    slope = forms.ChoiceField(
        label="Slope of Peak Exercise ST Segment",
        choices=[(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")],
        widget=forms.Select(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500'})
    )
    ca = forms.IntegerField(
        label="Number of Major Vessels (0-3)",
        widget=forms.NumberInput(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500', 'placeholder': 'Enter number of major vessels'})
    )
    thal = forms.ChoiceField(
        label="Thalassemia",
        choices=[(0, "Normal"), (1, "Fixed Defect"), (2, "Reversible Defect")],
        widget=forms.Select(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500'})
    )