from django.contrib import admin
from .models import CustomUser, Doctor, Patient, KidneyDiseasePrediction, HeartDiseasePrediction, DiabetesPrediction

# Register the CustomUser model to manage users in the admin interface
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('username', 'first_name', 'last_name', 'email', 'role')
    search_fields = ('username', 'first_name', 'last_name', 'email')
    list_filter = ('role',)

admin.site.register(CustomUser, CustomUserAdmin)


# Register Doctor model
class DoctorAdmin(admin.ModelAdmin):
    list_display = ('user', 'specialization', 'experience_years', 'hospital', 'phone', 'email')
    search_fields = ('user__first_name', 'user__last_name', 'specialization', 'hospital')
    list_filter = ('specialization',)

admin.site.register(Doctor, DoctorAdmin)


# Register Patient model
class PatientAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone', 'email', 'address')
    search_fields = ('user__first_name', 'user__last_name', 'phone', 'email')
    list_filter = ('user__username',)

admin.site.register(Patient, PatientAdmin)


# Register HeartDiseasePrediction model
class HeartDiseasePredictionAdmin(admin.ModelAdmin):
    list_display = ('patient', 'prediction_result', 'probability', 'created_at', 'age', 'sex', 'cp')
    search_fields = ('patient__user__username', 'age', 'sex', 'cp')
    list_filter = ('prediction_result',)

admin.site.register(HeartDiseasePrediction, HeartDiseasePredictionAdmin)


# Register DiabetesPrediction model
class DiabetesPredictionAdmin(admin.ModelAdmin):
    list_display = ('patient', 'prediction_result', 'probability', 'created_at', 'Age', 'BMI', 'Sex')
    search_fields = ('patient__user__username', 'Age', 'BMI', 'Sex')
    list_filter = ('prediction_result',)

admin.site.register(DiabetesPrediction, DiabetesPredictionAdmin)


from .models import KidneyDiseasePrediction

class KidneyDiseasePredictionAdmin(admin.ModelAdmin):
    list_display = (
        'patient',
        'get_prediction_status',
        'probability',
        'created_at',
        'age',
        'blood_pressure',
        'doctor'
    )
    search_fields = (
        'patient__user__username',
        'patient__user__first_name',
        'patient__user__last_name',
        'doctor__user__username',
        'doctor__user__first_name',
        'doctor__user__last_name',
        'age',
        'blood_pressure'
    )
    list_filter = (
        'prediction_result',
        'created_at',
        'hypertension',
        'diabetes_mellitus',
        'coronary_artery_disease',
    )
    readonly_fields = (
        'created_at',
        'probability',
        'prediction_result'
    )
    fieldsets = (
        ('Patient Information', {
            'fields': ('patient', 'doctor', 'created_at')
        }),
        ('Prediction Results', {
            'fields': ('prediction_result', 'probability'),
            'classes': ('collapse',)
        }),
        ('Clinical Measurements', {
            'fields': (
                ('age', 'blood_pressure'),
                ('specific_gravity', 'albumin', 'sugar'),
                ('blood_glucose_random', 'blood_urea', 'serum_creatinine'),
                ('sodium', 'potassium', 'hemoglobin'),
                ('packed_cell_volume', 'white_blood_cells', 'red_blood_cells')
            )
        }),
        ('Microscopic Examination', {
            'fields': (
                'red_blood_cells_normal',
                'pus_cells_normal',
                'pus_cell_clumps_present',
                'bacteria_present'
            ),
            'classes': ('collapse',)
        }),
        ('Medical History', {
            'fields': (
                'hypertension',
                'diabetes_mellitus',
                'coronary_artery_disease'
            ),
            'classes': ('collapse',)
        }),
        ('Symptoms', {
            'fields': (
                'appetite',
                'pedal_edema',
                'anemia'
            ),
            'classes': ('collapse',)
        }),
    )
    
    def get_prediction_status(self, obj):
        return "Positive" if obj.prediction_result else "Negative"
    get_prediction_status.short_description = 'Status'
    get_prediction_status.admin_order_field = 'prediction_result'

    def get_readonly_fields(self, request, obj=None):
        # Make all fields readonly if prediction already exists
        if obj:  # editing an existing object
            return self.readonly_fields + tuple(
                field.name for field in obj._meta.fields 
                if field.name not in ['id', 'patient', 'doctor', 'created_at']
            )
        return self.readonly_fields

admin.site.register(KidneyDiseasePrediction, KidneyDiseasePredictionAdmin)