from django.urls import path
from .views import( predict_heart_view,doctor_dashboard,
    predict_kidney_view,user_detail, edit_user, delete_user,
     create_user,predict_diabetes_view,signup_view,login_view,
     logout_view,dashboard_view,index,admin_dashboard,
     create_patient_profile, create_doctor_profile
     )

app_name = 'predictor'

urlpatterns = [
    path("signup/", signup_view, name="signup"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("dashboard/", dashboard_view, name="dashboard"), 
    path('admin-dashboard/',admin_dashboard,name='admin-dashboard'),
    path('predict/heart',predict_heart_view, name = 'predict-heart'),
    path('predict/kidney',predict_kidney_view, name ='predict-kidney'),
    path('predict/diabetes', predict_diabetes_view, name='predict-diabetes'),
    path("create-patient-profile/", create_patient_profile, name="create_patient_profile"),
    path("create-doctor-profile/", create_doctor_profile, name="create_doctor_profile"),
    path('user/<int:user_id>/', user_detail, name='user-detail'),
    path('edit-user/<int:user_id>/', edit_user, name='edit-user'),
    path('delete-user/<int:user_id>/', delete_user, name='delete-user'),
    path('create-user/', create_user, name='create-user'),
    path('doctor-dashboard/',doctor_dashboard,name = 'doctor-dashboard'),
    path('', index,name = 'landing-page')

]

