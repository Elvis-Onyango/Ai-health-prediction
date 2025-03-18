from django.urls import path
from .views import predict_view,signup_view,login_view,logout_view,dashboard_view,index,admin_dashboard

urlpatterns = [
    path('predict/', predict_view, name='predict'),
    path("signup/", signup_view, name="signup"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("dashboard/", dashboard_view, name="dashboard"), 
    path('admin-dashboard/',admin_dashboard,name='admin-dashboard'),
    path('', index,name = 'landing-page')
]

