from django.urls import path 
from main import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('index', views.index, name='index'),
    path('index2', views.index2, name='index2'),
    path('result', views.result, name='result'),
    path('res', views.res, name='res'),
    path('', views.index, name='index'),
    path('about_me', views.about_me, name='about_me'),
    path('technology', views.technology, name='technology'),
    path('projects', views.projects, name='projects')
] 

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
