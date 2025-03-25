from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('files/', views.list_indexed_files, name='list_files'),
    path('files/<int:file_id>/', views.get_file_details, name='file_details'),
    path('search/', views.search_files, name='search_files'),
]