from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from indexer.views import search_files

urlpatterns = [
    path('admin/', admin.site.urls),
    path('indexer/', include('indexer.urls')),
    path('', search_files, name='home'),  # Direct routing to search view
]