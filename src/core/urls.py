from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', lambda request: redirect('indexer/')),  # Redirect root to indexer
    path('indexer/', include('indexer.urls')),
]