from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

def redirect_to_search(request):
    return redirect('search_files')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('indexer/', include('indexer.urls')),
    path('', redirect_to_search, name='home'),
]