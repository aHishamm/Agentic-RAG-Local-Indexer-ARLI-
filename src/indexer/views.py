from django.shortcuts import render
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db.models import Q
from datetime import datetime, timedelta
import numpy as np
from .models import Indexer
from .forms import SearchForm
from .utils import search_files_with_rag, _format_size

def index(request):
    return render(request, 'indexer/index.html')

def list_indexed_files(request):
    # Get sort parameters
    sort_by = request.GET.get('sort', 'file_name')
    order = request.GET.get('order', 'asc')
    page_number = request.GET.get('page', 1)
    
    # Get filter parameters
    date_filter = request.GET.get('date_filter', '')
    size_filter = request.GET.get('size_filter', '')
    type_filter = request.GET.get('type_filter', '')
    
    # Start with all files
    files = Indexer.objects.all()
    
    # Apply date filter
    if date_filter:
        today = datetime.now()
        if date_filter == 'today':
            files = files.filter(creation_date__date=today.date())
        elif date_filter == 'yesterday':
            yesterday = today - timedelta(days=1)
            files = files.filter(creation_date__date=yesterday.date())
        elif date_filter == 'last_week':
            last_week = today - timedelta(days=7)
            files = files.filter(creation_date__gte=last_week)
        elif date_filter == 'last_month':
            last_month = today - timedelta(days=30)
            files = files.filter(creation_date__gte=last_month)
    
    # Apply size filter
    if size_filter:
        if size_filter == 'small':  # < 1MB
            files = files.filter(size__lt=1024*1024)
        elif size_filter == 'medium':  # 1MB - 10MB
            files = files.filter(size__gte=1024*1024, size__lt=10*1024*1024)
        elif size_filter == 'large':  # > 10MB
            files = files.filter(size__gte=10*1024*1024)
    
    # Apply type filter
    if type_filter:
        files = files.filter(file_type=type_filter)
    
    # Get unique file types for the filter dropdown
    file_types = Indexer.objects.values_list('file_type', flat=True).distinct()
    
    # Apply sorting
    files = files.order_by(f'{"-" if order == "desc" else ""}{sort_by}')
    
    # Format the size and embedding for display
    for file in files:
        file.formatted_size = _format_size(file.size)
        embedding = file.get_embedding()
        if embedding is not None:
            preview_values = embedding[:3]
            file.embedding_preview = f"[{', '.join(f'{v:.4f}' for v in preview_values)}...]"
        else:
            file.embedding_preview = None
    
    # Add pagination
    paginator = Paginator(files, 25)
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'sort_by': sort_by,
        'order': order,
        'total_files': files.count(),
        'file_types': file_types,
        'current_filters': {
            'date': date_filter,
            'size': size_filter,
            'type': type_filter,
        }
    }
    
    return render(request, 'indexer/list_files.html', context)

def get_file_details(request, file_id):
    try:
        file = Indexer.objects.get(id=file_id)
        return JsonResponse({
            "file": {
                "id": file.id,
                "name": file.file_name,
                "path": file.file_path,
                "type": file.file_type,
                "creation_date": file.creation_date,
                "size": file.size
            }
        })
    except Indexer.DoesNotExist:
        return JsonResponse({"error": "File not found"}, status=404)

def search_files(request):
    """
    Search for files using natural language and the RAG model
    """
    results = []
    query = ''
    
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            top_n = form.cleaned_data['top_n']
            
            # Use the search_files_with_rag utility function
            results = search_files_with_rag(query, top_n)
    else:
        form = SearchForm()

    # If it's an AJAX request, return JSON response
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'results': results,
            'query': query
        })
    
    # Otherwise render the template with the form and results
    context = {
        'form': form,
        'results': results,
        'query': query,
        'title': 'Search Files',
    }
    return render(request, 'indexer/search.html', context)