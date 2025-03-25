from django.shortcuts import render
from django.http import JsonResponse
from .models import Indexer
from .forms import SearchForm
from .utils import search_files_with_rag

def index(request):
    return JsonResponse({"message": "Welcome to the Agentic RAG Indexer!"})

def list_indexed_files(request):
    files = Indexer.objects.all()
    return JsonResponse({"files": list(files.values())})

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