from django.shortcuts import render
from django.http import JsonResponse
from .models import IndexedFile

def index(request):
    return JsonResponse({"message": "Welcome to the Agentic RAG Indexer!"})

def list_indexed_files(request):
    files = IndexedFile.objects.all()
    return JsonResponse({"files": list(files.values())})

def get_file_details(request, file_id):
    try:
        file = IndexedFile.objects.get(id=file_id)
        return JsonResponse({"file": {"id": file.id, "name": file.name, "description": file.description}})
    except IndexedFile.DoesNotExist:
        return JsonResponse({"error": "File not found"}, status=404)