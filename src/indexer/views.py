from django.shortcuts import render
from django.http import JsonResponse
from .models import Indexer

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