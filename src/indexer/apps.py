from django.apps import AppConfig
from django.db.models.signals import post_save, post_delete
import os

class IndexerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'indexer'
    path = os.path.dirname(os.path.abspath(__file__))
    
    def ready(self):
        # Import models here to avoid circular imports
        from .models import Indexer
        from django.dispatch import receiver
        
        @receiver(post_save, sender=Indexer)
        def handle_file_save(sender, instance, created, **kwargs):
            if created:
                print(f"New file indexed: {instance.file_name}")
            else:
                print(f"Updated indexed file: {instance.file_name}")
                
        @receiver(post_delete, sender=Indexer)
        def handle_file_delete(sender, instance, **kwargs):
            print(f"Deleted indexed file: {instance.file_name}")