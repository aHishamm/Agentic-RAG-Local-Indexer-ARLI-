from django.db import models

class Indexer(models.Model):
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    file_type = models.CharField(max_length=50)
    creation_date = models.DateTimeField()
    size = models.BigIntegerField()

    def __str__(self):
        return self.file_name

    class Meta:
        verbose_name = 'Indexer'
        verbose_name_plural = 'Indexers'