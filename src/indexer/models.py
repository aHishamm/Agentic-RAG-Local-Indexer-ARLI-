from django.db import models
import numpy as np

class Indexer(models.Model):
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    file_type = models.CharField(max_length=50)
    creation_date = models.DateTimeField()
    size = models.BigIntegerField()
    embedding = models.BinaryField(null=True)  # Store filename embeddings as binary
    is_textual = models.BooleanField(default=False) # True if the file is textual

    def set_embedding(self, embedding_array: np.ndarray):
        self.embedding = embedding_array.tobytes()

    def get_embedding(self) -> np.ndarray:
        if self.embedding:
            return np.frombuffer(self.embedding, dtype=np.float32)
        return None

    def __str__(self):
        return self.file_name

    class Meta:
        verbose_name = 'Indexer'
        verbose_name_plural = 'Indexers'