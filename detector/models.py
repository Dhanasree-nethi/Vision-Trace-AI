from django.db import models

class DetectedObject(models.Model):

    object_name = models.CharField(max_length=100)

    object_image = models.ImageField(upload_to='objects/')

    confidence = models.FloatField(default=0)

    detected_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.object_name