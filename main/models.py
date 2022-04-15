from django.db import models
import main

# Create your models here.

class ImageCaption(models.Model):
    # name = models.CharField(max_length=256)
    # img = models.ImageField(upload_to ='uploads/image', null=True, blank=True)
    img = models.FileField(upload_to ='uploads/image', null=True, blank=True)
    def __str__(self):
        return str(main.models.ImageCaption.objects.all()[0].img).split(".")[0].split("/")[-1]

class machine_translation(models.Model):
    OUTPUT_FORMAT = (
        ('transliteration', 'Machine Transliteration'),
        ('translation', 'Machine Translation'),
        ('both', 'Both')
    )
    enter_text = models.CharField(max_length=256)
    format = models.CharField(max_length=15, choices=OUTPUT_FORMAT)

