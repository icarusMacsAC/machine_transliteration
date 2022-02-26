from django.db import models

# Create your models here.

class machine_translation(models.Model):

    OUTPUT_FORMAT = (
        ('transliteration', 'Machine Transliteration'),
        ('translation', 'Machine Translation'),
        ('both', 'Both')
    )

    enter_text = models.CharField(max_length=256)

    format = models.CharField(max_length=15, choices=OUTPUT_FORMAT)






















