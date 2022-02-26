from django import forms

from main import models

class machine_translation_form(forms.ModelForm):

    class Meta:
        model = models.machine_translation
        fields = '__all__'  # include = ('name', 'roll_no')

