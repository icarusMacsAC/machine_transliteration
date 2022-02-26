from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.conf import settings
from main import (
    models, forms, trans_res
)

# Create your views here.

def index(request):
    trans_form = forms.machine_translation_form()
    if request.method == 'POST':
        trans_form = forms.machine_translation_form(request.POST)
        if trans_form.is_valid(): # application label validation
            student = trans_form.save()
            return HttpResponseRedirect('/result')
    context = {
        'trans_form' : trans_form
    }
    return render(request, 'main/index.html', context)

def result(request):
    trans_form = models.machine_translation.objects.all()

    text = models.machine_translation.objects.all().last().enter_text
    format = models.machine_translation.objects.all().last().format 
    print(text, format)
    result = list(trans_res.main(text, format))
    print(result)


    context = {
        'trans_form' : trans_form,
        "projects" : settings.DATA["PROJECTS"]
    }

    if format == 'translation':
        context['translation'] = result[0]
    elif format == 'transliteration':
        context['transliteration'] = result[0]
    else :
        context['transliteration'] = result[0]
        context['translation'] = result[1]

    return render(request, 'main/result.html', context)

def technology(request):
    context = {
        "technology" : settings.DATA["TECHNOLOGY"]
    }
    print(context)
    return render(request, 'main/technology.html', context)

def projects(request):
    context = {
        "projects" : settings.DATA["PROJECTS"]
    }
    print(context)
    return render(request, 'main/projects.html', context)

    