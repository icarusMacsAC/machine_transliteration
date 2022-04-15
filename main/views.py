from django.shortcuts import render, HttpResponseRedirect
import shutil, os
from main import models, forms, trans_res
from main.caption import add_pdf, find_pdf, find_fuzz, gen_res
from django.conf import settings
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
import numpy as np
import pandas as pd

# web: waitress-serve --port=$PORT ImageCaption.wsgi:application

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
    return render(request, 'main/index2.html', context)

def res(request):
    if request.method == 'GET':
        print('hhhhhhhhhhhhhhhhhhhhhh', request.get())

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

    # a = []
    b = []
    # for ele in context['transliteration'].split():
    #     print('hello', ele)
    #     a.append(find_pdf(ele))
    # print('a :', a)
    if request.method == 'POST':
        for ele in context['transliteration'].split():
            print('hello2', ele)
            b.append(find_fuzz(ele))
        print('b :', b)
        context['acc_df'] = gen_res(context['transliteration'], b)
    return render(request, 'main/result.html', context)

def index2(request):
    imgform = forms.ImageCaptionForm()
    context = {
        'imgform' : imgform
    }
    if request.method == 'POST':
        print("hello", request.POST)

        models.ImageCaption.objects.all().delete()
        try:
            shutil.rmtree("media/uploads/image/")
        except Exception as e:
            print(e)
        imgform = forms.ImageCaptionForm(request.POST, files=request.FILES)
        if imgform.is_valid():
            imgform = imgform.save()
            if str(models.ImageCaption.objects.all()[0].img) == "":
                url = "media/original/images.jpg"
            else:
                url = models.ImageCaption.objects.all()[0].img.url[1:]
            print(f"The selecterd image url is {url}")
            context = {
                'imgform' : url,
                # 'text' : generate_text(url),
                'text' : add_pdf(url),
                'res' : True
            }
            print(context['text'])
    return render(request, "main/index.html", context)

def about_me(request):
    context = {}
    return render(request, 'main/about_me.html', context)

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