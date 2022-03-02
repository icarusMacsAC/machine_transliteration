# MACHINE TRANSLITERATON
<br>
  
## Agenda : 
- Introduction
- Application
- Example
- About Model
- Steps to Run this app in your laptop
- Contact Information
  
## 1. Introduction 
  
> <img width="948" alt="Screenshot (2875)" src="https://user-images.githubusercontent.com/94113845/155877867-5d6ad8d4-54d7-4013-b1cc-5292c31f77da.png">
> <img width="948" alt="Screenshot (2876)" src="https://user-images.githubusercontent.com/94113845/155877885-ba6df062-f6cf-4d2d-9876-536a19a8185b.png">
  
  It is challenging to translate names and technical terms across languages with different alphabets and sound inventories. These items are commonly trnasliterated, i.e., replaced with approximate phonetic equivalents. 
  
  For example, "computer" in English comes out as "konpyuutaa" in Japanese. Translating such items from Japanese back to English is even more challenging, and of practical interest, as transliterated items make up the bulk of text phrases not found in bilingual dictionaries. We describe and evaluate a method for performing backwards transliterations by machine. This method uses a generative model, incorporating several distinct stages in the transliteration process.
  

## 2. Application
  
1. Aid to the People <br>
> <img width="905" alt="img-6" src="https://user-images.githubusercontent.com/94113845/155892169-8db2e1ef-c3f2-4fbc-9a95-73b55ba3b963.png">
  
2. Hindi Typing <br>
  
> <img width="765" alt="img-7" src="https://user-images.githubusercontent.com/94113845/155877949-68fc5ffa-30d7-408b-97d0-379f37d00f0f.png">
> <img width="762" alt="img-8" src="https://user-images.githubusercontent.com/94113845/155870581-3f18891c-5962-430b-a985-d85c70686b5b.png">
> Source : https://indiatyping.com/index.php/hindi-typing

Hindi Typing is very easy with above method. Just type in English as you type messages in Mobile and press space bar. It will convert in Hindi. If you think you don't get desired word, you can press backspace key to open word suggestion list from which you can select best fit word for your typing. Suggestions list will also appear when you click on that word with mouse. India Typing is Free and Fastest method for Type in Hindi online - हिन्दी मे टाइप करे, without learning Hindi keyboard actually.
  
## 3. Example :-

> <img width="1890" alt="img-9" src="https://user-images.githubusercontent.com/94113845/155872588-b5169509-6a7c-42d3-a6a5-f47e52468bb5.png">
  
> <img width="1896" alt="img-10" src="https://user-images.githubusercontent.com/94113845/155872715-9a80bf76-876e-4c80-a43b-d8a3e5e376bb.png">

## 4. ABOUT MODEL :-
1. A machine transliteration model is a Character based model.
2. In this model I used RNN to generate a transliteration based output of any Sentence.
  
> ![image](https://user-images.githubusercontent.com/94113845/155873911-644bcc98-ac49-4e47-90bc-881d7d9a11f3.png)
  
3. Here you can see that in a above fig, That there are 2 input to the model and 1 output
4. 2 Input are (Ist iteration):
- Whole English sentence
- Ist character of Hindi sentence(ie. <start>)
- After that it predict a next character as a output, 
5. Now Repeat 4th step until we get a end character(ie. <end>)
6. Then combile all character and display as a output


## 5. Steps to Run this app in your laptop
  
## Step 1 : Clone the git repository in your laptop
  
  1. Open Anaconda Promt
  2. Go to the any location where u want to save this project
  3. Copy a git repository url 

> <img width="946" alt="img-2" src="https://user-images.githubusercontent.com/94113845/155867444-03e2cdca-3020-412b-b0d9-d64374d0c823.png">

  4. Now clone a repository : "*git clone URL*" 
  
> <img width="946" alt="img-1" src="https://user-images.githubusercontent.com/94113845/155867281-dcfd6f67-c9ab-4e2d-9cd6-844b9c43074a.png">


## Step 2 : Create a virtual environment
  1. Enter inside machine_transliteration
  2. Now create a virtual env : "*virtualenv env*"
  3. Activate the environvent : "*.\env\Scripts\activate*" 
  
> <img width="946" alt="img-3" src="https://user-images.githubusercontent.com/94113845/155867612-46143d0a-e7a8-4bbd-a3a4-e7bf201d8e38.png">
  
## Step 3 : Download all library
  1. Open requirements.txt file and then install all library which are present there.
  
## Step 4 : Run the project
  
  1. Now run a python command to execute manage.py file : "*python manage.py runserver*" 
  <br>
  
  > <img width="946" alt="image" src="https://user-images.githubusercontent.com/94113845/155850438-6f21154f-ecf5-49aa-993b-dfc85bf288ca.png">
  
  <br>
  
  2. Now open any browser and type : "*http://localhost:8000/*"
  
  > <img width="946" alt="image" src="https://user-images.githubusercontent.com/94113845/155853199-4f8d82f0-4a95-483a-8b4f-992da67582db.png">
  
<!--   ![image](https://user-images.githubusercontent.com/94113845/155853199-4f8d82f0-4a95-483a-8b4f-992da67582db.png) -->
   <br>
  
## 6. Contact Information :-
  
> <img width="946" alt="img-11" src="https://user-images.githubusercontent.com/94113845/155877315-1523a65e-c18f-43f1-8396-6e8610e6efb9.png">
  
## Copyright

Copyright Amit Chourey

