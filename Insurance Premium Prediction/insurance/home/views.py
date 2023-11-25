from django.shortcuts import render,HttpResponse
import joblib

model = joblib.load("home/static/random_forest_regressor")

# Create your views here.
def index(request):
    return render(request,'index.html')


def pred(request):
    if request.method == 'POST':
        age = float(request.POST.get('age'))
        sex = float(request.POST.get('sex'))
        bmi = float(request.POST.get('bmi'))
        children = float(request.POST.get('children'))
        smoker = float(request.POST.get('smoker'))
        region = float(request.POST.get('region'))
        pred = model.predict([[age,sex,bmi,children,smoker,region]])
        output = {
            "output":pred
        }

        return render(request,'predict.html',output)

    else:
        return render(request,'predict.html')