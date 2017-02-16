from datetime import datetime
from django.shortcuts import render


def temp(request):
    time = str(datetime.now())
    tag_lib = {'time': time}
    return render(request, 'temperature.html', tag_lib)