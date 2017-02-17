from time import ctime
from django.shortcuts import render


def temp(request):
    time = ctime()
    tag_lib = {'time': time}
    return render(request, 'temperature.html', tag_lib)