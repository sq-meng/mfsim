from time import ctime
from django.shortcuts import render
import os


def temp(request):
    get = request.GET
    try:
        if get['pass'] == os.environ['MYPASSWORD']:

            time = ctime()
            tag_lib = {'time': time}
            return render(request, 'temperature.html', tag_lib)
        else:
            return render(request, 'temperror.html', {})
    except KeyError:
        return render(request, 'temperror.html', {})