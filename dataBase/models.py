# models.py (the database tables)

import views
from django.conf.urls.defaults import *
from models import Book
from django.shortcuts import render_to_response
from django.db import models


class Book(models.Model):
    name = models.CharField(max_length=50)
    pub_date = models.DateField()


# views.py (the business logic)


def latest_books(request):
    book_list = Book.objects.order_by('-pub_date')[:10]
    return render_to_response('latest_books.html', {'book_list': book_list})


# urls.py (the URL configuration)


urlpatterns = patterns('',
                       (r'^latest/$', views.latest_books),
                       )


# latest_books.html (the template)

<html > <head > <title > Books < /title > </head >
<body >
<h1 > Books < /h1 >
<ul >
{% for book in book_list % }
<li > {{book.name}} < /li >
{% endfor % }
</ul >
</body > </html >
