from django.shortcuts import render
posts = [
{
'author': 'The Gaurdian',
'title': 'Call of Duty: Modern Warfare',
'review': 'Call of Duty is perhaps the most divisive '
          'mainstream gaming brand of all time; '
          'a gung-ho, partisan blockbuster combat '
          'romp selling us a vision of rough and '
          'ready spec-ops superstars travelling the '
          'globe with their guns and their competence, '
          'helping freedom fighters while killing rogue '
          'paramilitary groups, without pausing too long '
          'to consider the differences between them.',
'date_posted': 'March 23, 2019',
},
{
'author': 'IGN',
'title': 'Star Wars Jedi: Fallen Order',
'review': 'Jedi: Fallen Order pushes all the right buttons '
          'for a Star Wars action-adventure. '
          'It is a genre remix that samples the combat'
          ' and exploration of a lightened-up Dark Souls '
          'and the action and energy of Uncharted, and '
          'that works out to be a great fit for the return '
          'of the playable Jedi.',
'date_posted': 'March 21, 2019',
}]


def home(request):
    context = {
        'posts':posts
    }
    return render(request, 'reviews/home.html', context)
