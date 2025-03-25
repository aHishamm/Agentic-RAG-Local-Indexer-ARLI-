from django import forms

class SearchForm(forms.Form):
    """Form for searching files using natural language queries"""
    query = forms.CharField(
        label='Search',
        max_length=255,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search for files using natural language...'
        })
    )
    top_n = forms.IntegerField(
        label='Results',
        min_value=1,
        max_value=50,
        initial=5,
        widget=forms.NumberInput(attrs={
            'class': 'form-control'
        })
    )