from django import template

register = template.Library()

@register.filter(name='get_field')
def get_field(form, key):
    """Retrieve a field from the form safely"""
    if form is None:
        return None
    try:
        return form[key]  # Return the BoundField object directly
    except KeyError:
        return None  # Return None if the field is missing
