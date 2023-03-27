from .base import *

# Database
# https://docs.djangoproject.com/en/3.2/ref/settings/#databases

EMAIL_BACKEND= "djcelery_email.backends.CeleryEmailBackend"
EMAIL_HOST=env("EMAIL_HOST")
EMAIL_USER_TLS=True
EMAIL_PORT=env("EMAIL_PORT")
EMAIL_HOST_USER=env("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD=env("EMAIL_HOST_PASS")
DEFAULT_FROM_EMAIL='info@real_estate.com'
DOMAIN=env("DOMAIN")
SITE_NAME="Real Estate"



DATABASES = {
    'default': {
        'ENGINE': env("POSTGRES_ENGINE"),
        "NAME": env("POSTGRES_DB"),
        "USER": env("POSTGRES_USER"),
        "PASSWORD": env("POSTGRES_PASSWORD"),
        "HOST": env("PG_HOST"),
        "PORT": env("PG_PORT"),
    }
}

CELERY_BROKER_URL = env("CELERY_BROKER")
CELERY_RESULT_BACKEND = env("CELERY_BACKEND")
CELERY_TIMEZONE = "Africa/Kigali"


# POSTGRES_ENGINE=django.db.backends.postgresql
# POSTGRES_USER=admin
# POSTGRES_PASSWORD=password123
# POSTGRES_DB=estate
# PG_HOST=localhost
# PG_PORT=5432