from .base import *

# Database
# https://docs.djangoproject.com/en/3.2/ref/settings/#databases

EMAIL_BACKEND='django.core.mail.backends.smtp.EmailBackend'
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


# POSTGRES_ENGINE=django.db.backends.postgresql
# POSTGRES_USER=admin
# POSTGRES_PASSWORD=password123
# POSTGRES_DB=estate
# PG_HOST=localhost
# PG_PORT=5432