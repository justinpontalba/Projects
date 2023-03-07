import json

from rest_framework.renderers import JSONOpenAPIRenderer


class ProfileJSONRenderer(JSONOpenAPIRenderer):
    charset = 'utf-8'

    def render(self, data, accepted_media_types=None, renderer_context = None):
        errors = data.get('errors', None)

        if errors is not None:
            return super(ProfileJSONRenderer(), self).render(data)

        return json.dumps({"profile": data})
        