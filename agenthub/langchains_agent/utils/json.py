import json


def my_encoder(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()


def dumps(obj, **kwargs):
    return json.dumps(obj, default=my_encoder, **kwargs)
