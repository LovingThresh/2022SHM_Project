import json
from collections import namedtuple


def dict2cls(dic):

    json_str = json.dumps(dic)
    dict_cls = json.loads(json_str, object_hook=lambda d: namedtuple("X", d.keys())(*d.values()))

    return dict_cls
