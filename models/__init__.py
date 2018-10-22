from __future__ import  absolute_import

from .vgg import *

# __factory = {
#     'densenet_org': densenet_org,
#
# }
#
# def create(name, *args, **kwargs):
#
#     if name not in __factory:
#         raise KeyError("Unknow model:", name)
#     return __factory[name](*args, **kwargs)