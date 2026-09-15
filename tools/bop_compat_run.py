"""API compatibility for pinned official BOP scripts; no metric/threshold changes."""
import runpy,sys,inspect
from collections import namedtuple
if not hasattr(inspect,"getargspec"):
    ArgSpec=namedtuple("ArgSpec","args varargs keywords defaults")
    inspect.getargspec=lambda f:ArgSpec(*inspect.getfullargspec(f)[:4])
import numpy as np
for name,value in [('float',float),('int',int),('bool',bool),('complex',complex),('str',str)]:
    np.__dict__.setdefault(name,value)
from OpenGL.arrays.ctypesparameters import CtypesParameterHandler
handler=CtypesParameterHandler();handler.register(handler.HANDLED_TYPES)
sys.argv=sys.argv[1:]
if sys.argv[0].endswith('eval_calc_errors.py') and '--error_type=vsd' in sys.argv:
    from glumpy import app
    app.use('glfw')
runpy.run_path(sys.argv[0],run_name='__main__')
