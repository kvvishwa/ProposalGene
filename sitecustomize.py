# Runs automatically if present on sys.path (PEP 370 behavior via site.py)
import numpy as _np

# Restore aliases removed in NumPy 2.x that older libs still reference
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64
if not hasattr(_np, "int_"):
    _np.int_ = _np.int64
if not hasattr(_np, "complex_"):
    _np.complex_ = _np.complex128

# (Optional) classic aliases some libs still reach for
for _name, _py in {"float": float, "int": int, "complex": complex, "bool": bool,
                   "object": object, "str": str}.items():
    if not hasattr(_np, _name):
        setattr(_np, _name, _py)
