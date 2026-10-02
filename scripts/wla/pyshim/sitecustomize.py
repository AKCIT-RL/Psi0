"""Dentro do container do SIMPLE (apptainer) `ctypes.util.find_library` não encontra GL/EGL (sem ldconfig/gcc/ld),
e o PyOpenGL do MuJoCo falha com 'NoneType has no attribute eglQueryString/glGetError'. Mapeia os sonames
conhecidos (as libs glvnd estão na imagem; o driver NVIDIA entra via --nv)."""
import ctypes.util as _u

_orig = _u.find_library
_MAP = {"GL": "libGL.so.1", "EGL": "libEGL.so.1", "GLU": "libGLU.so.1", "GLX": "libGLX.so.0",
        "OpenGL": "libOpenGL.so.0", "GLESv2": "libGLESv2.so.2", "OSMesa": "libOSMesa.so.8"}


def find_library(name):
    return _orig(name) or _MAP.get(name)


_u.find_library = find_library
