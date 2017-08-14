from distutils.core import setup, Extension
from Cython.Build import cythonize

module1 = Extension("kalman_filter_wrap",
                sources=["src/kalman_filter_wrap.pyx", "src/kalman_filter.c"],
                extra_compile_args=[],
                extra_link_args=["-lgsl", "-lgslcblas", "-lm"]
                )
              
setup(name = 'PyStateSpace', package_dir={'': 'src'}, py_modules=['py_state_space'], ext_modules = cythonize([module1]))      
