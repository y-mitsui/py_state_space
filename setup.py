from distutils.core import setup, Extension

module1 = Extension('kalman',
                    define_macros = [('HAVE_INLINE', '1')],
                    include_dirs = ['/usr/include/glib-2.0','/usr/lib/x86_64-linux-gnu/glib-2.0/include','/usr/include/atlas','/usr/include/apr-1.0','/usr/include/libxml2'],
                    libraries = ['m','gsl','gslcblas'],
                    library_dirs = ['/usr/local/lib','/usr/lib/x86_64-linux-gnu'],
                    sources = ['kalman.c'],
                    extra_compile_args = [])
                    
setup(
    name='PyStateSpace',
    version='1.0',
    py_modules=['py_state_space'],ext_modules = [module1]
)
