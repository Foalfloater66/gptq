from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# Optional: Set CUDA_HOME if it's not set globally and you know the path
# cuda_home_path = '/server/opt/cuda/cuda-11.7' # Example path
# if 'CUDA_HOME' not in os.environ and os.path.exists(cuda_home_path):
#     os.environ['CUDA_HOME'] = cuda_home_path
#     print(f"Set CUDA_HOME to {cuda_home_path}")

# Check if CUDA_HOME is set, fail early if not
if 'CUDA_HOME' not in os.environ:
    raise EnvironmentError("CUDA_HOME environment variable is not set. "
                           "Please set it or uncomment/edit the path in setup script.")
else:
    print(f"Using CUDA_HOME={os.environ['CUDA_HOME']}")


setup(
    name='logmatvec_cuda', # Name of the Python package
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='logmatvec_cuda', # Must match the name passed to PYBIND11_MODULE
            sources=[
                'logmatvec_cuda.cpp',
                'logmatvec_cuda_kernel.cu'
            ],
            # Optional: Add extra compile args if needed
            # extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O3']}
        )
    ],
    # Command mapping for the build process
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)

print("\nSetup script finished.")
print("To build the extension, run: python setup_logmatvec_cuda.py install")
