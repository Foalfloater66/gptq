from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# Set CUDA_HOME if necessary, or ensure nvcc is in your PATH
# Example: os.environ['CUDA_HOME'] = '/usr/local/cuda'
# Check if CUDA_HOME is set, fail early if not
if 'CUDA_HOME' not in os.environ:
    raise EnvironmentError("CUDA_HOME environment variable is not set. "
                           "Please set it or uncomment/edit the path in setup script.")
else:
    print(f"Using CUDA_HOME={os.environ['CUDA_HOME']}")


setup(
    name='quant_cuda_4bit', # The name of the package
    ext_modules=[cpp_extension.CUDAExtension(
        name='quant_cuda_4bit', # The name of the extension module (what you import)
        sources=[
            'quant_cuda_4bit.cpp',      # C++ source file (bindings)
            'quant_cuda_kernel_4bit.cu' # CUDA source file (kernel implementation)
        ],
        # Optional: Add extra compile args if needed, e.g., for specific GPU architectures
        # See PyTorch documentation for cpp_extension.CUDAExtension for details
        # Example for targeting specific compute capability:
        # extra_compile_args={'nvcc': ['-gencode=arch=compute_75,code=sm_75']} # Example for Turing
        # Force the CXX11 ABI to match PyTorch (likely built with new ABI)
        # Pass the flag to both the C++ compiler and via nvcc to its host compiler
        # extra_compile_args={
            # 'cxx': ['-D_GLIBCXX_USE_CXX11_ABI=1'],
            # 'nvcc': ['-Xcompiler', '-D_GLIBCXX_USE_CXX11_ABI=1']
        # }
    )],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension # Use PyTorch's build extension command
    }
)
