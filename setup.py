import os
import glob
import pocky
import textwrap
import numpy as np
from setuptools import Extension, setup
from setuptools.command.sdist import sdist

source_dir = 'src/greenlantern/ext'
header_dir = os.path.join(source_dir, 'include')
kernel_dir = os.path.join(source_dir, 'kernels')

def gen_kernel_paths():
    for kernel_input in glob.glob(os.path.join(kernel_dir, '*.cl')):
        kernel_name = os.path.splitext(os.path.basename(kernel_input))[0]
        yield kernel_name, kernel_input

kernel_paths = list(gen_kernel_paths())
kernel_frag_template = textwrap.dedent('''\
const char kernel_{kernel_name}[] = R"===(
{kernel_content}
)===";\n\n''')
kernel_agg_template = textwrap.dedent('''\
const cl_uint num_kernel_frags = {num_kernels};
const char *kernel_frags[] = {{ {kernel_vars} }};
''')

class GenerateKernelFragmentsCommand(sdist):
    def run(self):
        with open(os.path.join(header_dir, 'greenlantern_kernels.h'), 'w') as kernel_def:
            for kernel_name, kernel_input in kernel_paths:
                with open(kernel_input, 'r') as kernel_in:
                    kernel_content = kernel_in.read()
                    kernel_frag = kernel_frag_template.format(
                        kernel_content=kernel_content, kernel_name=kernel_name)
                    kernel_def.write(kernel_frag)

            kernel_agg = kernel_agg_template.format(
                num_kernels=len(kernel_paths),
                kernel_vars=', '.join([f'kernel_{name}' for name, _ in kernel_paths]))
            kernel_def.write(kernel_agg)

        return super().run()

source_files = ['greenlantern.c', 'greenlantern_context.c', 'ellipsoid.c']
source_files = [os.path.join(source_dir, fname) for fname in source_files]

header_files = ['greenlantern.h', 'greenlantern_context.h']
header_files = [os.path.join(header_dir, fname) for fname in header_files]

ext_modules = [
    Extension(name='greenlantern.ext', sources=source_files, language='c',
        include_dirs=[header_dir, pocky.get_include(), np.get_include()],
        libraries=['OpenCL'], depends=header_files)
]

cmdclass = {
    'sdist': GenerateKernelFragmentsCommand,
}

setup(cmdclass=cmdclass, ext_modules=ext_modules)
