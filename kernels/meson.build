
cusp_cu_files = files([
    'copy.cu', 
    'add.cu',
    'add_const.cu',
    'multiply_const.cu',
    'fft_shift.cu',
    'nlog10.cu',
    'conjugate.cu',
    'absolute_value.cu',
    'exponentiate.cu',
    'complex_to_mag.cu',
    'complex_to_mag_squared.cu'])

incdir = ['../include']

cusp = library('cusp', 
    cusp_cu_files, 
    include_directories : incdir, 
    install : true, 
    dependencies : [cuda_dep])

cusp_dep = declare_dependency(include_directories : incdir,
                    link_with : cusp,
                    dependencies : cuda_dep)