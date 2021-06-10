
cusp_cu_files = files([
    'copy.cu', 
    'add.cu',
    'add_const.cu',
    'multiply_const.cu',
    'fft_shift.cu',
    'absolute_value.cu',
    'conjugate.cu',
    'exponentiate.cu',
    'qa_complex_to_mag',
    'qa_complex_to_mag_squared',
    'nlog10'])

incdir = ['../include']

cusp = library('cusp', 
    cusp_cu_files, 
    include_directories : incdir, 
    install : true, 
    dependencies : [cuda_dep])

cusp_dep = declare_dependency(include_directories : incdir,
                    link_with : cusp,
                    dependencies : cuda_dep)