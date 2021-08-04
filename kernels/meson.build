
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
    'complex_to_mag_squared.cu',
    'and.cu',
    'or.cu',
    'xor.cu',
    'not.cu',
    'add.cu',
    'divide.cu',
    'multiply.cu',
    'convolve.cu',
    'dot_product.cu',
    'moving_average.cu',
    'threshold.cu',
    'max.cu',
    'deinterleave.cu',
    'subtract.cu',
    'arg_max.cu',
    'arg_min.cu',
    'keep_m_in_n.cu',
    'keep_one_in_n.cu',
    'min.cu'])
    'min.cu',
    'channelizer.cu'])

incdir = ['../include']

cusp = library('cusp', 
    cusp_cu_files, 
    include_directories : incdir, 
    install : true, 
    dependencies : [cuda_dep, cusignal_dep])

cusp_dep = declare_dependency(include_directories : incdir,
                    link_with : cusp,
                    dependencies : cuda_dep)