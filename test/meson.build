incdir = ['../include']

qa_tests = [
  'qa_absolute_value',
  'qa_add_const',
  'qa_add',
  'qa_and',
  'qa_arg_max',
  'qa_arg_min',
  'qa_channelizer',
  'qa_complex_to_mag_squared',
  'qa_complex_to_mag',
  'qa_conjugate',
  'qa_convolve',
  'qa_copy',
  'qa_deinterleave',
  'qa_divide',
  'qa_dot_product',
  'qa_exponentiate',
  'qa_fft_shift',
  'qa_keep_m_in_n',
  'qa_keep_one_in_n',
  'qa_max',
  'qa_min',
  'qa_moving_average',
  'qa_multiply_const',
  'qa_multiply',
  'qa_nlog10',
  'qa_not',
  'qa_or',
  'qa_subtract',
  'qa_threshold',
  'qa_xor'
]

foreach qa : qa_tests
    e = executable(qa, 
        qa + '.cu', 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [cusp_dep, gtest_dep], 
        install : false)
    test(qa, e)
endforeach


e = executable('qa_multiply_const',
    'qa_multiply_const.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep],
    install : false)
test('Test Multiply Constant', e)

e = executable('qa_fft_shift',
    'qa_fft_shift.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test FFT Shift', e)

e = executable('qa_nlog10',
    'qa_nlog10.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Nlog10', e)

e = executable('qa_conjugate',
    'qa_conjugate.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test conjugate', e)

e = executable('qa_absolute_value',
    'qa_absolute_value.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Absolute Value', e)

e = executable('qa_exponentiate',
    'qa_exponentiate.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Exponentiate', e)

e = executable('qa_complex_to_mag',
    'qa_complex_to_mag.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Complex to Mag', e)

e = executable('qa_complex_to_mag_squared',
    'qa_complex_to_mag_squared.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Complex to Mag Squared', e)

e = executable('qa_and',
    'qa_and.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test And', e)

e = executable('qa_or',
    'qa_or.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Or', e)

e = executable('qa_xor',
    'qa_xor.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Xor', e)

e = executable('qa_not',
    'qa_not.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Not', e)

e = executable('qa_add',
    'qa_add.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Add', e)

e = executable('qa_divide',
    'qa_divide.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Divide', e)

e = executable('qa_multiply',
    'qa_multiply.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Multiply', e)

e = executable('qa_convolve',
    'qa_convolve.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Convolve', e)

e = executable('qa_dot_product',
    'qa_dot_product.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Dot Product', e)

e = executable('qa_moving_average',
    'qa_moving_average.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Moving Average', e)

e = executable('qa_threshold',
    'qa_threshold.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Threshold', e)

e = executable('qa_max',
    'qa_max.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Max', e)

e = executable('qa_deinterleave',
    'qa_deinterleave.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Deinterleave', e)

e = executable('qa_arg_max',
    'qa_arg_max.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Arg Max', e)

e = executable('qa_arg_min',
    'qa_arg_min.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Arg Min', e)

e = executable('qa_keep_m_in_n',
    'qa_keep_m_in_n.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Keep M In N', e)

e = executable('qa_keep_one_in_n',
    'qa_keep_one_in_n.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Keep One In N', e)

e = executable('qa_subtract',
    'qa_subtract.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Subtract', e)

e = executable('qa_min',
    'qa_min.cu',
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : false)

test('Test Min', e)
