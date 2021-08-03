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
    test('Test Copy', e)
endforeach


