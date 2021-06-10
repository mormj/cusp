src1 = ['qa_copy.cu']
src2 = ['qa_add_const.cu']
src3 = ['qa_multiply_const.cu']
src4 = ['qa_fft_shift.cu']
incdir = ['../include']
e = executable('qa_copy', 
    'qa_copy.cu', 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
test('Test Copy', e)

e = executable('qa_add_const', 
    'qa_add_const.cu', 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)

test('Test Add Constant', e)

e = executable('qa_multiply_const',
    src3,
    include_directories : incdir,
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep],
    install : true)
test('Test Multiply Constant', e)

e = executable('qa_fft_shift',
    src4,
    include_directories : incdir,
    link_language : 'cpp',
    dependencies : [cusp_dep, gtest_dep],
    install : true)

test('Test FFT Shift', e)

e = executable('qa_add', 
    'qa_add.cu', 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
test('Test Add', e)

e = executable('qa_absolute_value', 
    'qa_absolute_value.cu', 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
test('Test Absolute Value', e)

e = executable('qa_conjugate', 
    'qa_absolute_value.cu', 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
test('Test Conjugate', e)

e = executable('qa_exponentiate', 
    'qa_absolute_value.cu', 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
test('Test Exponentiate', e)

e = executable('qa_complex_to_mag)', 
    'qa_absolute_value.cu', 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
test('Test Magnitude', e)

e = executable('qa_complex_to_mag_squared)', 
    'qa_absolute_value.cu', 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
test('Test Magnitude Squared', e)

e = executable('qa_nlog10)', 
    'qa_absolute_value.cu', 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
test('Test Log10', e)