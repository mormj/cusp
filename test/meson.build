src1 = ['qa_copy.cu']
src2 = ['qa_add_const.cu']
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

e = executable('qa_add', 
    'qa_add.cu', 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
test('Test Add', e)