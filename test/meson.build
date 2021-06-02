src1 = ['qa_copy.cu']
src2 = ['qa_add_const.cu']
incdir = ['../include']
e = executable('qa_copy', 
    src1, 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
e2 = executable('qa_add_const.cu', 
    src2, 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
test('Test Copy', e)
test('Test Add Constant', e2)