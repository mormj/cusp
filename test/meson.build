srcs = ['qa_copy.cu']
incdir = ['../include']
e = executable('qa_copy', 
    srcs, 
    include_directories : incdir, 
    link_language : 'cpp',
    dependencies: [cusp_dep, gtest_dep], 
    install : true)
test('Copy Kernel Tests', e)