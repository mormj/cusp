# SPDX-FileCopyrightText: 2020 Josh Morman 
#

project('newsched', 'cpp', 
  version : '0.0.0',
  license : 'GPLv3',
  default_options : ['cpp_std=c++14'])

cc = meson.get_compiler('cpp')
rt_dep = cc.find_library('rt', required : false)

cuda_available = add_languages('cuda', required : true)
cuda_dep = dependency('cuda', version : '>=10.1', required : true)
gtest_dep = dependency('gtest', main : true, version : '>=1.10', required : get_option('enable_testing'))

subdir('include/cusp')
subdir('kernels')
# subdir('lib')
subdir('test')
