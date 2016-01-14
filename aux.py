#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Image
import numpy as np
import os
import pycuda.autoinit
import pycuda.compiler
import pycuda.driver as cuda
import sys

__root_folder = './'

def makedir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def set_root_folder(folder):
    global __root_folder
    
    __root_folder = folder + '/'
    makedir(__root_folder)

def get_root_folder():
    global __root_folder
    
    return __root_folder

def save_as_image(data, folder, file_name):
    global __root_folder

    folder = __root_folder + folder
    makedir(folder)
    image = Image.fromarray(np.array(data).astype(np.uint8) * 255)
    image.save(folder + '/' + file_name)


class CudaModule(pycuda.compiler.SourceModule):
    
    __base_dir = os.path.dirname(os.path.realpath(__file__)) 
    
    def __init__(self, cuda_file, extra=''):
        with open(self.__base_dir + "/" + cuda_file, 'r') as cuda_file:
            cuda_code = "".join(cuda_file.readlines()) % extra
        
        try:
            super(CudaModule, self).__init__(cuda_code)
        except cuda.CompileError as e:
            sys.exit("CUDA: Not able to compile '%s'!\n'%s'" %
                     (cuda_file, e))
    
    def get_function(self, function_name):
        try:
            return super(CudaModule, self).get_function(function_name)
        except cuda.Error:
            sys.exit("CUDA: Not able to get function '%s'!" %
                     function_name)
