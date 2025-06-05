#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   multi_processing.py    

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/15 1:15   lintean      1.0         None
'''

from multiprocessing import Process
from model import main
from shap_wrapper import multi_run
import utils as util

if __name__ == "__main__":
    multiple = 1
    process = []
    path = "../KUL_single_single3"
    names = ['S' + str(i+1) for i in range(0, 16)]
    for name in names:
        p = Process(target=multi_run, args=(name, path, 10, "./32c1s"))
        p.start()
        process.append(p)
        util.monitor(process, multiple, 60)

