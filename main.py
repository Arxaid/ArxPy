# This file is part of ArxPy statistics library.
#
# Copyright (c) 2022 Vladislav Sosedov.

import ArxPy as arx

if __name__ == '__main__':

    datasheet1 = arx.DataLoading_txt('datasheets/VAR1.txt')
    datasheet2 = arx.DataLoading_txt('datasheets/VAR3.txt')
    datasheet3 = arx.DataLoading_txt('datasheets/VAR6.txt')
    datasheet4 = arx.DataLoading_txt('datasheets/VAR9.txt')
    alpha = 0.05

    print('\nStudent t-test\nVAR1 and VAR3')
    arx.TTest(datasheet1, datasheet2, alpha=alpha)
    print('\nStudent t-test\nVAR6 and VAR9')
    arx.TTest(datasheet4, datasheet3, alpha=alpha)