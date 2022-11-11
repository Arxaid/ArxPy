# This file is part of ArxPy statistics library.
#
# Copyright (c) 2022 Vladislav Sosedov.

import ArxPy as arx

if __name__ == '__main__':

    datasheet1 = arx.DataLoading_txt('datasheets/VAR1.txt')
    datasheet2 = arx.DataLoading_txt('datasheets/VAR3.txt')
    datasheet3 = arx.DataLoading_txt('datasheets/VAR6.txt')
    datasheet4 = arx.DataLoading_txt('datasheets/VAR9.txt')

    print('\nMann-Whitney U-test\nVAR1 and VAR3')
    arx.MannWhitneyTest(datasheet1, datasheet2, True)
    print('\nMann-Whitney U-test\nVAR6 and VAR9')
    arx.MannWhitneyTest(datasheet3, datasheet4, True)