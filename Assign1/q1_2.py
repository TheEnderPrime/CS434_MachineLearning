#!/usr/bin/python

#python q1_2.py housing_train.txt

import sys
import csv
import numpy as np

X = []
Y = []

with open(sys.argv[1]) as housing_train:
    csvReader = csv.reader(housing_train)
    for row in csvReader:
        print(row)

