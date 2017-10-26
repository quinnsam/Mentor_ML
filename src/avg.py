#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import walk
import re
from heapq import nsmallest

def get_avg(l):
     m = re.search(r"\[([0-9.]+)\]", l)
     if m:
         #print m.group(1)
         return float(m.group(1))
     else:
         return 0.0


if __name__ == "__main__":
    avg = {}
    f = []
    for (dirpath, dirnames, filenames) in walk("./results"):
        f.extend(filenames)
        break

    for net in f:
        total = 0.0
        with open('./results/' + net, 'rb') as tmp_net:
            for l in tmp_net:
                total += get_avg(l)

        avg[net] = total


    print "Top Ten Neural nets"

    for name, score in nsmallest(10, avg.iteritems(), key=lambda (k, v): (-v, k)):
        print '%s \t %s' % (name, score)
