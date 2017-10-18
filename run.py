#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src import mnist_loader
from src import network
import os.path
import pickle
import argparse

parser = argparse.ArgumentParser(description='Neural network wrapper')
parser.add_argument('--load_file', action="store", dest="load", help='Use a saved Network')
parser.add_argument('--save_file', action="store", dest="save", help='Save a Network')
parser.add_argument('-n', action="store", dest="network", nargs='+', type=int, default=[784,30,10], help='Description of network e.g. \'[784,30,10]\'')
parser.add_argument('-e', action="store", dest="epoch", type=int, default=3, help='Number of epochs to use')
parser.add_argument('-m', action="store", dest="mini", type=int, default=10, help='Mini-batch size')
parser.add_argument('-l', action="store", dest="learn", type=float, default=3.0, help='Learning rate to implement')
parser.add_argument('-p', action="store_true", dest="percentage", default=False, help='Show each digits percentage')
parser.add_argument('-f', action="store_true", dest="failed", default=False, help='Show failed digits and what the net thought they where')

results = parser.parse_args()

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

if results.load:
    if not os.path.isfile(results.load):
        print "No save found."
        exit()

    print "Save found using previous data"
    with open(results.load, 'rb') as save:
        net = pickle.load(save)
else:
    net = network.Network(results.network)
    net.SGD(training_data, results.epoch, results.mini, results.learn)



if results.save:
    with open(results.save, 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)

if results.percentage:
    net.percentage(test_data)
if results.failed:
    net.show_failed(test_data)
