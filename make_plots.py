# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

## data
weight = np.array([0.29, 0.36, 0.50, 0.30, 0.29, 0.39])
voltage = np.array([27.6, 19.2, 16.4, 38.0, 34.7, 26.0])
vi = 0.95E-3
bias = 100E3
l = 10E6

## calculate resistance
resistance = bias*l*voltage*1E-6/ (l*vi - l*voltage*1E-6 -bias*voltage*1E-6)

## resistance plot
plt.plot(weight, resistance, 'ok',ms=4)
plt.xlabel("Weight [g]")
plt.ylabel("Resistance [Ohms]")
plt.plot(weight[0:2], resistance[0:2], 'ro', ms=4)
plt.show()

## voltage plot
plt.figure()
plt.plot(weight, voltage, 'bo', ms=4)
plt.xlabel("Weight [g]")
plt.ylabel("Voltage [uV]")
## color some bad junctions
plt.plot(weight[0:2], voltage[0:2], 'ro', ms=4)
plt.show()

