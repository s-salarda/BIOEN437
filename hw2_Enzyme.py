import tellurium as te
import roadrunner
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

def mm(S, Vm, Km):
    return Vm*S/(Km+S)

r = te.loada("""
E + $S -> ES; k1*E*S - km1*ES
J2: ES -> E + P; k2*ES        

k1 = 40; km1 = 2; k2 = 5
E = 2
S = 0.1
P = 0
ES = 0
""")

x = []
y = []

SList = np.arange (0.1, 10, 0.2)
for SValue in SList:
    r.S = SValue
    m = r.simulate(0, 0.4, 2)
    v = r.J2
    x.append(SValue)
    y.append(v)
    
plt.plot(x,y)

popt, pcov = curve_fit(mm, x, y, p0=[1,1], bounds=(0,20))
print (popt)

# Transform the irreversible Michaelis Menten Equation into a straight line
# Lineweaver Burk Plot
x_inv = [1/i for i in x]
y_inv = [1/j for j in y]
plt.figure()
plt.xlabel('1 / [S]')
plt.ylabel('1 / v')
plt.title('Lineweaver-Burk Linear Regression')
plt.plot(x_inv, y_inv)


# What are the intercept values
result = linregress(x_inv,y_inv)
slope = result.slope
intercept = result.intercept
print("intercept=", intercept)

# Vmax and Kmax using the plot 
Vmax_plot = 1/intercept
Km_plot = slope / intercept
print("Vmax_plot",Vmax_plot)
print("Km_plot",Km_plot)
