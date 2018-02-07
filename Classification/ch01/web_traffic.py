import scipy as sp
data = sp.genfromtxt("data/web_traffic.tsv", delimiter = "\t")
print(data[:10])
print(data.shape)
x = data[:, 0]
y = data[:, 1]
print(sp.sum(sp.isnan(y)))
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]
print(x.shape, y.shape)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.title("Web traffic over last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i'%w for w in range(10)])
plt.autoscale(tight = True)
plt.grid()
#plt.show()

def error(f, x, y):
       return sp.sum((f(x) - y) ** 2)

#d = 1
fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
#print("Model parameters: %s" %  fp1)
#print(residuals)
f1 = sp.poly1d(fp1)
print("f1 Error", error(f1, x, y))

fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting
#plt.plot(fx, f1(fx), linewidth=4)
#plt.legend(["d=%i" % f1.order], loc="upper left")
#plt.show()

#d = 2
fp2 = sp.polyfit(x, y, 2)
#print("Model parameters: %s" %  fp2)
f2 = sp.poly1d(fp2)
print("f2 Error", error(f2, x, y))

fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting
#plt.plot(fx, f2(fx), linewidth=4)
#plt.legend(["d=%i" % f2.order], loc="upper left")
#plt.show()

#d = 100
fp100 = sp.polyfit(x, y, 100)
#print("Model parameters: %s" %  fp100)
f100 = sp.poly1d(fp100)
print("f100 Error", error(f100, x, y))

fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting
#plt.plot(fx, f100(fx), linewidth=4)
#plt.legend(["d=%i" % f1.order, "d=%i" % f2.order, "d=%i" % f100.order], loc="upper left")
#plt.show()

inflection = int(3.5 * 7 * 24) # calculate the inflection point in hours
xa = x[:inflection] # data before the inflection point
ya = y[:inflection]
xb = x[inflection:] # data after
yb = y[inflection:]
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))
fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)
#print("Error inflection=%f" % (fa + fb_error))
fx = sp.linspace(0,xa[-1], 1000)
plt.plot(fx, fa(fx), linewidth=4)
fx = sp.linspace(xa[-1],xb[-1], 1000)
plt.plot(fx, fb(fx), linewidth=4)
plt.show()

from scipy.optimize import fsolve
print(fb)
reached_max = fsolve(fb - 10000, 800) / (7 * 24)
print("100,000 hits/hour expected at week %f" % reached_max[0])
