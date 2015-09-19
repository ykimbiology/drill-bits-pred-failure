
"""
This example is based on the following paper:
###


1) Generate simulated drill-bits measurement data.
2) Run cross-validations to detemrine optimal number of hidden states.
3) Plot time vs drill-bits measurement data for N drills.
 
"""
import random
import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM

class DrillBit(object):
  """
  Generates simulated drill-bit thrust and torque data.
  """
  def __init__(self,):
    # Keeps track of how many times the drill-bit has been used:
    self.usage_count = 0
    self.x = np.linspace(0.1, 4, 1000)
    self.k = -5
    self.j = 0
    self.thrust = []
    self.torque = []
    #k = np.exp(-2)

  def update(self):
    #self.k = self.k + 3
    self.k = self.k + 1.2
    self.j = self.j + 0.1

  def get_kfactor(self):
    kfactor = np.exp(self.k)
    return kfactor

  def get_time(self):
    x = -1*self.x
    return x

  def get_thrust(self):
    x = self.x
    k = self.get_kfactor()
    j = self.j
    mu = 0.0 
    #print 'mu=',mu
    sigma = 0.5 + j
    #sigma = 0.5
    pdfa = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
    pdfa = (1.0 + k)*pdfa
    xmax = np.max(pdfa)
    v = np.array([0.1*random.random()*xmax for i in range(len(pdfa))])
    pdfa = pdfa + v
    pdfa = list(pdfa)
    pdfa.reverse()
    self.thrust.extend(pdfa)
    return np.array(pdfa)


  def get_torque(self):
    x = self.x
    k = self.get_kfactor()
    j = self.j
    print 'k=',k
    mu = 0.5 
    #print 'mu=',mu
    sigma = 0.5 + j
    #sigma = 0.5
    pdfb = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
    pdfb = (1.0+2*k)*pdfb
    xmax = np.max(pdfb)
    v = np.array([0.1*random.random()*xmax for i in range(len(pdfb))])
    pdfb = pdfb + v
    pdfb = list(pdfb)
    pdfb.reverse()
    self.torque.extend(pdfb)
    return np.array(pdfb)

  def drill(self):
    """
    After drilling a hole, return simulated thrust and torque data.
    """
    pdf_thrust = self.get_thrust()
    pdf_torque = self.get_torque()
    self.update()
    xdata = (pdf_thrust, pdf_torque)
    return xdata

def plot_thrust_vs_torque():
  pass

def plot_time_vs_x(): 
  pass

if __name__ == '__main__':
  da = DrillBit()
  xt = da.get_time()
  (xthrust, xtorque) = da.drill()
  (xthrust, xtorque) = da.drill()
  (xthrust, xtorque) = da.drill()
  (xthrust, xtorque) = da.drill()
  (xthrust, xtorque) = da.drill()
  (xthrust, xtorque) = da.drill()
  
  X = np.column_stack([xthrust, xtorque])

  # Train HMM!
  # make an HMM instance and execute fit
  model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000).fit(X)
  #model = GaussianHMM(n_components=10, covariance_type="diag", n_iter=1000).fit(X)

  # predict the optimal sequence of internal hidden state
  hidden_states = model.predict(X)



  fig = plt.figure()
  ax = fig.add_subplot(111)

  for i in range(model.n_components):
    # use fancy indexing to plot data in each state
    idx = (hidden_states == i)
    ax.plot(xthrust[idx], xtorque[idx], 'o', label="%dth hidden state" % i)

  #plt.plot(xt, xthrust, linewidth=2, color='r')
  #plt.plot(xt, xtorque, linewidth=2, color='b')
  #plt.plot(xthrust, xtorque, linewidth=2, color='b')

  #plt.axis('tight')
  plt.show()
