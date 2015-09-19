
import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM

class DrillBit(object):
  """
  Generates simulated drill-bit usage data: thrust and torque.
  """
  def __init__(self,):
    # Keeps track of how many times the drill-bit has been used:
    self.usage_count = 0
    self.x = np.linspace(0.1, 4, 5000)
    self.k = -5
    #k = np.exp(-2)

  def update(self):
    self.k = self.k + 3

  def get_kfactor(self):
    kfactor = np.exp(self.k)
    return kfactor

  def get_time(self):
    x = -1*self.x
    return x

  def get_thrust(self):
    x = self.x
    k = self.get_kfactor()
    mu = 0.0 
    #print 'mu=',mu
    sigma = 0.5 + k
    pdfa = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
    return pdfa


  def get_torque(self):
    x = self.x
    k = self.get_kfactor()
    mu = 0.5 
    #print 'mu=',mu
    sigma = 0.5 + 2*k
    pdfb = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
    return pdfb

  def drill(self):
    """
    After drilling a hole, return simulated thrust and torque data.
    """
    pdf_thrust = self.get_thrust()
    pdf_torque = self.get_torque()
    self.update()
    xdata = (pdf_thrust, pdf_torque)
    return xdata

if __name__ == '__main__':
  da = DrillBit()
  xt = da.get_time()
  (xthrust, xtorque) = da.drill()
  (xthrust, xtorque) = da.drill()
  
  X = np.column_stack([xthrust, xtorque])

  # Train HMM!
  # make an HMM instance and execute fit
  #model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000).fit(X)
  model = GaussianHMM(n_components=10, covariance_type="diag", n_iter=1000).fit(X)

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
