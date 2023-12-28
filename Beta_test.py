# beta distribution
import numpy as np
from scipy.stats import beta, norm
import matplotlib.pyplot as plt
from torch.distributions import Normal, Beta

norm_dis = Normal(2, 1)
beta_dis = Beta(2.2, 2.2)
yn = norm_dis.rsample()
yb = beta_dis.rsample()
yb2 = beta_dis.sample()
logp_yn = norm_dis.log_prob(yn)
logp_yb = beta_dis.log_prob(yb)
logp_yb2 = beta_dis.log_prob(yb2)
p_yn = logp_yn.exp()
p_yb = logp_yb.exp()
p_yb2 = logp_yb2.exp()
print('(yn: %.4f, yb: %.4f, yb2: %.4f)' % (yn, yb, yb2))
print('(logp_yn: %.4f, logp_yb: %.4f, logp_yb2: %.4f)' % (logp_yn, logp_yb, logp_yb2))
print('(p_yn: %.4f, p_yb: %.4f, p_yb2: %.4f)' % (p_yn, p_yb, p_yb2))

x = np.linspace(0, 1, 100)
y1 = beta.pdf(x, 2.2, 2.2)
plt.subplot(2, 2, 1)
plt.plot(x, y1)
plt.subplot(2, 2, 2)
y2 = norm.pdf(x, 2, 1)
plt.plot(x, y2)
plt.show()