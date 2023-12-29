import pylab as pl
import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

log1 = torch.load('/opt/data/private/csnl/experiment/msdsn_test1/loss_log.pt')
log2 = torch.load('/opt/data/private/csnl/experiment/msdsn_test2/loss_log.pt')
log3 = torch.load('/opt/data/private/csnl/experiment/msdsn1.0_csnl_4/loss_log.pt')
log4 = torch.load('/opt/data/private/csnl/experiment/msdsn_test3/loss_log.pt')
# log4 = torch.load('/opt/data/private/csnl/experiment/msdsn1.0_csnl_4/loss_log.pt')

# log1 = np.array(log1)
# log2 = np.array(log2)
fig = plt.figure()
loss1 = log1[:, 0]
loss2 = log2[:, 0]
loss3 = log3[:, 0]
loss4 = log4[:, 0]

epochs = list(range(600))
# epochs2 = list(range(230))
ax1 = fig.add_subplot(1, 1, 1)
# pl.plot(epochs, loss4, 'g', label=u'EDSR')
# pl.plot(epochs2, loss2, 'g', label=u'RCAN')
# pl.plot(epochs, loss3, 'r', label=u'RDN')
# pl.plot(epochs, loss4, 'b', label=u'EDSR')
# pl.plot(epochs, loss5, 'y', label=u'WDSR_A_X3')
# pl.plot(epochs, loss6, 'gold', label=u'WDSR_B')
# pl.plot(epochs, loss8, 'b', label=u'PANEDSR')
# pl.plot(epochs, loss9, 'r', label=u'EDSR+CSNL')
# pl.plot(epochs, loss10, 'k', label=u'EDSR+CSNLA')
# pl.plot(epochs, loss11, 'gold', label=u'TEST_scale2to1')
# pl.plot(epochs, loss12, 'g', label=u'EDSR+')
# pl.plot(epochs, loss13, 'k', label=u'EDSR_csnla1')
# pl.plot(epochs, loss14, 'b', label=u'EDSR_csnla_res')
# pl.plot(epochs, loss16, 'r', label=u'EDSR+600')
# pl.plot(epochs, loss17, 'b', label=u'EDSR_600_CSNLB')
# pl.plot(epochs, loss18, 'k', label=u'EDSR_600_CSNLA')
pl.plot(epochs, loss1, 'r', label=u'scale={3}')
pl.plot(epochs, loss2, 'g', label=u'scale={3,5}')
pl.plot(epochs, loss3, 'b', label=u'scale={3,5,7}')
pl.plot(epochs, loss4, 'k', label=u'scale={3,5,7,9}')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('compare loss ')
tx0 = 500
tx1 = 600
#设置想放大区域的横坐标范围
ty0 =7.25
ty1 =7.5

#设置想放大区域的纵坐标范围
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
pl.plot(sx, sy, "purple")
axins = inset_axes(ax1, width=1.5, height=1.5, loc='center')
#loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"


axins.plot(epochs, loss1, color='r', ls='-')
axins.plot(epochs, loss2, color='g', ls='-')
axins.plot(epochs, loss3, color='b', ls='-')
axins.plot(epochs, loss4, color='k', ls='-')
axins.axis([500, 600, 7.250, 7.450])
plt.savefig('/opt/data/private/csnl/experiment/Draw_pic/Compare_scale_loss.png')
# plt.show

# plt.close(fig)
# plt.savefig('/media/C/19-20CVPR代码论文/画图/test_loss_L1.pdf')
