import pylab as pl
import matplotlib
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

log1 = torch.load('/opt/data/private/csnl/experiment/msdsn_test1/psnr_log.pt')
log2 = torch.load('/opt/data/private/csnl/experiment/msdsn_test2/psnr_log.pt')
log3 = torch.load('/opt/data/private/csnl/experiment/msdsn1.0_csnl_4/psnr_log.pt')
log4 = torch.load('/opt/data/private/csnl/experiment/msdsn_test3/psnr_log.pt')
# log4 = torch.load('/media/C/19-20CVPR代码论文/画图/EDSR/psnr_log.pt')

# log1 = np.array(log1)
# log2 = np.array(log2)
fig = plt.figure()
loss1 = log1[:, 0]
loss2 = log2[:, 0]
loss3 = log3[:, 0]
loss4 = log4[:, 0]
print(loss1.shape)
# print(loss2.shape)
# print(loss3.shape)
epochs = list(range(60))
# epochs2 = list(range(230))
ax1 = fig.add_subplot(1, 1, 1)
pl.plot(epochs, loss1, 'k', label=u'scale={3}')
pl.plot(epochs, loss2, 'g', label=u'scale={3,5}')
pl.plot(epochs, loss3, 'b', label=u'scale={3,5,7}')
pl.plot(epochs, loss4, 'k', label=u'scale={3,5,7,9}')
plt.legend()
plt.xlabel('Epochs/10')
plt.ylabel('PSNR')
plt.title(' Compare psnr ')
tx0 = 50
tx1 = 60
#设置想放大区域的横坐标范围
ty0 = 27
ty1 = 27.200
#设置想放大区域的纵坐标范围
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
pl.plot(sx, sy, "purple")
axins = inset_axes(ax1, width=1.5, height=1.5, loc='center')
#loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"
# axins.plot(epochs2, loss2, color='g', ls='-')
# axins.plot(epochs, loss3, color='r', ls='-')
# axins.plot(epochs, loss4, color='r', ls='-')
# axins.plot(epochs, loss5, color='y', ls='-')
# axins.plot(epochs, loss1, color='g', ls='-')
# axins.plot(epochs, loss6, color='gold', ls='-')
# axins.plot(epochs, loss4, color='r', ls='-')
# axins.plot(epochs, loss8, color='b', ls='-')
# axins.plot(epochs, loss12, color='g', ls='-')
# axins.plot(epochs, loss13, color='k', ls='-')
# axins.plot(epochs, loss14, color='b', ls='-')
# axins.plot(epochs, loss15, color='r', ls='-')
# axins.plot(epochs, loss16, color='r', ls='-')
# axins.plot(epochs, loss17, color='b', ls='-')
axins.plot(epochs, loss1, color='r', ls='-')
axins.plot(epochs, loss2, color='g', ls='-')
axins.plot(epochs, loss3, color='b', ls='-')
axins.plot(epochs, loss4, color='k', ls='-')
# axins.plot(epochs, loss11, color='gold', ls='-')
axins.axis([50,60,27,27.200])
plt.savefig('/opt/data/private/csnl/experiment/Draw_pic/Compare_scale_psnr.png')
plt.show
#pl.show()也可以


# plt.close(fig)
# plt.savefig('/media/C/19-20CVPR代码论文/画图/test_loss_L1.pdf')
