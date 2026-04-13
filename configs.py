# 所有超参数集中管理，便于修改

IMG_SIZE = 28          # 图像尺寸（MNIST 原生 28x28）
IMG_CH = 1             # 图像通道数（灰度图）
BATCH_SIZE = 128       # 训练批次大小
N_CLASSES = 10         # 类别数量（0~9）

nrows = 10             # 扩散过程展示的行数
ncols = 15             # 扩散过程展示的列数
T = nrows * ncols      # 扩散总步数（150 步）