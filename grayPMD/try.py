from matplotlib import pyplot as plt

img1 = plt.imread("C:\\Users\\韩宸宇\\Desktop\\1.png")
img2 = plt.imread("C:\\Users\\韩宸宇\\Desktop\\2.png")
error_xm = abs(img1 - img2)
plt.imshow(error_xm, cmap='gray')  # 误差值
plt.axis('off')
plt.figure()
plt.show()

sum_error = 0
for i in range(230):
    for j in range(230):
        sum_error = sum_error + error_xm[i, j]
mae = sum_error / 480 / 480
print(mae)
