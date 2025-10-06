from PIL import Image
img1= Image.open('/home/sahand/datasets/vindr_cxr/train2017/0a2d01ecb9e01cf972c1e1d31ccacb98.png')
img2= Image.open('/home/sahand/datasets/vindr_cxr/train2017/fee46e386a84e134836e01d0b9a38154.png')

print('Original mode1:', img1.mode)
img1.save('pilcheck_original1.png')
print('Original mode2:', img2.mode)
img2.save('pilcheck_original2.png')