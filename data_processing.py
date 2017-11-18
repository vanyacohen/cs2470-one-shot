import cv2, os, math as m, numpy as np, random as rand

def affine_transformation(img, theta, rho_x, rho_y, s_x, s_y, t_x, t_y):
	out = img
	rows, cols = out.shape
	if rand.uniform(0,2) > 1:
		a = m.cos(theta)
		b = m.sin(theta)
		M = np.float32([[a, b, ((1 - a) * 53) - (b * 53)],[-b, a, (b * 53) + ((1 - a) * 53)]])
		out = cv2.warpAffine(img,M,(cols,rows),borderValue=1.0)
	if rand.uniform(0,2) > 1:
		M = np.float32([[1, rho_x, 0],[rho_y, 1, 0]])
		out = cv2.warpAffine(img,M,(cols,rows),borderValue=1.0)
	if rand.uniform(0,2) > 1:
		out = cv2.resize(out,None,fx=s_x, fy=s_y, interpolation = cv2.INTER_CUBIC)
		y_len, x_len = out.shape
		if x_len > 105:
			x_dif = x_len - 105
			out = out[:,m.ceil(x_dif / 2):m.ceil(x_len - x_dif / 2)]
		else:
			x_dif = 105 - x_len
			temp = np.ones((y_len,105))
			temp[:,m.ceil(x_dif / 2):m.ceil(105 - x_dif / 2)] = out
			out = temp
		if y_len > 105:
			y_dif = y_len - 105
			out = out[m.ceil(y_dif / 2):m.ceil(y_len - y_dif / 2),:]
		else:
			y_dif = 105 - y_len
			temp = np.ones((105,105))
			temp[m.ceil(y_dif / 2):m.ceil(105 - y_dif / 2),:] = out
			out = temp
		if out.shape != (105, 105):
			print(out.shape)
	if rand.uniform(0,2) > 1:
		M = np.float32([[1,0,t_x],[0,1,t_y]])
		out = cv2.warpAffine(out,M,(cols,rows),borderValue=1.0)
	return out

def get_data(num, trans_num, step):
	pairs = []
	labels = []
	if step == 'train':
		base = 'images_background'
		alpha_lb = 0
		alpha_ub = 30
		img_lb = 0
		img_ub = 12
	elif step == 'test':
		base = 'images_evaluation'
		alpha_lb = 0
		alpha_ub = 10
		img_lb = 12
		img_ub = 16
	else:
		base = 'images_evaluation'
		alpha_lb = 10
		alpha_ub = 20
		img_lb = 16
		img_ub = 20
	alphabets = os.listdir('images_background')
	for i in range(num):
		alph1 = os.path.join(base, alphabets[rand.randint(alpha_lb,alpha_ub - 1)])
		characters1 = os.listdir(alph1)
		char1 = os.path.join(alph1, characters1[rand.randint(0,len(characters1) - 1)])
		images1 = os.listdir(char1)
		img1 = os.path.join(char1, images1[rand.randint(img_lb,img_ub - 1)])
		x = cv2.imread(img1,0) / 255
		if rand.uniform(0,2) > 1:
			label = 1
			img2 = os.path.join(char1, images1[rand.randint(img_lb,img_ub - 1)])
			while img2 == img1:
				img2 = os.path.join(char1, images1[rand.randint(img_lb,img_ub - 1)])
			y = cv2.imread(img2,0) / 255
		else:
			label = 0
			alph2 = os.path.join(base, alphabets[rand.randint(alpha_lb,alpha_ub - 1)])
			characters2 = os.listdir(alph2)
			char2 = os.path.join(alph2, characters2[rand.randint(0,len(characters2) - 1)])
			while char2 == char1:
				char2 = os.path.join(alph2, characters2[rand.randint(0,len(characters2) - 1)])
			images2 = os.listdir(char2)
			img2 = os.path.join(char2, images2[rand.randint(img_lb,img_ub - 1)])
			y = cv2.imread(img2,0) / 255
		pairs.append([x, y])
		labels.append(label)
		for j in range(trans_num):
			a_x = affine_transformation(x, rand.uniform(-10,10), rand.uniform(-0.3,0.3), rand.uniform(-0.3,0.3), rand.uniform(0.8,1.2), rand.uniform(0.8,1.2), rand.uniform(-2,2), rand.uniform(-2,2))
			a_y = affine_transformation(y, rand.uniform(-10,10), rand.uniform(-0.3,0.3), rand.uniform(-0.3,0.3), rand.uniform(0.8,1.2), rand.uniform(0.8,1.2), rand.uniform(-2,2), rand.uniform(-2,2))
			pairs.append([a_x, a_y])
			labels.append(label)
	return (pairs, labels)
