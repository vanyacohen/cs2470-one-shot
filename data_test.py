import data_processing as dp
import cv2

pairs, labels = dp.get_data(10, 4, True)

for pair in pairs:
	cv2.imshow('image', pair[0])
	cv2.waitKey(0)
	cv2.destroyAllWindows()
print(len(pairs))
print(len(labels))
