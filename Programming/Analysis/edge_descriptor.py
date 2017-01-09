import cv2


img = cv2.imread("Images/Cropped images/Bordered with black color/Dataset_160_160/Colorful/Hard/IMG_0811.JPG", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imshow("as3", img)
cv2.waitKey(0)

def make_ages(min, max, tr):
    edges = cv2.Canny(gray, min, max, L2gradient=tr)
    cv2.imshow("as3", edges)
    cv2.waitKey(0)

#make_ages(100, 150, False)
#make_ages(100, 150, True)
#make_ages(150, 200, False)
#make_ages(150, 200, True)
#make_ages(100, 255, True)
make_ages(50, 255, True)
#make_ages(50, 255, False)
