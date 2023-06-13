# import the necessary packages
from preprocessing import SimplePreprocessor # Import modul SimplePreprocessor
from datasets import simpledatasetloader #Import modul simpledatasetloader
import matplotlib.pyplot as plt
import cv2
import pickle #Thư viện này để đọc file model

# Khởi tạo danh sách nhãn
classLabels = ["butterfly", "cat", "chitken", "cow", "dog", "elephant", "horse", "panda", "sheep", "spider", "squirrel"]

print("[INFO] Đang nạp ảnh để cho bộ phân lớp dự đoán...")

# Thiết lập kích thước ảnh 32 x 32
sp = SimplePreprocessor(32, 32)

# Tạo bộ nạp dữ liệu
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp])

# Nạp dữ liệu file ảnh và lưu dưới dạng mảng
data, _ = sdl.load(["Image\\image1233.jpg"])

#Thay đổi cách biểu diễn mảng dữ liệu ảnh
data = data.reshape((data.shape[0], 3072))

# Nạp model SVM đã train
print("[INFO] Nạp model SVM ...")
model = pickle.load(open('svm.model', 'rb'))

# Dự đoán
print("[INFO] Thực hiện dự đoán ảnh để phân lớp...")
preds = model.predict(data) # Trả về danh sách nhãn dự đoán: 0->cat, 1->dog, 2->Panda

# Đọc file ảnh
image = cv2.imread("Image\\h2.jpg")
fig = plt.figure(figsize=(4, 4))
(ax1) = fig.subplots(1, 1)
ax1.imshow(image)
ax1.set_title(classLabels[preds[0]])
plt.show()
