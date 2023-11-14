from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,recall_score)

# Định nghĩa lớp cây quyết định
class ID3:
    def __init__(self): # hàm tạo, gán cây quyết định ban đầu là rỗng
        self.tree = None
    
    def entropy(self, y): #  Định nghĩa hàm tính entropy
        classes, counts = np.unique(y, return_counts=True) # trả về mảng chứa các giá trị không bị trùng lặp và counts: mảng chứa số lần xuất hiện tương ứng 
        probabilities = counts / len(y) # Tỷ lệ xác suất của mỗi lớp trong tập dữ liệu
        entropy = np.sum(-probabilities * np.log(probabilities)) # Tính entropy
        return entropy
    
    def information_gain(self, X, y, feature):
        unique_values = np.unique(X[:, feature]) # trả về mảng chứa các giá trị không bị trùng lặp
        entropy_parent = self.entropy(y) # Tính entropy của tập dữ liệu gốc (y) bằng cách gọi hàm entropy(y) ở trên
        entropy_children = 0 # Khởi tạo giá trị entropy của các tập dữ liệu con bằng 0
        
        for value in unique_values:
            indices = np.where(X[:, feature] == value) # tìm ra các chỉ số của các mẫu trong tập dữ liệu gốc (X)
            entropy = self.entropy(y[indices]) # Tính entropy của tập dữ liệu con tương ứng với giá trị duy nhất hiện đang được xét.
            entropy_children += len(indices[0]) / len(X) * entropy # Tính entropy của tập dữ liệu con
        
        information_gain = entropy_parent - entropy_children # Tính G(x,S)
        return information_gain
    
    def choose_best_feature(self, X, y):
        num_features = X.shape[1] # số lượng features trong tập dữ liệu (số lượng cột) 
        # X.shape[1] đề cập đến số lượng cột (features).

        best_feature = None # Gán nhãn phổ biến nhất ban đầu là Rỗng
        max_information_gain = -1 # Gán G(x,S) lớn nhất khởi tạo là -1
        
        for feature in range(num_features):
            information_gain = self.information_gain(X, y, feature) 
            # Tính G(x,S) cho từng thuộc tính, nếu giá trị này lớn hơn  max_information_gain  
            # thì gán max_information_gain  bằng nó, gán biến best_feature = thuộc tính đang xét
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_feature = feature
        
        # Sau khi lặp qua tất cả các thuộc tính, ta trả về information_gain lớn nhất tương ứng với thuộc tính phổ biến nhất
        return best_feature
    
    def fit(self, X, y): # Hàm huấn luyện cây quyết định
        self.tree = self.build_tree(X, y)
    
    def build_tree(self, X, y):
        # 3 điều kiện dừng: Hết thuộc tính để chia, nút con rỗng , tất cả các mẫu trong nút con đều cùng 1 lớp 

        if len(np.unique(y)) == 1:  # Tất cả các mẫu cùng thuộc một lớp
            return np.unique(y)[0]
        
        if X.shape[1] == 0:  # Hết thuộc tính để xét, nút con rỗng
            return np.bincount(y).argmax() # node sẽ được gán nhãn là nhãn phổ biến nhất, hay có số lượng nhiều nhẩt
        
        best_feature = self.choose_best_feature(X, y) # chọn thuộc tính phổ biến nhất 
        tree = {best_feature: {}} # Tạo một nút trên cây quyết định với feature được chọn làm nút cha và một dictionary rỗng để lưu các giá trị con của nút này.
        unique_values = np.unique(X[:, best_feature]) #  Lấy ra danh sách các giá trị duy nhất của feature
        #Dictionary lưu trữ các cặp key-value không có thứ tự và không thể trùng lặp
        for value in unique_values:
            indices = np.where(X[:, best_feature] == value) # Tìm ra các mẫu trong tập dữ liệu gốc X mà giá trị của feature bằng giá trị đang xét.
            sub_X = X[indices] # Lấy ra các mẫu tương ứng với giá trị duy nhất đang xét.
            sub_y = y[indices] #  Lấy ra nhãn tương ứng với các mẫu đã được chọn.
            
            tree[best_feature][value] = self.build_tree(sub_X, sub_y) # thực hiện đệ quy, gán kết quả vào nút con với giá trị value làm key.
        
        # Trả về cây quyết định đã được xây dựng.
        return tree
    
    def predict_sample(self, sample, tree):
        if not isinstance(tree, dict):  # Kiểm tra xem nút hiện tại của cây có phải là một lá hay không 
            # bằng cách kiểm tra Nếu nút hiện tại không phải là một dictionary, tức là nó là một giá trị lá
            return tree
        
        feature = list(tree.keys())[0] # Lấy thuộc tính đầu tiên mà nút hiện tại sử dụng để chia tập dữ liệu.
        value = sample[feature] # Lấy giá trị của thuộc tính đó từ mẫu đầu vào.

        if value not in tree[feature]:  # Giá trị không tồn tại trong cây
            return np.bincount(y_train.astype(int)).argmax() # Đặt giá trị dự đoán bằng lớp phổ biến nhất

        
        subtree = tree[feature][value] # Nếu giá trị của thuộc tính tồn tại trong cây, lấy nút con tương ứng với giá trị đó.

        return self.predict_sample(sample, subtree)
    
    def predict(self, X):
        predictions = [] #  Khởi tạo một danh sách rỗng để lưu các dự đoán.
        
        for sample in X:
            prediction = self.predict_sample(sample, self.tree)
            predictions.append(prediction)
        
        return np.array(predictions) # Trả về các dự đoán dưới dạng một mảng. Mảng này chứa lớp dự đoán tương ứng với từng mẫu trong tập X.

data= pd.read_csv('glass.csv')
data = data.values
# Chuẩn bị dữ liệu
X = data[:, :-1]
y = data[:, -1]

# Chia thành train và test
dt_Train, dt_Test = train_test_split(data, test_size=0.1, shuffle=True)

X_train = dt_Train[:, :9]
y_train = dt_Train[:, 9]
X_test = dt_Test[:, :9]
y_test = dt_Test[:, 9]

# tính tỷ lệ đúng của y dự đoán
def accuracy(y, y_pred):
    count = 0
    for i in range(0, len(y)):
        if (y[i] == y_pred[i]):
            count += 1
    return count/len(y)

# hàm trả về mô hình có độ chính xác theo thuật toán ID3
def the_best_id3(dt_Train):
    k = 3
    kf = KFold(n_splits=k, random_state=None) 
    max = -999999
    for train_index, validation_index in kf.split(dt_Train): #vòng lặp for để chia dữ liệu thành bộ train và bộ validation.
        #Đặt các biến "x_train, x_validation, y_train, y_validation" để lưu trữ dữ liệu train và validation tương ứng.
        X_train, X_validation = dt_Train[train_index, :9], dt_Train[validation_index, :9]
        y_train, y_validation = dt_Train[train_index, 9], dt_Train[validation_index, 9]
        id3 = ID3()
        id3.fit(X_train, y_train) #sử dụng fit để huấn luyên mô hình X_train, y_train
        y_train_pred = id3.predict(X_train)# dự đoán nhãn cho dư liệu train
        y_validation_pred = id3.predict(X_validation)# dự đoán nhãn cho dư liệu validation
        y_train = np.array(y_train)# Chuyển đổi các train từ kiểu dữ liệu list sang kiểu dữ liệu array.
        y_validation = np.array(y_validation)# Chuyển đổi các nhãn từ kiểu dữ liệu list sang kiểu dữ liệu array.
        sum_accuracy = accuracy(y_train, y_train_pred) + accuracy(y_validation, y_validation_pred)# Tính tổng tỷ lệ dự đoán đúng cho cả dữ liệu train và validation.
        if (sum_accuracy > max):
            max = sum_accuracy
            id3_best = id3
    return id3_best #  lưu trữ mô hình cây quyết định vào biến "id3_best".
# # Huấn luyện mô hình
id3 = the_best_id3(dt_Train)
y_pred_id3 = id3.predict(X_test)

form = Tk() # Khởi tạo đối tượng Tk()
form.title("Dự đoán loại thủy tinh:") # title của form
form.geometry("600x550") # kích thước của form

# Khai báo label
lable_ten = Label(form, text="Nhập thông tin cho loại Thủy tinh:", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row=1, column=1, padx=40, pady=10) # xét vị trí cho label

lable_RI = Label(form, text = "RI")
lable_RI.grid(row = 2, column = 1, pady = 10, padx = 40 )
textbox_RI = Entry(form)
textbox_RI.grid(row = 2, column = 2)

lable_Na = Label(form, text = "Na")
lable_Na.grid(row = 3, column = 1, pady = 10, padx = 40 )
textbox_Na = Entry(form)
textbox_Na.grid(row = 3, column = 2)

lable_Mg = Label(form, text = "Mg")
lable_Mg.grid(row = 4, column = 1, pady = 10, padx = 40 )
textbox_Mg = Entry(form)
textbox_Mg.grid(row = 4, column = 2)

lable_Al = Label(form, text = "Al")
lable_Al.grid(row = 5, column = 1, pady = 10, padx = 40 )
textbox_Al = Entry(form)
textbox_Al.grid(row = 5, column = 2)

lable_Si = Label(form, text = "Si")
lable_Si.grid(row = 6, column = 1, pady = 10, padx = 40 )
textbox_Si = Entry(form)
textbox_Si.grid(row = 6, column = 2)

lable_K = Label(form, text = "K")
lable_K.grid(row = 7, column = 1, pady = 10, padx = 40 )
textbox_K = Entry(form)
textbox_K.grid(row = 7, column = 2)

lable_Ca = Label(form, text = "Ca")
lable_Ca.grid(row = 8, column = 1, pady = 10, padx = 40 )
textbox_Ca = Entry(form)
textbox_Ca.grid(row = 8, column = 2)

lable_Ba = Label(form, text = "Ba")
lable_Ba.grid(row = 9, column = 1, pady = 10, padx = 40 )
textbox_Ba = Entry(form)
textbox_Ba.grid(row = 9, column = 2)

lable_Fe = Label(form, text = "Fe")
lable_Fe.grid(row = 10, column = 1, pady = 10, padx = 40 )
textbox_Fe = Entry(form)
textbox_Fe.grid(row = 10, column = 2)

def duDoanId3():
    RI = textbox_RI.get()
    Na = textbox_Na.get()
    Mg = textbox_Mg.get()
    Al = textbox_Al.get()
    Si = textbox_Si.get()
    K = textbox_K.get()
    Ca = textbox_Ca.get()
    Ba = textbox_Ba.get()
    Fe = textbox_Fe.get()
    if ((RI == '') or (Na == '') or (Mg == '') or (Al == '') or (Si == '') or (K == '') or (
            Ca == '') or (Ba == '') or (Fe == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        predictions = id3.predict([[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]])
        lbl2.configure(text=predictions)

button_id3 = Button(form, text='Kết quả dự đoán theo ID3', command=duDoanId3)
button_id3.grid(row=11, column=2, pady=20)
lbl2 = Label(form, text="...")
lbl2.grid(column=3, row=11)

lbl3 = Label(form)
lbl3.grid(column=1, row=11)
lbl3.configure(text="Tỉ lệ dự đoán đúng của ID3: %.2f " % accuracy_score(y_test, y_pred_id3) + '\n'
                    + 'Precision micro: %.4f ' % precision_score(y_test, y_pred_id3, average='micro') + '\n'
                    + 'Recall micro: %.4f ' % recall_score(y_test, y_pred_id3, average='micro') + '\n'
                    + 'F1 score micro: %.4f ' % f1_score(y_test, y_pred_id3, average='macro') + '\n')


form.mainloop() # Hiển thị form