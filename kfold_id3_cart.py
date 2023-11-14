from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,recall_score)

df = pd.read_csv('glass.csv')
# print(df)
# Chia dữ liệu 70% làm data_tran, 30% làm data_test
dt_Train, dt_Test = train_test_split(df.values, test_size=0.3, shuffle=True)

# print(dt_Test)
# print(dt_Train)

# tính tỷ lệ đúng của y dự đoán
def accuracy(y, y_pred):
    count = 0
    for i in range(0, len(y)):
        if (y[i] == y_pred[i]):
            count += 1
    return count/len(y)

# hàm trả về mô hình dộ chính xác theo thuật toán CART
def the_best_cart(dt_Train):
    
    k = 3 # chia dữ liệu làm 3 phần
    kf = KFold(n_splits=k, random_state=None) # một đối tượng kfold với số phân chia là k và random_state = none (không sử dụng ngẫu nhiên).

    max = -999999 # Khai báo giá trị max ban đầu là -999999 để lưu trữ giá trị accuracy lớn nhất.
    for train_index, validation_index in kf.split(dt_Train): #vòng lặp for để chia dữ liệu thành bộ train và bộ validation.
        #Đặt các biến "x_train, x_validation, y_train, y_validation" để lưu trữ dữ liệu train và validation tương ứng.
        X_train, X_validation = dt_Train[train_index, :9], dt_Train[validation_index, :9] 
        y_train, y_validation = dt_Train[train_index, 9], dt_Train[validation_index, 9]

        cart = DecisionTreeClassifier(criterion = 'gini', max_depth = 8 , min_samples_leaf = 5)
        # max_deep: độ sâu tối đa, nếu> 8, quá trình xây dựng cây sẽ dừng lại và nút lá sẽ được trả về.
        # min_samples_leaf: số lượng mẫu tối thiểu cần có ở một nút lá. Nếu số lượng mẫu trong nút lá < 5 thì tt dừng và trả về nút lá.
        cart.fit(X_train, y_train)
        y_train_pred = cart.predict(X_train)  # dự đoán nhãn cho dư liệu train
        y_validation_pred = cart.predict(X_validation) # dự đoán nhãn cho dư liệu validation
        y_train = np.array(y_train) # Chuyển đổi các train từ kiểu dữ liệu list sang kiểu dữ liệu array.
        y_validation = np.array(y_validation) # Chuyển đổi các nhãn từ kiểu dữ liệu list sang kiểu dữ liệu array.
        sum_accuracy = accuracy(y_train, y_train_pred) + accuracy(y_validation, y_validation_pred) # Tính tổng tỷ lệ dự đoán đúng cho cả dữ liệu train và validation.
        if (sum_accuracy > max):  #So sánh tổng tỷ lệ dự đoán đúng với giá trị max hiện tại. nếu tổng tỷ lệ đúng >  max, 
            max = sum_accuracy #cập nhật max  
            cart_best = cart #  lưu trữ mô hình cây quyết định vào biến "cart_best".
    return cart_best # hàm trả về mô hình dộ chính xác theo thuật toán CART

# hàm trả về mô hình có độ chính xác theo thuật toán ID3
def the_best_id3(dt_Train):
    k = 3
    kf = KFold(n_splits=k, random_state=None) #random state sd gtri ngẫu nhiên mỗi lần chạy quá trình cross-validation, các chỉ số mẫu được sinh ngẫu nhiên sẽ khác nhau
    max = -999999# Khai báo giá trị max ban đầu là -999999 để lưu trữ giá trị accuracy lớn nhất
    for train_index, validation_index in kf.split(dt_Train): #vòng lặp for để chia dữ liệu thành bộ train và bộ validation.
        #Đặt các biến "x_train, x_validation, y_train, y_validation" để lưu trữ dữ liệu train và validation tương ứng.
        X_train, X_validation = dt_Train[train_index, :9], dt_Train[validation_index, :9]
        y_train, y_validation = dt_Train[train_index, 9], dt_Train[validation_index, 9]
        id3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_split=2, min_samples_leaf=7,max_depth = 8)
# "min_samples_split=2"  số lượng mẫu tối thiểu cần có để tiếp tục chia thành các nút con. nếu nhỏ hơn thì tt dừng
# "min_samples_leaf=7" số lượng mẫu tối thiểu cần phải có ở mỗi lá để cây quyết định còn chia nhỏ. nếu lượng mẫu tại 1 lá <7 thì tt dừng
# "max_depth = 8 Nếu sâu của một nút vượt quá giá trị này, tt dừng và nút đó trở thành nút lá
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


# form
form = Tk() # Khởi tạo đối tượng Tk()
form.title("Dự đoán loại thủy tinh:") # title của form
form.geometry("800x650") # kích thước của form

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

# cart
cart_best = the_best_cart(dt_Train) # Lấy ra mô hình tốt nhất theo thuật toán cart trong tập dt_train và gán cho cart_best
y_pred_cart = cart_best.predict(dt_Test[:, :9]) # Dự đoán y_pred trên tập dt_test
y_test = np.array(dt_Test[:, 9])
lbl1 = Label(form)
lbl1.grid(column=1, row=11)
lbl1.configure(text="Tỉ lệ dự đoán đúng của CART: %.5f " % accuracy_score(y_test, y_pred_cart) + '\n'
                    + 'Precision: %.9f ' % precision_score(y_test, y_pred_cart, average='micro') +'\n'
                    + 'Recall: %.9f ' % recall_score(y_test, y_pred_cart, average='micro') + '\n'
                    + 'F1 score: %.9f ' % f1_score(y_test, y_pred_cart, average='macro') + '\n')

def duDoanCart():
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
        RI = float(RI)
        Mg = float(Mg)
        Al = float(Al)
        Si = float(Si)
        K = float(K)
        Ca = float(Ca)
        Ba = float(Ba)
        Fe = float(Fe)
        # Lấy data từ các input, cho vào cuối data rồi mã hóa dữ liệu
        y_kqua = cart_best.predict([[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]]) # Từ cái mô hình theo thuật toán cart đc tìm thấy bên trên và X dự đoán => tìm y dự đoán
        lbl.configure(text=y_kqua)
        

# id3
id3_best = the_best_id3(dt_Train)
y_pred_id3 = id3_best.predict(dt_Test[:, :9])
lbl3 = Label(form)
lbl3.grid(column=3, row=11)
lbl3.configure(text="Tỉ lệ dự đoán đúng của ID3: %.5f " % accuracy_score(y_test, y_pred_id3) + '\n'
                    + 'Precision: %.9f ' % precision_score(y_test, y_pred_id3, average='micro') + '\n'
                    + 'Recall: %.9f ' % recall_score(y_test, y_pred_id3, average='micro') + '\n'
                    + 'F1 score: %.9f ' % f1_score(y_test, y_pred_id3, average='macro') + '\n')


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
        RI = textbox_RI.get()
        Na = textbox_Na.get()
        Mg = float(Mg)
        Al = float(Al)
        Si = float(Si)
        K = float(K)
        Ca = float(Ca)
        Ba = float(Ba)
        Fe = float(Fe)
        y_kqua = cart_best.predict([[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]]) # Từ cái mô hình theo thuật toán cart đc tìm thấy bên trên và X dự đoán => tìm y dự đoán
        lbl2.configure(text=y_kqua)


accuracy_cart = accuracy_score(y_test, y_pred_cart) # tính tỷ lệ đự đoán đúng của mô hình theo thuật toán cart
accuracy_id3 = accuracy_score(y_test, y_pred_id3) # tính tỷ lệ dự đoán đúng của mô hình theo thuật toán id3
if (accuracy_cart >= accuracy_id3): # nếu tỷ lệ đự đoán đúng của mô hình cart lớn hơn tỷ lệ dự đoán đúng cùa mô hình
    # id3 thì hiển thị nút dự đoán bên phía mô hình cart
    lbl_cart = Label(form, font=("Arial Bold", 10), fg="red")
    lbl_cart.configure(text="=> Dự đoán theo CART.")
    lbl_cart.grid(row=12, column=3, pady=20)

    button_cart = Button(form, text='Kết quả dự đoán theo CART', command=duDoanCart)
    button_cart.grid(row=12, column=1, pady=20)
    lbl = Label(form, text="...")
    lbl.grid(column=2, row=12)
else: # ngược lại
    lbl_id3 = Label(form, font=("Arial Bold", 10), fg="red")
    lbl_id3.grid(row = 12, column = 2, pady=10)
    lbl_id3.configure(text="=> Dự đoán theo ID3.")

    button_id3 = Button(form, text='Kết quả dự đoán theo ID3', command=duDoanId3)
    button_id3.grid(row=12, column=3, pady=20)
    lbl2 = Label(form, text="...")
    lbl2.grid(column=4, row=12)

form.mainloop() # Hiển thị form