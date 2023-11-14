from sklearn.model_selection import train_test_split
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import numpy as np
from sklearn import preprocessing 

def data_encoder(X) : 
    for i, j in enumerate(X) : 
        for k in range(0, 6) : 
            if(j[k]=="F"):
                j[k]=0
            elif(j[k]=="M"):
                j[k]=1
            elif(j[k]=="HIGH"):
                j[k]=2
            elif(j[k]=="LOW"):
                j[k]=3
            elif(j[k]=="NORMAL"):
                j[k]=4
            elif(j[k]=="drugA"):
                j[k]=5
            elif(j[k]=="drugB"):
                j[k]=6
            elif(j[k]=="drugC"):
                j[k]=7
            elif(j[k]=="drugX"):
                j[k]=8
            elif(j[k]=="DrugY"):
                j[k]=9
    return X

df = pd.read_csv('drug200.csv')

# mã hoá dữ liệu chữ
X_data=np.array(df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']].values)

data = data_encoder(X_data)
print(data)

# chia tập data thành 10% test và 90% train
dt_Train, dt_Test = train_test_split(data, test_size=0.1, shuffle=False)


kmeans = KMeans(n_clusters=2, init="random", n_init=10, max_iter=300, random_state=42).fit(dt_Train)
# init="random": chọn ngẫu nhiên các điểm dữ liệu để làm trung tâm cụm ban đầu
#  k-means++ chọn các tâm cụm ban đầu sao cho chúng cách nhau xa nhất có thể
# n_init: số lần chạy thuật toán với các tâm cụm khác nhau
# max_iter=300 : số lần lặp tối đa cho mỗi lần chạy, nếu đến 300 mà ko hội tụ thì thuật toán sẽ dừng và in ra kết quả
# random_state=42: tạo ra đường tròn có bán kính = 42 và lấy dữ liệu trong khoảng đó 

y_pred = kmeans.predict(dt_Test) # dự đoán nhãn cho tập cần kiểm tra
labels = kmeans.labels_ # trả về danh sách cụm ứng với từng vị trí
centers = kmeans.cluster_centers_ # trả về tâm của từng cụm

silhouetteScore = silhouette_score(dt_Test, y_pred)
daviesBouldinScore = davies_bouldin_score(dt_Test, y_pred)

# print('Chia nhóm:', labels)
# print('Tâm:', centers)


#form
form = Tk()
form.title("Dự đoán phân nhóm thuốc:")
form.geometry("900x300") # khởi tạo cửa sổ có dài = 900px rộng = 300px

genderOptions =['Male', 'Female']
levelOptions = ['HIGH','NORMAL', 'LOW']
drugOptions = ['drugA','drugB','drugC','drugX','DrugY']

lable_ten = Label(form, text = "Nhập thông tin cho mẫu thuốc:", font=("Arial Bold", 10), fg="red")
# Tạo 1 label có text là ...., font là Aria Bold, size là 10px và có màu chữ là đỏ
lable_ten.grid(row = 1, column = 1, padx = 40, pady = 10) # gán vị trí cho label, thêm độ đệm theo trục X là 40px là theo trục y là 10px

lable_age = Label(form, text = "Tuổi:")
lable_age.grid(row = 2, column = 1, padx = 40, pady = 10)
textbox_age = Entry(form)
textbox_age.grid(row = 2, column = 2)

lable_sex = Label(form, text = "Giới tính")
lable_sex.grid(row = 2, column = 3, pady = 10, padx = 40)
comboBox_sex = ttk.Combobox(form, values=genderOptions)
comboBox_sex.grid(row=2, column=4)

lable_BP = Label(form, text = "Mức huyết áp")
lable_BP.grid(row = 3, column = 1,pady = 10, padx = 40)
comboBox_BP = ttk.Combobox(form, values=levelOptions)
comboBox_BP.grid(row=3, column=2)


lable_choles = Label(form, text = "Mức độ cholesterol")
lable_choles.grid(row = 3, column = 3, pady = 10, padx = 40)
comboBox_choles = ttk.Combobox(form, values=levelOptions)
comboBox_choles.grid(row=3, column=4)


lable_naToK = Label(form, text = "Tỷ lệ natri và kali trong máu")
lable_naToK.grid(row = 4, column = 1, pady = 10, padx = 40 )
textbox_naToK = Entry(form)
textbox_naToK.grid(row = 4, column = 2)

lable_drug = Label(form, text = "Loại thuốc")
lable_drug.grid(row = 4, column = 3, pady = 10, padx = 40 )
comboBox_drug = ttk.Combobox(form, values=drugOptions)
comboBox_drug.grid(row=4, column=4)

def dudoan():
    # nếu 1 trong 3 input không có text thì thông báo lỗi
    if (textbox_age.get() == '' or comboBox_sex.get() =='' or comboBox_BP.get() =='' or comboBox_choles.get() =='' or textbox_naToK.get() == '' or comboBox_drug.get() ==''):
        messagebox.showerror("Lỗi", "Bạn cần nhập đầy đủ thông tin")
    else:
        age = textbox_age.get()
        sex = 'M' if comboBox_sex.get() == 'Male' else 'F'
        bp = comboBox_BP.get() 
        choles = comboBox_choles.get()
        naToK = textbox_naToK.get()
        drug = comboBox_drug.get()

        testing =np.array(data_encoder([[age, sex, bp, choles, naToK, drug]])).reshape(1,-1)
        
        print(testing)
        y_kqua = kmeans.predict(testing)
        lbl.configure(text=y_kqua[0])

acc = Label(form, text="Độ đo đánh giá chất lượng mô hình", font=("Arial Bold", 10), fg="red", pady=20)
acc.grid(column=1, row=9)
silhouette = Label(form, text="Silhouette Score:",anchor='w')
silhouette.grid(column=2, row=9)
Label(form, text=silhouetteScore,anchor='e').grid(column=3, row=9)

davies = Label(form, text=f"Davies Bouldin:", anchor='w')
davies.grid(column=2, row=10)
Label(form, text=daviesBouldinScore, anchor='e').grid(column=3, row=10)
    

button_cart = Button(form, text = 'Kết quả dự đoán theo KMeans', command = dudoan)
button_cart.grid(row = 11, column = 1, pady = 20)
lbl = Label(form, text="...", pady=10)
lbl.grid(column=2, row=11)

form.mainloop()