
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split

df = pd.read_csv('drug200.csv')
le = preprocessing.LabelEncoder()
data = df.apply(le.fit_transform)
dt_Train, dt_Test = train_test_split(data, test_size=0.1, shuffle=True)

from yellowbrick.cluster import SilhouetteVisualizer
fig, ax = plt.subplots(2, 2, figsize =(15, 8)) #tạo ra một grid có 2x2 ô và lưu vào biến fig và ax.
for k in [2,3,4,5]:

    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(k,2) #tính kq phép chia số cụm `k` cho 2  (chia lấy dư -> q, mod có gtri từ 0-1)
    #q` thương, số dòng cho giá trị k ở câu lệnh lặp (dòng, lượng subplot theo c dọc)
    #mod soos dư, vị trí của giá trị k được xét trong lướt (cột)

    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', ax=ax[q-1][mod])
    #`ax` để chỉ định vị trí cho biểu đồ trong lưới đã tạo
    #ax[q-1][mod]` để đảm bảo chỉ số của hàng trong mảng bắt đầu từ 0, vì `q` được tính từ 0-1.
#vd nếu q là 3, thì vị trí dòng trong lưới sẽ = 3-1=2 , tức là dòng thứ 3 
# k=2` ax[0][0]`.
# k=3`both 1 ax[0][1]`.
# k=4`q2 mod0 ax[1][0]`.
# k=5`q2 mod1 ax[1][1]`. 

    visualizer.fit(dt_Train) #khớp train vẽ biểu đồ silhouette
    
visualizer.show()


# sil = (b-a)/max(a,b)
#a: tbc kc từ 1 điểm tới tất cả các điểm trong cụm
# b: tbc kc từ điểm đó tới tất cả các điểm ở trong cụm gần nhất 