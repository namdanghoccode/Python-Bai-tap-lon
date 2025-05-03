import time
import pandas as pd
from bs4 import BeautifulSoup, Comment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from kneed import KneeLocator
from io import StringIO
import matplotlib.pyplot as plt
import os

# Thư mục gốc nơi các file sẽ được lưu vào
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Đường dẫn đến directory csv
csv_dir = os.path.join(base_dir, "csv")

# Đường dẫn đến file result.csv
result_path = os.path.join(csv_dir, "result.csv")

# Mở file csv để tính toán, các giá trị "N/A" được tính như NaN
df = pd.read_csv(result_path, na_values=["N/A"])

# Tạo ra bản copy của df để tính toán, không làm ảnh hưởng đến file result.csv
df_calc = df.copy()

# Những cột không cần tính toán
exclude_columns = ["Player", "Nation", "Team", "Position"]

# Lấy ra những cột cần tính toán
df_calc = df_calc.drop(columns=[col for col in exclude_columns if col in df_calc.columns])

# Chuyển đổi tất cả các cột trong DataFrame df_calc sang kiểu số
df_calc = df_calc.apply(pd.to_numeric, errors='coerce')

# Thay thế tất cả giá trị NaN trong df_calc bằng 0
df_calc = df_calc.fillna(0)

# Chuẩn hóa các đặc trưng trong df_calc để chúng có trung bình bằng 0 và độ lệch chuẩn bằng 1
scaler = StandardScaler()
# - fit: Tính trung bình và độ lệch chuẩn của từng cột
# - transform: Chuyển đổi dữ liệu bằng cách trừ trung bình và chia cho độ lệch chuẩn
scaled_features = scaler.fit_transform(df_calc)

# Khởi tạo danh sách rỗng để lưu giá trị inertia cho từng k
inertia = []

# Thực hiện thuật toán K-means với số cụm (k) từ 1 đến 15 và tính giá trị inertia (tổng bình phương khoảng cách từ các điểm đến tâm cụm gần nhất) để xác định số cụm tối ưu
k_range = range(1, 16)

for k in k_range:
    # Khởi tạo mô hình K-means với số cụm k, random_state=42 để đảm bảo kết quả tái lập, và n_init=10 để chạy thuật toán 10 lần với các tâm cụm ban đầu khác nhau, chọn kết quả tốt nhất
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    # Huấn luyện mô hình trên dữ liệu đã chuẩn hóa
    kmeans.fit(scaled_features)
    # Lấy giá trị inertia (độ đo mức độ tập trung của các cụm) và thêm vào danh sách inertia
    inertia.append(kmeans.inertia_)

# Tìm số cụm tối ưu (elbow_k) dựa trên phương pháp khuỷu tay (elbow method)
#   - phân tích đường cong inertia để tìm điểm mà việc tăng số cụm không còn cải thiện đáng kể giá trị inertia (điểm khuỷu tay)
knee_locator = KneeLocator(list(k_range), inertia, curve='convex', direction='decreasing')
# Trả về giá trị k tại điểm khuỷu tay (số cụm tối ưu)
elbow_k = knee_locator.knee

print(f"The optimal number of clusters (elbow point) is: {elbow_k}")

# Tạo đường dẫn để lưu hình ảnh
histograms_dir = os.path.join(base_dir, "histograms")
kmeans_dir = os.path.join(histograms_dir, "K-means")

# Khởi tạo mô hình PCA với tham số n_components=2, nghĩa là giảm dữ liệu xuống còn 2 chiều (2 thành phần chính)
pca = PCA(n_components=2)
# Kết quả pca_features là một mảng NumPy với shape (n_samples, 2), trong đó mỗi hàng là một điểm dữ liệu được biểu diễn bởi 2 thành phần chính
pca_features = pca.fit_transform(scaled_features)

# Áp dụng thuật toán K-means để phân cụm dữ liệu với số cụm tối ưu (elbow_k) được xác định trước đó (từ phương pháp khuỷu tay)
kmeans = KMeans(n_clusters=elbow_k, random_state=42, n_init=10)
# Clusters là một mảng NumPy chứa nhãn cụm (từ 0 đến elbow_k-1) cho từng điểm dữ liệu trong scaled_features
clusters = kmeans.fit_predict(scaled_features)

# Define the path for the PCA cluster plot
pca_plot_path = os.path.join(kmeans_dir, "PCA_2D_Cluster_Plot.png")

# Vẽ hình ảnh PCA 2 chiều
plt.figure(figsize=(8, 5))
scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel('Creative and Offensive Ability') # PCA1
plt.ylabel('Defensive and Ball Control Ability') # PCA2
plt.title(f'2D Cluster Visualization with k={elbow_k}')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.savefig(pca_plot_path)
plt.close()

print(f"2D PCA cluster plot saved at: {pca_plot_path}")

# Nếu cầu thủ có giá trị PCA1 cao => khả năng tấn công và sáng tạo rất tốt.
# Nếu cầu thủ có giá trị PCA1 thấp => có thể thiên về thủ môn hoặc phòng ngự.

# Nếu cầu thủ có giá trị PCA2 cao => cầu thủ chắc chắn, chuyền bóng tốt, hoạt động phòng ngự mạnh.
# Nếu cầu thủ có giá trị PCA2 thấp: cầu thủ có xu hướng tấn công, ghi bàn, dứt điểm nhiều.