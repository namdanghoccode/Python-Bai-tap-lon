import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
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
os.makedirs(kmeans_dir, exist_ok=True)  # Create the directories if they don't exist
plot_path = os.path.join(kmeans_dir, "The optimal number of clusters.png")

# Vẽ hình ảnh
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, 'bo-', label='Inertia')
plt.vlines(elbow_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red', label=f'Elbow at k={elbow_k}')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.legend()
plt.grid(True)
plt.savefig(plot_path)
plt.close()

# Giải thích tại sao chọn cluster = 4
"""
Như được đề cập trong slide thì chúng ta sẽ chọn số cluster dựa trên K-means clustering
Elbow point là điểm mà ở đó tốc độ suy giảm của hàm biến dạng sẽ thay đổi nhiều nhất. Tức là kể từ sau vị trí này thì gia tăng thêm số lượng cụm cũng không giúp hàm biến dạng giảm đáng kể.
Dựa vào hình ảnh trong histograms directory thì ta có thể thấy rằng:
    Khi cluster(k) còn nhỏ (ví dụ k = 1,2), inertia (tổng khoảng cách bình phương từ mỗi điểm dữ liệu đến tâm cụm gần nhất) rất lớn => gom cụm còn rất tệ.
    Khi cluster(k) tăng dần, inertia giảm đi => việc gom cụm trở nên tốt hơn.
    Tuy nhiên, sau một giá trị cluster(k) nhất định, việc thêm cụm mới sẽ không làm inertia giảm đáng kể nữa.
    
Trong bài này, em lấy số cluster là 4 vì:
    Từ k = 1 đến k = 4, đường cong giảm rất mạnh, nghĩa là mỗi lần tăng thêm 1 cụm thì hiệu quả gom nhóm được cải thiện đáng kể.
    Sau k = 4, đường cong bắt đầu "bẻ góc" và dần phẳng ra => nghĩa là thêm nhiều cụm nữa cũng chỉ giảm inertia rất ít thôi.
    Vì vậy, k = 4 được chọn, bởi vì nó chính là điểm "elbow" — điểm mà việc thêm cụm không còn đáng giá nữa.
"""
