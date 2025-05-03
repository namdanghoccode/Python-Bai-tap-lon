import pandas as pd
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

# Lọc ra danh sách các cột dạng số để xử lý
numeric_columns = [col for col in df_calc.columns if col not in exclude_columns]

# Với mỗi cột số: ép kiểu về numeric, các lỗi sẽ thành NaN, sau đó điền NaN thành 0
for col in numeric_columns:
    df_calc[col] = pd.to_numeric(df_calc[col], errors="coerce").fillna(0)

# Nhóm dữ liệu theo cột "Team" và tính giá trị trung bình cho các cột dạng số
# - .reset_index() để chuyển Team từ chỉ mục nhóm thành cột bình thường.
team_means = df_calc.groupby("Team")[numeric_columns].mean().reset_index()

# Tạo danh sách rỗng để lưu thông tin mỗi đội cao nhất ở từng thống kê.
highest_teams = []

# Lặp qua từng chỉ số dạng số đã xác định.
for stat in numeric_columns:
    #  Nếu chỉ số stat không tồn tại trong bảng dữ liệu thì bỏ qua, tránh lỗi.
    if stat not in df_calc.columns:
        print(f"Statistic {stat} not found in DataFrame. Skipping...")
        continue

    # Tìm dòng (đội) có giá trị trung bình lớn nhất cho chỉ số stat.
    #   - idxmax() trả về chỉ số hàng có giá trị lớn nhất trong cột stat.
    max_row = team_means.loc[team_means[stat].idxmax()]

    # Thêm vào danh sách highest_teams một dict gồm:
    #   - Tên chỉ số
    #   - Tên đội có giá trị cao nhất
    #   - Giá trị trung bình đã làm tròn đến 2 chữ số
    highest_teams.append({
        "Statistic": stat,
        "Team": max_row["Team"],
        "Mean Value": round(max_row[stat], 2)
    })

# Chuyển danh sách highest_teams thành bảng DataFrame để dễ lưu và xử lý.
highest_teams_df = pd.DataFrame(highest_teams)

# Đảm bảo thư mục để lưu file CSV đã tồn tại, nếu chưa thì tạo mới.
os.makedirs(csv_dir, exist_ok=True)
# Tạo đường dẫn đến file CSV sẽ ghi kết quả.
highest_team_stats_path = os.path.join(csv_dir, "highest_team_stats.csv")

# Ghi bảng highest_teams_df ra file CSV
highest_teams_df.to_csv(highest_team_stats_path, index=False, encoding="utf-8-sig")

print(f"✅ Saved highest team stats to {highest_team_stats_path} with {highest_teams_df.shape[0]} rows.")

