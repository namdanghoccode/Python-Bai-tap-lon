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

highest_team_stats_path = os.path.join(base_dir, "highest_team_stats.csv")

# Tạo danh sách các thống kê mang tính tiêu cực
negative_stats = [
    "GA90", "CrdY", "CrdR", "Lost", "Mis", "Dis", "Fls", "Off", "Aerl Lost"
]

# Đọc file CSV trước đó đã lưu (highest_team_stats.csv) thành DataFrame
highest_teams_df = pd.read_csv(highest_team_stats_path)

# Lọc ra những dòng có thống kê tích cực (không nằm trong danh sách negative_stats).
positive_stats_df = highest_teams_df[~highest_teams_df["Statistic"].isin(negative_stats)]

# Đếm số lần mỗi đội xuất hiện trong positive_stats_df (tức là số thống kê tích cực mà đội đó dẫn đầu)
team_wins = positive_stats_df["Team"].value_counts()

# Tìm tên đội có nhiều thống kê tích cực nhất (nhiều lần dẫn đầu nhất).
best_team = team_wins.idxmax()

# Lấy số lượng chỉ số tích cực mà đội đó dẫn đầu (giá trị lớn nhất trong team_wins).
win_count = team_wins.max()

# Tạo nội dung văn bản để ghi ra file .txt
result_text = (
    f"The best-performing team in the 2024-2025 Premier League season is: {best_team}\n"
    f"They lead in {win_count} out of {len(positive_stats_df)} positive statistics."
)

# Tạo thư mục txt trong base_dir nếu chưa tồn tại
txt_dir = os.path.join(base_dir, "txt")
os.makedirs(txt_dir, exist_ok=True)

# Tạo đường dẫn đầy đủ tới file txt kết quả.
txt_result_path = os.path.join(txt_dir, "The best-performing team.txt")

# Mở (hoặc tạo mới) file TXT để ghi nội dung kết quả (result_text)
with open(txt_result_path, "w", encoding="utf-8") as f:
    f.write(result_text)

print(f"\nResult also saved to: {txt_result_path}")