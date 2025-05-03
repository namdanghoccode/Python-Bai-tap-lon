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

# Convert NaN to 0 in numeric columns for calculations
numeric_columns = [col for col in df_calc.columns if col not in exclude_columns]

# 2. Calculate median, mean, and standard deviation for results2.csv

# Khởi tạo danh sách rows
rows = []

# Khởi tạo từ điển all_stats
all_stats = {"": "all"}

# Duyệt qua từng cột trong numeric_columns
for col in numeric_columns:
    # Tính median
    all_stats[f"Median of {col}"] = df_calc[col].median()
    # Tính mean
    all_stats[f"Mean of {col}"] = df_calc[col].mean()
    # Tính std
    all_stats[f"Std of {col}"] = df_calc[col].std()
    # Tẩt cả được lưu vào all_stats

# Thêm từ điển all_stats vào danh sách rows.
rows.append(all_stats)

# Lấy danh sách tất cả các đội từ cột "Team" trong df_calc và sắp xếp chúng theo thứ tự alphabetically.
teams = sorted(df_calc["Team"].unique())

# Lặp qua từng đội và tính toán thống kê
for team in teams:
    # Lọc ra các cầu thủ có cột "Team" trùng với đội hiện tại
    team_df = df_calc[df_calc["Team"] == team]

    # Khởi tạo một từ điển chứa tên đội (dưới dạng khóa rỗng "": team)
    team_stats = {"": team}

    # Lặp qua các cột số liệu trong numeric_columns, tính toán các thống kê (median, mean, std) cho mỗi đội tương
    for col in numeric_columns:
        # Tính median
        team_stats[f"Median of {col}"] = team_df[col].median()
        # Tính mean
        team_stats[f"Mean of {col}"] = team_df[col].mean()
        # Tính std
        team_stats[f"Std of {col}"] = team_df[col].std()
        # Tất cả được lưu vào trong team_stats

    # Thêm từ điển team_stats vào danh sách rows.
    rows.append(team_stats)

# Tạo một DataFrame mới từ danh sách rows
results_df = pd.DataFrame(rows)

# ổi tên cột của DataFrame để đảm bảo cột đầu tiên có tên hợp lệ
results_df = results_df.rename(columns={"": ""})

for col in results_df.columns:
    if col != "":
        # Làm tròn các giá trị trong DataFrame với 2 số sau dấu .
        results_df[col] = results_df[col].round(2)

# Tạo ra file result2.csv
results2_path = os.path.join(csv_dir, "results2.csv")

# Lưu DataFrame results_df vào file CSV
results_df.to_csv(results2_path, index=False, encoding="utf-8-sig")
print(f"✅ Successfully saved statistics to {results2_path} with {results_df.shape[0]} rows and {results_df.shape[1]} columns.")