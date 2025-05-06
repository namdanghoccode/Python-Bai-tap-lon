import pandas as pd
import os

# Thư mục gốc nơi các file sẽ được lưu vào
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Đường dẫn đến directory csv
csv_dir = os.path.join(base_dir, "csv")

# Đường dẫn đến file result.csv
result_path = os.path.join(csv_dir, "result.csv")

# Tạo thư mục txt trong base_dir nếu chưa tồn tại
txt_dir = os.path.join(base_dir, "txt")
os.makedirs(txt_dir, exist_ok=True)

# Mở file csv để tính toán, các giá trị "N/A" được tính như NaN
df = pd.read_csv(result_path, na_values=["N/A"])

# Tạo ra bản copy của df để tính toán, không làm ảnh hưởng đến file result.csv
df_calc = df.copy()

# Những cột không cần tính toán
exclude_columns = ["Player", "Nation", "Team", "Position"]

# Lấy ra những cột cần tính toán
numeric_columns = [col for col in df_calc.columns if col not in exclude_columns]

# Tạo ra dictionary tên là rankings
rankings = {}

for col in numeric_columns:
    # Lấy 3 dòng đầu có giá trị cao nhất trong cột col, kèm theo thông tin của Player và Team
    top_3_high = df_calc[["Player", "Team", col]].sort_values(by=col, ascending=False).head(3)

    # Đổi tên cột hiện tại (ví dụ "Succ%", "Tkld%", v.v.) thành "Value" để dễ xử lý hoặc hiển thị
    top_3_high = top_3_high.rename(columns={col: "Value"})

    # Thêm cột "Rank" với thứ hạng 1–2–3 tương ứng cho 3 dòng top đầu
    top_3_high["Rank"] = ["1st", "2nd", "3rd"]

    # Lấy 3 dòng đầu có giá trị thấp nhất trong cột col, bỏ qua NaN
    top_3_low = df_calc[["Player", "Team", col]].sort_values(by=col, ascending=True).dropna(subset=[col]).head(3)

    # Đổi tên cột hiện tại thành "Value"
    top_3_low = top_3_low.rename(columns={col: "Value"})

    # Thêm cột "Rank" với thứ hạng 1–2–3 tương ứng cho 3 dòng top thấp
    top_3_low["Rank"] = ["1st", "2nd", "3rd"]

    rankings[col] = {
        "Highest": top_3_high,
        "Lowest": top_3_low
    }

# Tạo ra file top_3.txt
top_3_path = os.path.join(txt_dir, "top_3.txt")

# Mở file top_3.txt
with open(top_3_path, "w", encoding="utf-8") as f:
    # Viết dữ liệu
    for stat, data in rankings.items():
        f.write(f"\nStatistic: {stat}\n")
        f.write("\nTop 3 Highest:\n")
        f.write(data["Highest"][["Rank", "Player", "Team", "Value"]].to_string(index=False))
        f.write("\n\nTop 3 Lowest:\n")
        f.write(data["Lowest"][["Rank", "Player", "Team", "Value"]].to_string(index=False))
        f.write("\n" + "-" * 50 + "\n")
print(f"✅ Saved top 3 rankings to {top_3_path}")