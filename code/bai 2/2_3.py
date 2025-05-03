import pandas as pd
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

# Lọc ra danh sách các cột dạng số để xử lý
numeric_columns = [col for col in df_calc.columns if col not in exclude_columns]

# Với mỗi cột số: ép kiểu về numeric, các lỗi sẽ thành NaN, sau đó điền NaN thành 0
for col in numeric_columns:
    df_calc[col] = pd.to_numeric(df_calc[col], errors="coerce").fillna(0)

# Lọc dữ liệu: chỉ giữ cầu thủ chơi ít nhất 300 phút (nếu có cột 'Min')
if 'Min' in df_calc.columns:
    df_calc = df_calc[df_calc['Min'] >= 300]

# Chỉ định các chỉ số thống kê sẽ vẽ biểu đồ
selected_stats = ["Gls per 90", "xG per 90", "SCA90", "GA90", "TklW", "Blocks"]

# Tạo đường dẫn thư mục để lưu biểu đồ
histograms_dir = os.path.join(base_dir, "histograms")
league_dir = os.path.join(histograms_dir, "all_players")   # Biểu đồ toàn giải
teams_dir = os.path.join(histograms_dir, "teams") # Biểu đồ theo đội

# Tạo thư mục nếu chưa tồn tại
os.makedirs(league_dir, exist_ok=True)
os.makedirs(teams_dir, exist_ok=True)

# Lấy danh sách đội bóng, sắp xếp theo thứ tự ABC
teams = sorted(df_calc["Team"].unique())

# Lặp qua từng chỉ số cần vẽ
for stat in selected_stats:
    # Nếu không tìm thấy cột trong DataFrame thì bỏ qua và thông báo.
    if stat not in df_calc.columns:
        print(f"Statistic {stat} not found in DataFrame. Skipping...")
        continue

    #  Vẽ biểu đồ histogram toàn giải cho chỉ số stat và lưu vào thư mục có đường dẫn là league_dir
    plt.figure(figsize=(10, 6))
    plt.hist(df_calc[stat], bins=20, color="skyblue", edgecolor="black")

    # Đặt tiêu đề và nhãn trục cho biểu đồ
    plt.title(f"League-Wide Distribution of {stat}")
    plt.xlabel(stat)
    plt.ylabel("Number of Players")
    plt.grid(True, alpha=0.3)

    # Lưu biểu đồ cho đội hiện tại
    plt.savefig(os.path.join(league_dir, f"{stat}_league.png"), bbox_inches="tight")
    plt.close()
    print(f"Saved league-wide histogram for {stat}")

    # Lặp qua từng đội để tạo biểu đồ riêng cho từng đội.
    for team in teams:
        team_data = df_calc[df_calc["Team"] == team]

        # Vẽ histogram cho đội hiện tại
        plt.figure(figsize=(8, 6))

        # Dùng màu xanh lá cho chỉ số phòng ngự, màu xanh da trời cho chỉ số khác
        plt.hist(team_data[stat], bins=10,
                 color="lightgreen" if stat in ["GA90", "TklW", "Blocks"] else "skyblue",
                 edgecolor="black", alpha=0.7)

        # Đặt tiêu đề và nhãn trục cho biểu đồ
        plt.title(f"{team} - Distribution of {stat}")
        plt.xlabel(stat)
        plt.ylabel("Number of Players")
        plt.grid(True, alpha=0.3)

        # Lưu biểu đồ cho đội hiện tại, thay dấu cách trong tên chỉ số bằng _ để tạo tên file hợp lệ
        stat_filename = stat.replace(" ", "_")
        plt.savefig(os.path.join(teams_dir, f"{team}_{stat_filename}.png"), bbox_inches="tight")
        plt.close()
        print(f"Saved histogram for {team} - {stat}")

print("✅ All histograms for selected statistics have been generated and saved under 'histograms'.")