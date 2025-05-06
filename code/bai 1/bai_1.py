import time # Khai báo thư viện time để làm việc với thời gian và ngày giờ ở cấp độ thấp
import pandas as pd # Khai báo thư viện để xử lý dữ liệu dạng bảng
from bs4 import BeautifulSoup, Comment # Khai báo thư viện dùng để phân tích và trích xuất dữ liệu từ HTML hoặc XML, Comment dùng để nhận diện các đoạn mã HTML nằm trong thẻ chú thích
from selenium import webdriver
# Selenium là thư viện tự động hóa trình duyệt (được dùng để crawl web động – các trang cần tương tác JS mới hiển thị dữ liệu).
# Webdriver dùng để điều khiển trình duyệt (như Chrome, Firefox).
from selenium.webdriver.chrome.options import Options # Dùng để cấu hình tùy chọn cho Chrome, ví dụ chạy ẩn (headless), không hiển thị ảnh, tắt thông báo
from selenium.webdriver.chrome.service import Service # Dùng để tạo một dịch vụ điều khiển trình điều khiển ChromeDriver.
from webdriver_manager.chrome import ChromeDriverManager # Tự động tải và cấu hình ChromeDriver phù hợp với phiên bản trình duyệt Chrome trên máy bạn.
from io import StringIO # Dùng để  xử lý chuỗi giống như file.
import os # Dùng để thao tác với hệ điều hành: đường dẫn, thư mục, file, biến môi trường



# Thư mục gốc nơi các file sẽ được lưu vào
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))



# Hàm để chuyển số tuổi từ dạng "AA-DDD" sang dang float của tuổi
def convert_age_to_decimal(age_str):
    #Chuyển tuổi sang thành dạng chuỗi với không có dấu cách ở 2 đầu
    age_str = str(age_str).strip()

    # Tách chuỗi theo dấu gạch ngang
    if "-" in age_str:
        # Tách chuỗi thành số năm tuổi và số ngày tuổi
        years, days = map(int, age_str.split("-"))

        # Làm tròn số ngày tuổi
        decimal_age = years + (days / 365)

        # Trả về giá trị được làm tròn 2 số của chữ số thập phân
        return round(decimal_age, 2)
    # Trả về N/A nếu không tìm thấy
    return "N/A"

# DÙng để làm đẹp cột nation.
def extract_country_code(nation_str):
    try:
        # Ví dụ: Từ "eng-Eng" thành "Eng"
        return nation_str.split()[-1]

    except (AttributeError, IndexError):
        return "N/A"



"""
Truy cập trang web tự động
"""
# Cấu hình cho Selenium WebDriver
options = Options() # Tạo một đối tượng Options để cấu hình trình duyệt Chrome
options.add_argument("--headless") # Chạy trình duyệt ở chế độ ẩn (headless) — không mở cửa sổ trình duyệt thật ra
options.add_argument("--disable-gpu") # Tắt GPU hardware acceleration
options.add_argument("--no-sandbox") # Tắt chế độ sandbox (bảo vệ) của Chrome.

# Khởi tạo trình điều khiển Chrome (webdriver.Chrome) với:
#    ChromeDriverManager().install() => Tự động tải và chỉ định đúng phiên bản ChromeDriver.
#    options=options => Áp dụng tất cả cấu hình vừa khai báo ở trên.
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)



"""
Khai báo các urls cần có để crawl data và các tabel id để có thể tìm tới đúng bảng để lấy dữ liệu
"""
# Liệt kê ra các urls cần truy cập
urls = [
    "https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/keepers/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/shooting/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/passing/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/gca/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/defense/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/possession/2024-2025-Premier-League-Stats",
    "https://fbref.com/en/comps/9/2024-2025/misc/2024-2025-Premier-League-Stats",
]

# Liệt kê ra các table_ids cần crawl
table_ids = [
    "stats_standard",
    "stats_keeper",
    "stats_shooting",
    "stats_passing",
    "stats_gca",
    "stats_defense",
    "stats_possession",
    "stats_misc",
]

# Liệt kê ra các cột cần lấy dữ liệu
required_columns = [
    "Player", "Nation", "Team", "Position", "Age",
    "Matches Played", "Starts", "Minutes",
    "Gls", "Ast", "crdY", "crdR",
    "xG", "xAG",
    "PrgC", "PrgP", "PrgR",
    "Gls per 90", "Ast per 90", "xG per 90", "xAG per 90",
    "GA90", "Save%", "CS%", "PK Save%",
    "SoT%", "SoT per 90", "G per Sh", "Dist",
    "Cmp", "Cmp%", "TotDist", "ShortCmp%", "MedCmp%", "LongCmp%", "KP", "Pass into 1_3", "PPA", "CrsPA",
    "SCA", "SCA90", "GCA", "GCA90",
    "Tkl", "TklW",
    "Deff Att", "Lost",
    "Blocks", "Sh", "Pass", "Int",
    "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen",
    "Take-Ons Att", "Succ%", "Tkld%",
    "Carries", "ProDist", "Carries 1_3", "CPA", "Mis", "Dis",
    "Rec", "Rec PrgR",
    "Fls", "Fld", "Off", "Crs", "Recov",
    "Aerl Won", "Aerl Lost", "Aerl Won%"
]

# Thay đổi các tên cột sao cho giống tên các cột cần lấy dữ liệu vì tên thật của các cột trên website không giống
column_rename_dict = {
    "stats_standard": {
        "Unnamed: 1": "Player",
        "Unnamed: 2": "Nation",
        "Unnamed: 3": "Position",
        "Unnamed: 4": "Team",
        "Unnamed: 5": "Age",
        "Playing Time": "Matches Played",
        "Playing Time.1": "Starts",
        "Playing Time.2": "Minutes",
        "Performance": "Gls",
        "Performance.1": "Ast",
        "Performance.6": "crdY",
        "Performance.7": "crdR",
        "Expected": "xG",
        "Expected.2": "xAG",
        "Progression": "PrgC",
        "Progression.1": "PrgP",
        "Progression.2": "PrgR",
        "Per 90 Minutes": "Gls per 90",
        "Per 90 Minutes.1": "Ast per 90",
        "Per 90 Minutes.5": "xG per 90",
        "Per 90 Minutes.6": "xAG per 90"
    },
    "stats_keeper": {
        "Unnamed: 1": "Player",
        "Unnamed: 4": "Team",
        "Performance.1": "GA90",
        "Performance.4": "Save%",
        "Performance.9": "CS%",
        "Penalty Kicks.4": "PK Save%"
    },
    "stats_shooting": {
        "Unnamed: 1": "Player",
        "Unnamed: 4": "Team",
        "Standard.3": "SoT%",
        "Standard.5": "SoT per 90",
        "Standard.6": "G per Sh",
        "Standard.8": "Dist"
    },
    "stats_passing": {
        "Unnamed: 1": "Player",
        "Unnamed: 4": "Team",
        "Total": "Cmp",
        "Total.2": "Cmp%",
        "Total.3": "TotDist",
        "Short.2": "ShortCmp%",
        "Medium.2": "MedCmp%",
        "Long.2": "LongCmp%",
        "Unnamed: 26": "KP",
        "Unnamed: 27": "Pass into 1_3",
        "Unnamed: 28": "PPA",
        "Unnamed: 29": "CrsPA",
    },
    "stats_gca": {
        "Unnamed: 1": "Player",
        "Unnamed: 4": "Team",
        "SCA.1": "SCA90",
        "GCA.1": "GCA90",
    },
    "stats_defense": {
        "Unnamed: 1": "Player",
        "Unnamed: 4": "Team",
        "Tackles": "Tkl", "Tackles.1": "TklW",
        "Challenges.1": "Deff Att",
        "Challenges.3": "Lost",
        "Blocks": "Blocks",
        "Blocks.1": "Sh",
        "Blocks.2": "Pass",
        "Unnamed: 20": "Int",
    },
    "stats_possession": {
        "Unnamed: 1": "Player",
        "Unnamed: 4": "Team",
        "Touches": "Touches",
        "Touches.1": "Def Pen",
        "Touches.2": "Def 3rd",
        "Touches.3": "Mid 3rd",
        "Touches.4": "Att 3rd",
        "Touches.5": "Att Pen",
        "Touches.6": "Live",
        "Take-Ons": "Take-Ons Att",
        "Take-Ons.2": "Succ%",
        "Take-Ons.4": "Tkld%",
        "Carries": "Carries",
        "Carries.2": "ProDist",
        "Carries.4": "Carries 1_3",
        "Carries.5": "CPA",
        "Carries.6": "Mis",
        "Carries.7": "Dis",
        "Receiving": "Rec",
        "Receiving.1": "Rec PrgR",
    },
    "stats_misc": {
        "Unnamed: 1": "Player",
        "Unnamed: 4": "Team",
        "Performance.3": "Fls",
        "Performance.4": "Fld",
        "Performance.5": "Off",
        "Performance.6": "Crs",
        "Performance.12": "Recov",
        "Aerial Duels": "Aerl Won",
        "Aerial Duels.1": "Aerl Lost",
        "Aerial Duels.2": "Aerl Won%"
    }
}


"""
Lấy dữ liệu trên trang web
"""
# Khởi tạo dictionary tên là all_tables
all_tables = {}

# Duyệt qua từng url và tabel_id
for url, table_id in zip(urls, table_ids):
    print(f"Processing {table_id} from {url}")
    driver.get(url) # Dùng Selenium WebDriver để mở trang web có địa chỉ là url
    time.sleep(3) # Tạm dừng chương trình trong 3 giây để chờ trang web tải xong

    # Trích xuất dữ liệu HTML bằng beautiful soup
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Tìm bảng có id=table_id trên trang web
    table = soup.find("table", {"id": table_id})

    if not table:
        print(f"Table {table_id} not found!")
        continue

    # Chuyển bảng HTML thành DataFrame thông qua Pandas
    #   - str(table) để chuyển đối tượng table thành dạng chuỗi
    #   - dùng stringIO để giả lập một file từ chuỗi HTML bên trên vì pd.read_html() mong muốn đọc từ file
    #   - pd.read_html() dùng để phân tích HTML và tự động tìm bảng và chuyển thành DataFrame
    #   - dòng đầu tiên trong bảng HTML (chỉ số dòng = 0) sẽ được dùng làm tên cột (header) trong DataFrame
    #   - vì pd.read_html() trả về một list các DataFrame (vì HTML có thể chứa nhiều bảng), dùng [0] để lấy bảng đầu tiên.
    df = pd.read_html(StringIO(str(table)), header=0)[0]

    print(f"Original columns in {table_id}:", df.columns.tolist())

    # Đổi tên các cột của df dictionary tên là column_rename_dict
    df = df.rename(columns=column_rename_dict.get(table_id))

    # Loại bỏ các cột trùng tên trong df, giữ lại cột đầu tiên trong số các cột bị trùng
    #   - df.columns.duplicated() trả về một mảng Boolean, trong đó True tại các vị trí tương ứng với cột bị lặp
    #   - ~ là toán tử phủ định, chuyển True thành False và ngược lại.
    #   - df.loc[:, ...] chọn tất cả các hàng (:) và chỉ giữ các cột không bị lặp (tức là False trong mảng duplicated()).
    df = df.loc[:, ~df.columns.duplicated()]

    # Tạo thêm cột PLayer_Team để có thể dựa vào đấy để merge các cột từ các bảng khác nhau
    df["Player_Team"] = df["Player"].astype(str) + "_" + df["Team"].astype(str)

    #Chuyển đổi age sang giá trị float
    if "Age" in df.columns:
        df["Age"] = df["Age"].apply(convert_age_to_decimal)

    print(f"Renamed columns in {table_id}:", df.columns.tolist())

    # Tạo danh sách required_for_table chứa các cột từ required_columns có trong df, đồng thời đảm bảo cột Player_Team được bao gồm
    required_for_table = [col for col in required_columns if col in df.columns] + ["Player_Team"]
    # Cập nhật df để chỉ chứa các cột được liệt kê trong required_for_table, loại bỏ tất cả các cột khác
    df = df[required_for_table]
    # Lưu DataFrame df đã được làm sạch và lọc vào từ điển all_tables
    all_tables[table_id] = df

"""
Ghép các DataFrame vào chung 1 bảng
"""

# Khởi tạo DataFrame để gộp
merged_df = None

# Duyệt dictionary all_tables
for table_id, df in all_tables.items():
    # Kiểm tra xem merge_df có phải None hay không
    if merged_df is None:
        # Gán df cho merge_df
        merged_df = df
    else:
        # Xác định các hàng có giá trị Player_Team trùng lặp trong merged_df và df để kiểm tra dữ liệu trước khi hợp nhất
        merged_df_duplicates = merged_df[merged_df['Player_Team'].duplicated(keep=False)]
        df_duplicates = df[df['Player_Team'].duplicated(keep=False)]

        # Loại bỏ các hàng trùng lặp trong df dựa trên cột Player_Team, giữ lại hàng đầu tiên
        df = df.drop_duplicates(subset=['Player_Team'], keep='first')

        # Hợp nhất merged_df (DataFrame tổng hợp từ các bảng trước) với df (DataFrame của bảng hiện tại) dựa trên cột Player_Team
        merged_df = pd.merge(
            merged_df,
            df,
            on=["Player_Team"],
            how="outer",
            suffixes=(None, f"_{table_id}")
        )

        # Xử lý các cột trùng tên (như Player, Team) xuất hiện sau hợp nhất, gộp dữ liệu từ cả hai cột và xóa cột thừa
        for col in required_columns:
            if col == "Player_Team":
                continue
            col_y = f"{col}_{table_id}"
            if col in merged_df.columns and col_y in merged_df.columns:
                merged_df[col] = merged_df[col].combine_first(merged_df[col_y])
                merged_df = merged_df.drop(columns=[col_y])

# Xóa cột Player_Team
merged_df = merged_df.drop(columns=["Player_Team"])

# Giữ lại đúng thứ tự và tập hợp các cột mong muốn trong merged_df, loại bỏ những cột không có mặt sau khi merge
merged_df = merged_df.loc[:, [col for col in required_columns if col in merged_df.columns]]

# Định nghĩa các kiểu dữ liệu của cột
int_columns = ["Matches Played", "Starts", "Minutes", "Gls", "Ast", "crdY", "crdR", "PrgC", "PrgP", "PrgR",
               "Cmp", "KP", "Pass into 1_3", "PPA", "CrsPA", "ProDist", "TotDist", "Tkl", "TklW", "Deff Att", "Lost", "Blocks", "Sh", "Pass", "Int",
               "Touches", "Def Pen", "Def 3rd", "Mid 3rd", "Att 3rd", "Att Pen", "Take-Ons Att",
               "Carries", "Carries 1_3", "CPA", "Mis", "Dis", "Rec", "Rec PrgR",
               "Fls", "Fld", "Off", "Crs", "Recov", "Aerl Won", "Aerl Lost"]

float_columns = ["Age", "xG", "xAG", "Gls per 90", "Ast per 90", "xG per 90", "xAG per 90", "GA90", "Save%", "CS%", "PK Save%",
                 "SoT%", "SoT per 90", "G per Sh", "Dist", "Cmp%", "ShortCmp%", "MedCmp%", "LongCmp%", "SCA", "SCA90", "GCA", "GCA90", "Succ%", "Tkld%", "Aerl Won%"]

string_columns = ["Player", "Nation", "Team", "Position"]

# Duyệt từng cột trong int_columns
for col in int_columns:
    if col in merged_df.columns:
        # Chuyển giá trị trong col sang kiểu số int
        # Nếu có giá trị không thể chuyển được, thì sẽ coerce nó thành NaN thay vì báo lỗi.
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").astype("Int64")

# Duyệt từng cột trong float_columns
for col in float_columns:
    if col in merged_df.columns:
        # Chuyển các giá trị trong col sang kiểu số float và làm tròn 2 số thập phân.
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").round(2)

# Lọc các cầu thủ có số phút thi đấu trên 90 phút
merged_df = merged_df[merged_df["Minutes"] > 90]

# Làm sạch dữ liệu quốc tịch
if "Nation" in merged_df.columns:
    merged_df["Nation"] = merged_df["Nation"].apply(extract_country_code)

# Tạo ra đường dẫn đến directory csv
csv_dir = os.path.join(base_dir, "csv")

# Tạo thư mục
#   - exist_ok=True: nếu thư mục đã tồn tại, không gây lỗi
os.makedirs(csv_dir, exist_ok=True)

# Tạo ra đường dẫn kết quả
result_path = os.path.join(csv_dir, "result.csv")

# Lưu kết quả vào thư mục đã có đường dẫn = result_path
#   - index=False: không ghi chỉ số dòng (index) vào file.
#   - encoding="utf-8-sig": mã hóa file theo chuẩn UTF-8 có BOM (giúp mở file đúng trong Excel).
#   - na_rep="N/A": tất cả giá trị NaN trong DataFrame sẽ được ghi thành chuỗi "N/A" trong file CSV.
merged_df.to_csv(result_path, index=False, encoding="utf-8-sig", na_rep="N/A")

print(f"✅ Successfully saved merged data to {result_path} with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")

# Đóng WebDriver
driver.quit()