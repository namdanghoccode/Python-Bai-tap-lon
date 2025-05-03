from fuzzywuzzy import fuzz, process
# fuzz: Cung cấp các hàm so sánh mức độ tương đồng giữa hai chuỗi
# process: Cung cấp các công cụ để tìm kiếm chuỗi phù hợp nhất trong một danh sách hoặc tập dữ liệu
from selenium import webdriver
# Selenium là thư viện tự động hóa trình duyệt (được dùng để crawl web động – các trang cần tương tác JS mới hiển thị dữ liệu).
# Webdriver dùng để điều khiển trình duyệt (như Chrome, Firefox).
from selenium.webdriver.chrome.service import Service  # Dùng để tạo một dịch vụ điều khiển trình điều khiển ChromeDriver.
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options # Dùng để cấu hình tùy chọn cho Chrome, ví dụ chạy ẩn (headless), không hiển thị ảnh, tắt thông báo
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager # Tự động tải và cấu hình ChromeDriver phù hợp với phiên bản trình duyệt Chrome trên máy bạn.
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

# Lọc số cầu thủ có trên 900 phút thi đấu và lưu vào DataFrame
df_calc_filtered = df_calc[df_calc['Minutes'] > 900].copy()
print(f"Number of players with more than 900 minutes: {len(df_calc_filtered)}")

# Ghi danh sách cầu thủ đủ điều kiện ra file CSV mới.
filtered_path = os.path.join(base_dir, "csv", "players_over_900_minutes.csv")
df_calc_filtered.to_csv(filtered_path, index=False, encoding='utf-8-sig')
print(f"Saved filtered players to {filtered_path} with {df_calc_filtered.shape[0]} rows and {df_calc_filtered.shape[1]} columns.")

# Hàm cắt ngắn tên cầu thủ thành 2 từ đầu tiên (để tăng độ chính xác khi so khớp tên)
def shorten_name(name):
    parts = name.strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name

# Đọc lại danh sách cầu thủ đã lọc.
csv_file = os.path.join(base_dir, "csv", "players_over_900_minutes.csv")
df_players = pd.read_csv(csv_file)

# Tạo ra danh sách tên rút gọn để fuzzy match
player_names = [shorten_name(name) for name in df_players['Player'].str.strip()]

# Tạo ra dict để tra cứu phút thi đấu theo tên cầu thủ
player_minutes = dict(zip(df_players['Player'].str.strip(), df_players['Minutes']))

# Định cấu hình trình duyệt headless
options = Options() # Tạo một đối tượng Options để cấu hình trình duyệt Chrome
options.add_argument("--headless") # Chạy trình duyệt ở chế độ ẩn (headless) — không mở cửa sổ trình duyệt thật ra
options.add_argument("--no-sandbox") # Tắt chế độ sandbox (bảo vệ) của Chrome.
options.add_argument("--disable-dev-shm-usage") # Yêu cầu Chrome không dùng /dev/shm (shared memory) làm nơi lưu trữ tạm thời.

# Khởi tạo trình điều khiển Chrome (webdriver.Chrome) với:
#    ChromeDriverManager().install() => Tự động tải và chỉ định đúng phiên bản ChromeDriver.
#    options=options => Áp dụng tất cả cấu hình vừa khai báo ở trên.
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Tạo danh sách các URL từ trang 1 đến 14 của danh sách chuyển nhượng Premier League mùa 2024-2025.
base_url = "https://www.footballtransfers.com/us/transfers/confirmed/2024-2025/uk-premier-league/"
urls = [f"{base_url}{i}" for i in range(1, 15)]

# Tạo danh sách rỗng để lưu thông tin cầu thủ khớp
data = []

try:
    # Duyệt từng URL trong danh sách
    for url in urls:
        driver.get(url) # Dùng Selenium WebDriver để mở trang web có địa chỉ là url
        print(f"Scraping: {url}")
        try:
            # Đợi bảng chuyển nhượng xuất hiện (tối đa 10 giây).
            table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "transfer-table"))
            )

            # Tìm các dòng tr
            rows = table.find_elements(By.TAG_NAME, "tr")

            # Lặp qua từng dòng trong bảng, tìm các ô (td)
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")

                # Kiểm tra cols phải có giá trị, độ dài của cols (số phần tử trong danh sách) phải lớn hơn hoặc bằng 2
                if cols and len(cols) >= 2:

                    # Lấy ra tên các cầu thủ
                    player_name = cols[0].text.strip().split("\n")[0].strip()

                    # Rút gọn tên cầu thủ trong player_name để so sánh với fuzzywuzzy
                    shortened_player_name = shorten_name(player_name)

                    # Tìm ra giá trị chuyển nhượng, nếu không có dữ liệu thì hiện ra "N/A"
                    tv = cols[-1].text.strip() if len(cols) >= 3 else "N/A"

                    # So sánh tên cầu thủ trên website và tên cầu thủ trong danh sách player_names
                    best_match = process.extractOne(shortened_player_name, player_names, scorer=fuzz.token_sort_ratio)

                    # Kiểm tra xem best_match phải có giá tr và giá trị tương đồng phải trên 85
                    if best_match and best_match[1] >= 85:
                        # Lấy ra tên cầu thủ khớp nhất
                        matched_name = best_match[0]

                        # Đẩy vào danh sách data
                        data.append([player_name, tv])
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
finally:
    # Đóng WebDriver
    driver.quit()


# Lưu file tên player_transfer_fee.csv vào thư mục csv
if data:
    df_tv = pd.DataFrame(data, columns=['Player', 'Price'])
    df_tv.to_csv(os.path.join(base_dir, "csv", "player_transfer_fee.csv"), index=False)
    print(f"Results saved to '{os.path.join(base_dir, 'csv', 'player_transfer_fee.csv')}'")
else:
    print("No matching players found.")