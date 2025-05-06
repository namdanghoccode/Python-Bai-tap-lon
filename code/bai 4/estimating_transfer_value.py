from selenium import webdriver
# Selenium là thư viện tự động hóa trình duyệt (được dùng để crawl web động – các trang cần tương tác JS mới hiển thị dữ liệu).
# Webdriver dùng để điều khiển trình duyệt (như Chrome, Firefox).
from selenium.webdriver.chrome.service import Service # Dùng để tạo một dịch vụ điều khiển trình điều khiển ChromeDriver.
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options # Dùng để cấu hình tùy chọn cho Chrome, ví dụ chạy ẩn (headless), không hiển thị ảnh, tắt thông báo
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager # Tự động tải và cấu hình ChromeDriver phù hợp với phiên bản trình duyệt Chrome trên máy bạn.
import pandas as pd # Khai báo thư viện để xử lý dữ liệu dạng bảng
import os # Dùng để thao tác với hệ điều hành: đường dẫn, thư mục, file, biến môi trường
from fuzzywuzzy import fuzz, process
# fuzz: Cung cấp các hàm so sánh mức độ tương đồng giữa hai chuỗi
# process: Cung cấp các công cụ để tìm kiếm chuỗi phù hợp nhất trong một danh sách hoặc tập dữ liệu


"""
Mục đích của file này là lấy ra giá trị chuyển nhượng ước tính trên trang web để lấy dữ liệu training cho mô hình AI ở trong file 4_2.py
"""


# Thư mục gốc
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Đường dẫn đến thư mục csv
csv_dir = os.path.join(base_dir, "csv")

# Đường dẫn đến file result.csv
result_path = os.path.join(csv_dir, "result.csv")

# Hàm để thu ngắn tên nhằm tăng độ chính xác của thư viện fuzzywuzzy
def shorten_name(name):
    if name == "Manuel Ugarte Ribeiro": return "Manuel Ugarte"
    elif name == "Igor Júlio": return "Igor"
    elif name == "Igor Thiago": return "Thiago"
    elif name == "Felipe Morato": return "Morato"
    elif name == "Nathan Wood-Gordon": return "Nathan Wood"
    elif name == "Bobby Reid": return "Bobby Cordova-Reid"
    elif name == "J. Philogene": return "Jaden Philogene Bidace"

    # Nếu tên dài quá 3 từ thì chỉ lây từ đầu tiên và từ cuối cùng
    parts = name.strip().split(" ")
    return parts[0] + " " + parts[2] if len(parts) >= 3 else name

# Đọc file result.csv và gán vào DataFrame
df_players = pd.read_csv(result_path)

# Tạo từ điển ánh xạ tên người chơi rút gọn với tên và vị trí ban đầu của họ.
player_positions = dict(zip(df_players['Player'].str.strip().apply(shorten_name), df_players['Position']))

# Tạo một dictionary player_original_names ánh xạ từ tên rút gọn (sau khi xử lý bằng hàm shorten_name) sang tên gốc của cầu thủ (đã được .strip() để loại bỏ khoảng trắng thừa)
player_original_names = dict(zip(df_players['Player'].str.strip().apply(shorten_name), df_players['Player'].str.strip()))

# Tạo ra một danh sách tên cầu thủ từ phần key của player_positions
player_names = list(player_positions.keys())

# Định cấu hình trình duyệt headless
options = Options() # Tạo một đối tượng Options để cấu hình trình duyệt Chrome
options.add_argument("--headless") # Chạy trình duyệt ở chế độ ẩn (headless) — không mở cửa sổ trình duyệt thật ra
options.add_argument("--no-sandbox") # Tắt chế độ sandbox (bảo vệ) của Chrome.
options.add_argument("--disable-dev-shm-usage") # Yêu cầu Chrome không dùng /dev/shm (shared memory) làm nơi lưu trữ tạm thời.

# Khởi tạo trình điều khiển Chrome (webdriver.Chrome) với:
#    ChromeDriverManager().install() => Tự động tải và chỉ định đúng phiên bản ChromeDriver.
#    options=options => Áp dụng tất cả cấu hình vừa khai báo ở trên.
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# url cần lấy dữ liệu
base_url = "https://www.footballtransfers.com/us/players/uk-premier-league/"
urls = [f"{base_url}{i}" for i in range(1, 23)]

# danh sách dữ liệu thủ môn
data_gk = []
# danh sách dữ liệu hậu vệ
data_df = []
# danh sách dữ liệu tiền vệ
data_mf = []
# danh sách dữ liệu tiền đạo
data_fw = []

try:
    # Duyệt qua từng url
    for url in urls:
        driver.get(url) # Dùng Selenium WebDriver để mở trang web có địa chỉ là url
        print(f"Crawling: {url}")
        try:
            # Đợi bảng chuyển nhượng xuất hiện (tối đa 10 giây).
            table = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "similar-players-table"))
            )

            # Tìm các dòng tr
            rows = table.find_elements(By.TAG_NAME, "tr")

            # Lặp qua từng dòng trong bảng, tìm các ô (td)
            for row in rows:
                cols = row.find_elements(By.TAG_NAME, "td")

                # Kiểm tra cols phải có giá trị, độ dài của cols (số phần tử trong danh sách) phải lớn hơn hoặc bằng 2
                if cols and len(cols) >= 2:

                    # Lấy ra tên các cầu thủ
                    player_name = cols[1].text.strip().split("\n")[0].strip()

                    # Rút gọn tên cầu thủ trong player_name
                    shortened_player_name = shorten_name(player_name)

                    # Tìm ra giá trị chuyển nhượng ước tính
                    etv = cols[-1].text.strip() if len(cols) >= 3 else "N/A"

                    # So sánh tên cầu thủ trên website và tên cầu thủ trong danh sách player_names
                    best_match = process.extractOne(shortened_player_name, player_names, scorer=fuzz.token_sort_ratio)

                    # Kiểm tra xem best_match phải có giá tr và giá trị tương đồng phải trên 80
                    if best_match and best_match[1] >= 80:
                        # Lấy ra tên cầu thủ khớp nhất
                        matched_name = best_match[0]

                        # Lấy ra tên gốc của các cầu thủ
                        original_name = player_original_names.get(matched_name, matched_name)

                        # Lấy ra vị trí của các cầu thủ dựa vào file result.csv
                        position = player_positions.get(matched_name, "Unknown")

                        # Nếu là thủ môn thì lưu vào danh sách thủ môn
                        if "GK" in position:
                            data_gk.append([original_name, position, etv])
                        # Nếu là thủ môn thì lưu vào danh sách hậu vệ
                        if position.startswith("DF"):
                            data_df.append([original_name, position, etv])
                        # Nếu là thủ môn thì lưu vào danh sách tiền vệ
                        if position.startswith("MF"):
                            data_mf.append([original_name, position, etv])
                        # Nếu là thủ môn thì lưu vào danh sách tiền đạo
                        if position.startswith("FW"):
                            data_fw.append([original_name, position, etv])
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
finally:
    # Đóng WebDriver
    driver.quit()

# Gộp tất cả các danh sách lại
all_data = data_gk + data_df + data_mf + data_fw

# Lưu vào thư mục csv với tên file là all_estimate_transfer_fee.csv với 3 cột 'Player', 'Position', 'Price'
if all_data:
    df_all = pd.DataFrame(all_data, columns=['Player', 'Position', 'Price'])
    combined_path = os.path.join(csv_dir, "all_estimate_transfer_fee.csv")
    df_all.to_csv(combined_path, index=False)
    print(f"All results saved to '{combined_path}'")