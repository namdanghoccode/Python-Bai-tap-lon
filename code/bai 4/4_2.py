import pandas as pd # Khai báo thư viện để xử lý dữ liệu dạng bảng
import numpy as np
import os # Dùng để thao tác với hệ điều hành: đường dẫn, thư mục, file, biến môi trường
import re
from fuzzywuzzy import process, fuzz
# fuzz: Cung cấp các hàm so sánh mức độ tương đồng giữa hai chuỗi
# process: Cung cấp các công cụ để tìm kiếm chuỗi phù hợp nhất trong một danh sách hoặc tập dữ liệu
from sklearn.model_selection import train_test_split # Tách dữ liệu thành tập huấn luyện và kiểm tra (train/test).
from sklearn.linear_model import LinearRegression # Mô hình hồi quy tuyến tính.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# StandardScaler: Chuẩn hóa dữ liệu dạng số về phân phối chuẩn (mean = 0, std = 1).
# OneHotEncoder: Biến đổi các biến phân loại (categorical features) thành các vector nhị phân (one-hot vectors)
from sklearn.compose import ColumnTransformer
# Công cụ quan trọng khi áp dụng nhiều bước tiền xử lý khác nhau cho các cột dữ liệu khác nhau trong cùng một bảng dữ liệu
from sklearn.pipeline import Pipeline # Xây dựng một chuỗi xử lý đầy đủ gồm: tiền xử lý => huấn luyện mô hình => dự đoán.
from sklearn.metrics import mean_squared_error, r2_score
# mean_squared_error: Tính sai số bình phương trung bình giữa giá trị dự đoán và giá trị thực tế.
# r2_score: Tính hệ số xác định R², đo lường mức độ mô hình giải thích được phương sai của dữ liệu

# Thư mục gốc nơi các file sẽ được lưu vào
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Đường dẫn đến directory csv
csv_dir = os.path.join(base_dir, "csv")

# Đường dẫn đến file result.csv
result_path = os.path.join(csv_dir, "result.csv")

# Các cột trong file csv
standard_output_columns = [
    'Player', 'Team', 'Nation', 'Position', 'Actual_Transfer_Value_M', 'Predicted_Transfer_Value_M'
]

# Đường dẫn đến file all_estimate_transfer_fee
etv_path = os.path.join(csv_dir, 'all_estimate_transfer_fee.csv')

# Cấu hình cho các vị trí
#   - position_filter: Tên các vị trí
#   - etv_path: Đường dẫn đến file csv
#   - features: Các dữ liệu đặc trưng để biểu thị phong độ của từng vị trí
#   - important_features: Những dữ liệu quan trọng có trọng số cao hơn của từng vị trí
positions_config = {
    'GK': {
        'position_filter': 'GK',
        'etv_path': etv_path,
        'features': [
            'Save%', 'CS%', 'GA90', 'Minutes', 'Age', 'PK Save%', 'Team', 'Nation'
        ],
        'important_features': ['Save%', 'CS%', 'PK Save%']
    },
    'DF': {
        'position_filter': 'DF',
        'etv_path': etv_path,
        'features': [
            'Tkl', 'TklW', 'Int', 'Blocks', 'Recov', 'Minutes', 'Team', 'Age', 'Nation', 'Aerl Won%',
            'Aerl Won', 'Cmp', 'Cmp%', 'PrgP', 'LongCmp%', 'Carries', 'Touches', 'Dis', 'Mis'
        ],
        'important_features': ['Tkl', 'TklW', 'Int', 'Blocks', 'Aerl Won%', 'Aerl Won', 'Recov']
    },
    'MF': {
        'position_filter': 'MF',
        'etv_path': etv_path,
        'features': [
            'Cmp%', 'KP', 'PPA', 'PrgP', 'Tkl', 'Ast', 'SCA', 'Touches', 'Minutes', 'Team', 'Age', 'Nation',
            'Pass into 1_3', 'xAG', 'Carries 1_3', 'ProDist', 'Rec', 'Mis', 'Dis'
        ],
        'important_features': ['KP', 'PPA', 'PrgP', 'SCA', 'xAG', 'Pass into 1_3', 'Carries 1_3']
    },
    'FW': {
        'position_filter': 'FW',
        'etv_path': etv_path,
        'features': [
            'Gls', 'Ast', 'Gls per 90', 'xG per 90', 'SoT%', 'G per Sh', 'SCA90', 'GCA90',
            'PrgC', 'Carries 1_3', 'Aerl Won%', 'Team', 'Age', 'Minutes'
        ],
        'important_features': ['Gls', 'Ast', 'Gls per 90', 'xG per 90', 'SCA90', 'GCA90']
    }
}

# Hàm rút ngắn tên để có thể so sánh dễ dàng hơn với fuzzywuzzy
def shorten_name(name):
    if not isinstance(name, str):
        return ""
    parts = name.strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name

# Chuyển đổi giá trị cầu thủ từ dạng "€2.5M" sang thành dạng "2_500_000"
def parse_etv(etv_text):
    # Kiểm tra nếu etv_text là giá trị thiếu (NaN) hoặc là "N/A" hay chuỗi rỗng "" => trả về np.nan (not a number)
    if pd.isna(etv_text) or etv_text in ["N/A", ""]:
        return np.nan
    try:
        # Loại bỏ ký hiệu tiền tệ € hoặc £ rồi loại bỏ khoảng trắng đầu/cuối, chuyển chữ thường thành chữ hoa (đảm bảo bắt được M hoặc K chuẩn xác)
        etv_text = re.sub(r'[€£]', '', etv_text).strip().upper()

        # Xác định hệ số nhân dựa trên đơn vị
        #   - M => nhân 1000000
        #   - K => nhân 1000
        multiplier = 1000000 if 'M' in etv_text else 1000 if 'K' in etv_text else 1

        # Bỏ M, K ra khỏi chuỗi, rồi chuyển phần còn lại thành float, sau đó nhân với hệ số
        value = float(re.sub(r'[MK]', '', etv_text)) * multiplier

        # Trả về giá trị
        return value

    # Nếu xảy ra lỗi khi chuyển đổi (ví dụ chuỗi không hợp lệ), trả về np.nan.
    except (ValueError, TypeError):
        return np.nan

# Tìm tên trong choices gần giống nhất với name (so khớp mờ), chỉ trả kết quả nếu điểm tương đồng ≥ 90
def fuzzy_match_name(name, choices, score_threshold=90):

    # Nếu name không phải chuỗi (ví dụ: NaN, số...), trả về None.
    if not isinstance(name, str):
        return None, None

    # Đơn giản hóa tên, sau đó, chuyển tất cả về chữ thường để dễ so sánh.
    shortened_name = shorten_name(name).lower()

    # Làm điều tương tự cho tất cả tên trong choices: chuẩn hóa + chữ thường.
    shortened_choices = [shorten_name(c).lower() for c in choices if isinstance(c, str)]

    # Dùng fuzzywuzzy.process.extractOne() để so khớp mờ.
    #   - corer=fuzz.token_sort_ratio giúp so sánh nội dung dù thứ tự từ khác nhau
    #   - score_cutoff: chỉ nhận kết quả nếu điểm khớp ≥ ngưỡng cho trước
    match = process.extractOne(
        shortened_name,
        shortened_choices,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=score_threshold
    )

    # Nếu có kết quả khớp: tìm lại tên gốc trong choices theo chỉ số tương ứng, trả về tên gốc và điểm khớp.
    if match is not None:
        matched_idx = shortened_choices.index(match[0])
        return choices[matched_idx], match[1]

    # Không có tên nào đủ điểm khớp → trả về None.
    return None, None

# Xử lý dữ liệu cầu thủ theo vị trí, làm sạch và biến đổi dữ liệu, áp dụng hồi quy tuyến tính để dự đoán giá trị chuyển nhượng ước tính (ETV)
def process_position(position, config):
    # Load data
    try:
        # Đọc dữ liệu từ 2 file CSV: một là kết quả thống kê cầu thủ (result.csv), một là dữ liệu giá trị chuyển nhượng website ước tính (etv.csv)
        df_result = pd.read_csv(result_path)
        df_etv = pd.read_csv(config['etv_path'])
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy tệp cho {position} - {e}")
        return None, None

    # Lấy vị trí chính của cầu thủ bằng cách tách chuỗi và giữ phần đầu tiên.
    df_result['Primary_Position'] = df_result['Position'].astype(str).str.split(r'[,/]').str[0].str.strip()

    # Lọc ra chỉ những cầu thủ đúng với position_filter trong config (ví dụ chỉ lấy hậu vệ nếu đang xử lý 'DF')
    df_result = df_result[df_result['Primary_Position'].str.upper() == config['position_filter'].upper()].copy()

    # Lấy ra danh sách tên các cầu thủ
    player_names = df_etv['Player'].dropna().tolist()

    # Khởi tạo các cột đẻ so sánh theo tên
    df_result['Matched_Name'] = None
    df_result['Match_Score'] = None
    df_result['ETV'] = np.nan

    # Lặp qua từng hàng của DataFrame df_result
    for idx, row in df_result.iterrows():
        # So khớp mờ (fuzzy matching) giữa tên cầu thủ trong cột Player của hàng hiện tại (row['Player']) và một danh sách hoặc tập hợp tên cầu thủ (player_names)
        matched_name, score = fuzzy_match_name(row['Player'], player_names)
        if matched_name:
            # Gán giá trị matched_name vào cột Matched_Name của DataFrame df_result tại chỉ số hàng hiện tại (idx)
            df_result.at[idx, 'Matched_Name'] = matched_name

            # Gán giá trị score (điểm tương đồng từ so khớp mờ) vào cột Match_Score của DataFrame df_result tại chỉ số hàng hiện tại (idx)
            df_result.at[idx, 'Match_Score'] = score

            # Lọc DataFrame df_etv để tìm các hàng mà cột Player khớp chính xác với matched_name
            matched_row = df_etv[df_etv['Player'] == matched_name]
            if not matched_row.empty:
                # Lấy giá trị từ cột Price của hàng đầu tiên trong matched_row
                etv_value = parse_etv(matched_row['Price'].iloc[0])

                # Gán giá trị etv_value vào cột ETV của DataFrame df_result tại chỉ số hàng hiện tại (idx).
                df_result.at[idx, 'ETV'] = etv_value

    # Tạo một DataFrame mới df_filtered bằng cách lọc các hàng từ df_result mà cột Matched_Name không rỗng
    df_filtered = df_result[df_result['Matched_Name'].notna()].copy()

    # Loại bỏ các hàng trùng lặp trong df_filtered dựa trên cột Matched_Name, giữ lại hàng đầu tiên xuất hiện cho mỗi giá trị duy nhất trong cột Matched_Name và xóa các hàng trùng lặp khác
    df_filtered = df_filtered.drop_duplicates(subset='Matched_Name')

    # Tạo danh sách các cầu thủ không được khớp (tức là Matched_Name là NaN)
    unmatched = df_result[df_result['Matched_Name'].isna()]['Player'].dropna().tolist()
    if unmatched:
        # In ra các cầu thủ không khớp, do không còn chơi trong giải NHA
        print(f"Cầu thủ {position} không khớp: {len(unmatched)} cầu thủ không được khớp.")
        print(unmatched)

    # Lấy danh sách các đặc trưng (features) từ một biến config
    features = config['features']

    # Xác định biến mục tiêu (target) cho phân tích hoặc mô hình, ở đây là cột ETV
    target = 'ETV'

    # Xử lý giá trị thiếu trong các cột đặc trưng (features) của DataFrame df_filtered.
    for col in features:
        # Nếu cột là 'Team' hoặc 'Nation'
        if col in ['Team', 'Nation']:

            # Giá trị thiếu (NaN) được thay bằng chuỗi 'Unknown'
            df_filtered[col] = df_filtered[col].fillna('Unknown')

        else:
            # Chuyển đổi giá trị trong cột thành kiểu số, các giá trị không chuyển đổi được sẽ thành NaN
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

            # Tính giá trị trung vị (median_value) của cột
            median_value = df_filtered[col].median()

            # Thay giá trị thiếu bằng trung vị; nếu trung vị là NaN, thay bằng 0.
            df_filtered[col] = df_filtered[col].fillna(median_value if not pd.isna(median_value) else 0)

    # Tạo danh sách numeric_features chứa các cột số (loại bỏ 'Team' và 'Nation')
    numeric_features = [col for col in features if col not in ['Team', 'Nation']]
    for col in numeric_features:

        # Áp dụng biến đổi log cho các cột số để giảm độ lệch (skewness) và xử lý giá trị âm
        #   - clip(lower=0) đảm bảo giá trị không nhỏ hơn 0 (tránh log của số âm)
        #   - np.log1p áp dụng hàm log(1 + x) để biến đổi dữ liệu, giúp phân phối dữ liệu gần với phân phối chuẩn hơn
        df_filtered[col] = np.log1p(df_filtered[col].clip(lower=0))


    # Điều chỉnh trọng số của một số đặc trưng để tăng hoặc giảm tầm quan trọng của chúng.
    for col in config['important_features']:
        if col in df_filtered.columns:
            # Nhân giá trị của cột với 2.0 để tăng tầm quan trọng
            df_filtered[col] = df_filtered[col] * 2.0
    if 'Minutes' in df_filtered.columns:
        # Nhân giá trị với 1.5.
        df_filtered['Minutes'] = df_filtered['Minutes'] * 1.5
    if 'Age' in df_filtered.columns:
        # Nhân giá trị với 0.5 (giảm tầm quan trọng của tuổi)
        df_filtered['Age'] = df_filtered['Age'] * 0.5

    # Tạo DataFrame df_ml để huấn luyện mô hình, loại bỏ các hàng thiếu giá trị mục tiêu (ETV)
    df_ml = df_filtered.dropna(subset=[target]).copy()
    if df_ml.empty:
        print(f"Lỗi: Không có dữ liệu ETV hợp lệ cho {position}.")
        return None, unmatched

    # Ma trận đặc trưng (các cột trong features)
    X = df_ml[features]
    # Vector mục tiêu (cột ETV)
    y = df_ml[target]

    # Chia dữ liệu thành tập huấn luyện và kiểm tra.
    if len(df_ml) > 5:
        # Nếu số mẫu lớn hơn 5, dùng train_test_split để chia dữ liệu: 80% cho huấn luyện (X_train, y_train), 20% cho kiểm tra (X_test, y_test), với random_state=42 để đảm bảo tái lập
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        print(f"Cảnh báo: Không đủ dữ liệu cho {position} để chia tập huấn luyện/kiểm tra.")
        X_train, y_train = X, y
        X_test, y_test = X, y

    # Xây dựng và huấn luyện pipeline học máy để dự đoán ETV

    # Tạo danh sách categorical_features chứa các cột danh mục ('Team', 'Nation')
    categorical_features = [col for col in features if col in ['Team', 'Nation']]

    # ColumnTransformer xử lý riêng các cột
    preprocessor = ColumnTransformer(
        transformers=[
            # Cột số (numeric_features): Chuẩn hóa bằng StandardScaler (đưa về mean=0, std=1)
            ('num', StandardScaler(), numeric_features),
            # Cột danh mục (categorical_features): Mã hóa one-hot bằng OneHotEncoder, bỏ qua các giá trị chưa thấy (handle_unknown='ignore')
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # Pipeline kết hợp tiền xử lý (preprocessor) và mô hình hồi quy tuyến tính (LinearRegression)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Huấn luyện pipeline trên tập huấn luyện.
    pipeline.fit(X_train, y_train)

    # Đánh giá mô hình trên tập kiểm tra (nếu có)
    if len(X_test) > 0:
        # Dự đoán giá trị ETV trên tập kiểm tra (y_pred)
        y_pred = pipeline.predict(X_test)
        # Tính RMSE (Root Mean Squared Error) để đo sai số trung bình
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # Tính R² Score để đo mức độ giải thích của mô hình
        r2 = r2_score(y_test, y_pred)

    # Dự đoán ETV cho toàn bộ df_filtered và lưu vào cột Predicted_Transfer_Value.
    df_filtered['Predicted_Transfer_Value'] = pipeline.predict(df_filtered[features])
    # Giới hạn giá trị dự đoán trong khoảng [100,000, 200,000,000] bằng clip
    df_filtered['Predicted_Transfer_Value'] = df_filtered['Predicted_Transfer_Value'].clip(lower=100_000, upper=200_000_000)
    # Chuyển đổi giá trị dự đoán và thực tế (ETV) sang triệu euro, làm tròn 2 chữ số thập phân, lưu vào cột Predicted_Transfer_Value_M và Actual_Transfer_Value_M
    df_filtered['Predicted_Transfer_Value_M'] = (df_filtered['Predicted_Transfer_Value'] / 1_000_000).round(2)
    df_filtered['Actual_Transfer_Value_M'] = (df_filtered['ETV'] / 1_000_000).round(2)

    # Đảm bảo tất cả cột trong standard_output_columns (danh sách cột đầu ra chuẩn) tồn tại trong df_filtered
    for col in standard_output_columns:
        if col not in df_filtered.columns:
            # Nếu cột thiếu, thêm cột với giá trị mặc định, NaN cho Actual_Transfer_Value_M và Predicted_Transfer_Value_M, Chuỗi rỗng ('') cho các cột khác
            df_filtered[col] = np.nan if col in ['Actual_Transfer_Value_M', 'Predicted_Transfer_Value_M'] else ''

    # Thêm cột Position với giá trị position (vị trí cầu thủ).
    df_filtered['Position'] = position

    # Tạo DataFrame result chứa các cột trong standard_output_columns
    result = df_filtered[standard_output_columns].copy()

    # Tạo danh sách numeric_features_no_age (các cột số trừ 'Age')
    numeric_features_no_age = [col for col in numeric_features if col != 'Age']
    for col in numeric_features_no_age:
        # Áp dụng np.expm1 (nghịch đảo của log1p) và làm tròn 2 chữ số
        if col in result.columns:
            result[col] = np.expm1(result[col]).round(2)
    if 'Age' in result.columns:
        # Hoàn nguyên bằng np.expm1, làm tròn thành số nguyên.
        result['Age'] = np.expm1(result['Age']).round(0)
        # Thay giá trị thiếu bằng trung vị tuổi và chuyển sang kiểu int
        median_age = result['Age'].median()
        result['Age'] = result['Age'].fillna(median_age).astype(int)

    # Trả về kết quả
    #   - result: DataFrame chứa dữ liệu đầu ra đã xử lý (các cột chuẩn, giá trị hoàn nguyên)
    #   - unmatched: Danh sách các cầu thủ không được khớp.
    return result, unmatched

# Lưu trữ các DataFrame kết quả từ từng vị trí cầu thủ.
all_results = []
# Lưu trữ thông tin về các cầu thủ không được khớp từ tất cả các vị trí.
all_unmatched = []

# Vòng lặp xử lý từng vị trí cầu thủ được định nghĩa trong positions_config
for position, config in positions_config.items():
    print(f"\nĐang xử lý {position}...")
    # Gọi hàm process_position(position, config) để xử lý dữ liệu cho vị trí đó
    result, unmatched = process_position(position, config)
    # Nếu result không phải None, thêm vào all_results
    if result is not None:
        all_results.append(result)
    # Nếu unmatched không rỗng, tạo danh sách các cặp (position, player) và thêm vào all_unmatched
    if unmatched:
        all_unmatched.extend([(position, player) for player in unmatched])

# Combine and save results
if all_results:
    try:
        # Nối tất cả DataFrame trong all_results theo chiều dọc thành một DataFrame duy nhất (combined_results)
        combined_results = pd.concat(all_results, ignore_index=True)
        # Sắp xếp combined_results theo cột Predicted_Transfer_Value_M (giá trị chuyển nhượng dự đoán, tính bằng triệu euro) theo thứ tự giảm dần (từ cao đến thấp)
        combined_results = combined_results.sort_values(by='Predicted_Transfer_Value_M', ascending=False)
        # Lưu combined_results vào file ml_transfer_values_linear.csv trong thư mục được chỉ định bởi csv_dir
        combined_results.to_csv(os.path.join(csv_dir, 'ml_transfer_values_linear.csv'), index=False)
        print(f"Các cầu thủ khớp đã được lưu vào '{os.path.join(csv_dir, 'ml_transfer_values_linear.csv')}'")
    except ValueError as e:
        print(f"Lỗi khi nối: {e}")
        print("Các cột trong mỗi DataFrame kết quả:")
        for i, df in enumerate(all_results):
            print(f"Cột vị trí {list(positions_config.keys())[i]}: {df.columns.tolist()}")

# In các cầu thủ không còn chơi tại Ngoại Hạng Anh nữa, mặc dù có tên trong result.csv
if all_unmatched:
    print("\nDanh sách cầu thủ không khớp:")
    for position, player in all_unmatched:
        print(f"Vị trí: {position}, Cầu thủ: {player}")