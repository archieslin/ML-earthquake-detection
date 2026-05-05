import pandas as pd
import numpy as np
import os

def analyze_seismic_catalog(file_path, time_window_sec=25, min_stations=3):
    # 1. 載入資料
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
        return None

    # 處理時間與排序：確保使用 start_time_str 轉換
    time_col = 'start_time_str' if 'start_time_str' in df.columns else 'start_time'
    df['start_time'] = pd.to_datetime(df[time_col], format='mixed', utc=True)
    df = df.sort_values('start_time').reset_index(drop=True)
    
    # 2. 關聯標記邏輯 (恢復為純粹 25s 時間窗分離)
    df['event_id'] = -1
    current_event_id = 0

    if not df.empty:
        df.at[0, 'event_id'] = current_event_id
        for i in range(1, len(df)):
            # 計算與前一筆紀錄的時間差
            time_diff = (df.loc[i, 'start_time'] - df.loc[i-1, 'start_time']).total_seconds()
            
            if time_diff <= time_window_sec:
                # 若在視窗內，歸類為同一個事件
                df.at[i, 'event_id'] = current_event_id
            else:
                # 若超過視窗，開啟新事件
                current_event_id += 1
                df.at[i, 'event_id'] = current_event_id

    # 3. 判定註記 (統計不重複測站數)
    event_counts = df.groupby('event_id')['station'].transform('nunique')

    def classify_event(count):
        if count >= 5: 
            return f"Earthquake (Total {count} sta)"
        elif count >= min_stations: 
            return f"Suspected (Total {count} sta)"
        else: 
            return f"Isolated (Total {count} sta)"

    df['note'] = event_counts.apply(classify_event)

    # 4. 格式化輸出時間 (補 .00 邏輯)
    def format_microsecond(dt_obj):
        ts = dt_obj.strftime('%Y-%m-%d %H:%M:%S.%f')
        trimmed = ts.rstrip('0')
        return trimmed + '00' if trimmed.endswith('.') else trimmed

    df['start_time_str'] = df['start_time'].apply(format_microsecond)
    
    # 只挑選需要輸出的欄位
    output_columns = ['station', 'start_time_str', 'end_time', 'peak_value', 'event_id', 'note']
    return df[output_columns]

# --- 主程式執行 ---
INPUT_FILE = "seismic_plots/detections_results.csv"
OUTPUT_CSV = "seismic_plots/associated_catalog_with_notes.csv"
OUTPUT_REPORT = "seismic_plots/association_report.txt"

processed_df = analyze_seismic_catalog(INPUT_FILE)

if processed_df is not None: 
    # 1. 儲存完整資料
    processed_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    # 2. 分類篩選與去重 (用於生成報告)
    # 使用英文標籤匹配
    events_only = processed_df[processed_df['note'].str.contains("Earthquake")]
    suspected_only = processed_df[processed_df['note'].str.contains("Suspected")]
    isolated_only = processed_df[processed_df['note'].str.contains("Isolated")]

    summary_events = events_only.drop_duplicates(subset=['event_id'], keep='first')
    summary_suspected = suspected_only.drop_duplicates(subset=['event_id'], keep='first')
    summary_isolated = isolated_only.drop_duplicates(subset=['event_id'], keep='first')

    # 3. 統計數據
    total_events = events_only['event_id'].nunique()
    total_suspected = suspected_only['event_id'].nunique()
    total_isolated = isolated_only['event_id'].nunique()
    
    # 4. 準備報告內容
    report_content = [
        "分析完成！",
        "------------------------------------",
        f"原始觸發總數：{len(processed_df)} 筆",
        f"【確定】地震事件數 (>=5 站)：{total_events} 個",
        f"【疑似】地震事件數 (3-4 站)：{total_suspected} 個",
        f"【判定】孤立訊號群數 (<3 站)：{total_isolated} 個",
        f"完整原始編目儲存至：{OUTPUT_CSV}",
        "\n--- 地震事件列表 (>=5 站) ---",
        summary_events.to_string(index=False) if not summary_events.empty else "無符合條件之地震事件",
        "\n--- 疑似地震列表 (3-4 站) ---",
        summary_suspected.to_string(index=False) if not summary_suspected.empty else "無疑似地震",
        "\n--- 孤立訊號列表 (<3 站) ---",
        summary_isolated.to_string(index=False) if not summary_isolated.empty else "無孤立訊號"
    ]

    full_report_text = "\n".join(report_content)
    print(full_report_text)
    
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(full_report_text)
    
    print(f"\n[系統通知] 統計報告已同步儲存至：{OUTPUT_REPORT}")