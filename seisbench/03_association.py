import pandas as pd
import numpy as np
import os

def analyze_seismic_catalog(file_path, time_window_sec=25, min_stations=3):
    # --- 前段載入與關聯邏輯保持不變 ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
        return None

    df['start_time'] = pd.to_datetime(df['start_time'], format='mixed', utc=True)
    df = df.sort_values('start_time').reset_index(drop=True)
    
    df['event_id'] = -1
    current_event_id = 0
    if not df.empty:
        df.at[0, 'event_id'] = current_event_id
        for i in range(1, len(df)):
            time_diff = (df.loc[i, 'start_time'] - df.loc[i-1, 'start_time']).total_seconds()
            if time_diff <= time_window_sec:
                df.at[i, 'event_id'] = current_event_id
            else:
                current_event_id += 1
                df.at[i, 'event_id'] = current_event_id

    # 統計每個 event_id 包含的唯一測站數量
    event_counts = df.groupby('event_id')['station'].transform('nunique')

    # 定義分類邏輯函數
    def classify_event(count):
        if count >= 5:
            return f"地震事件 (共 {count} 測站)"
        elif count >= min_stations:  # 這裡 min_stations 預設為 3
            return f"疑似地震 (共 {count} 測站)"
        else:
            return f"孤立訊號 (僅 {count} 測站)"

    df['note'] = event_counts.apply(classify_event)

    # --- 【修改重點】格式化時間：確保若無微秒則補上 .0 ---
    # 先格式化到微秒精度
    df['start_time_str'] = df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # 這裡使用自定義函數處理：去掉末尾多餘的 0，但如果變成以 . 結尾，則補上 0
    def format_microsecond(time_str):
        trimmed = time_str.rstrip('0')
        if trimmed.endswith('.'):
            return trimmed + '00'
        return trimmed

    df['start_time_str'] = df['start_time_str'].apply(format_microsecond)
    
    output_columns = ['station', 'start_time_str', 'end_time', 'peak_value', 'event_id', 'note']
    return df[output_columns]

# --- 主程式執行 (包含 drop_duplicates 邏輯) ---
INPUT_FILE = "seismic_plots/detections_results.csv"
OUTPUT_CSV = "seismic_plots/associated_catalog_with_notes25.csv"
OUTPUT_REPORT = "seismic_plots/association_report25.txt"

processed_df = analyze_seismic_catalog(INPUT_FILE)

if processed_df is not None: 
    # 1. 儲存完整資料
    processed_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    # 2. 篩選與去重
    events_only = processed_df[processed_df['note'].str.contains("地震事件")]
    suspected_only = processed_df[processed_df['note'].str.contains("疑似地震")]
    isolated_only = processed_df[processed_df['note'].str.contains("孤立")]

    summary_events = events_only.drop_duplicates(subset=['event_id'], keep='first')
    summary_suspected = suspected_only.drop_duplicates(subset=['event_id'], keep='first')
    summary_isolated = isolated_only.drop_duplicates(subset=['event_id'], keep='first')

    # 3. 統計數據
    total_events = events_only['event_id'].nunique()
    total_suspected = suspected_only['event_id'].nunique()
    total_isolated = isolated_only['event_id'].nunique()
    
    # 4. 準備報告
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
    
    print(f"\n[系統通知] 報告已產出。時間格式檢查：12:00:00.000 -> 12:00:00.0")