import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime

# --- 參數設定 ---
MSEED_PATH = "../Data/0504_09_10.mseed"
INPUT_FILE = "seismic_plots/associated_catalog_with_notes.csv"
EVENT_PLOT_DIR = "seismic_plots/events"
if not os.path.exists(EVENT_PLOT_DIR):
    os.makedirs(EVENT_PLOT_DIR, exist_ok=True)

# 1. 載入波形資料 (建議預先載入，或是繪圖時再局部 read)
print("正在載入主波形檔...")
full_stream = read(MSEED_PATH)
full_stream.detrend("demean")
full_stream.filter("bandpass", freqmin=1.0, freqmax=10.0)

def plot_associated_event(event_id, all_events_df, stream, output_dir):
    """
    針對特定 event_id 繪圖，並區分不同 event_id 的 picker 顏色
    """
    # 1. 取得當前事件的紀錄
    current_picks = all_events_df[all_events_df['event_id'] == event_id].copy()
    current_picks['start_time_dt'] = pd.to_datetime(current_picks['start_time_str'])
    
    # 2. 定義繪圖時間窗 (前 5s, 後 15s)
    t_start = UTCDateTime(current_picks['start_time_dt'].min()) - 5
    t_end = UTCDateTime(current_picks['start_time_dt'].max()) + 15
    
    # 3. 找出在這個時間窗內「所有」可能的觸發紀錄 (包含其他 event_id)
    all_events_df['temp_dt'] = pd.to_datetime(all_events_df['start_time_str'])
    nearby_picks = all_events_df[
        (all_events_df['temp_dt'] >= t_start.datetime.replace(tzinfo=pd.Timestamp.max.tzinfo)) & 
        (all_events_df['temp_dt'] <= t_end.datetime.replace(tzinfo=pd.Timestamp.max.tzinfo))
    ]

    sub = stream.slice(t_start, t_end).copy()
    if len(sub) == 0: return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    full_station_names = current_picks['station'].unique()
    valid_plot_count = 0

    for i, full_name in enumerate(full_station_names):
        parts = full_name.split('.')
        sta_code = parts[1] if len(parts) >= 2 else full_name
        sta_traces = sub.select(station=sta_code)
        if not sta_traces: continue
        
        tr = sta_traces.select(component="Z")[0] if sta_traces.select(component="Z") else sta_traces[0]
        data = tr.data.astype(float)
        normed = (data - np.mean(data)) 
        if np.max(np.abs(normed)) > 0:
            normed = normed / np.max(np.abs(normed))
        
        y_offset = valid_plot_count * 1.5
        ax.plot(tr.times(), normed + y_offset, lw=1, color='black', alpha=0.8)
        
        # --- 修正後的鄰近事件判定：僅標記在該波形的上下界 ---
        # 找出該測站在這個時間窗內所有的觸發
        sta_all_nearby = nearby_picks[nearby_picks['station'] == full_name].sort_values('dt_obj')
        
        current_event_seen = False
        
        # 定義該測站波形的垂直顯示範圍 (y_offset 為中心，上下各加減 1)
        y_min = y_offset - 0.8
        y_max = y_offset + 0.8
        
        for _, row in sta_all_nearby.iterrows():
            pick_time = UTCDateTime(row['start_time_str'])
            rel_time = pick_time - tr.stats.starttime
            
            is_main_id = (row['event_id'] == event_id)
            
            # 設定顏色與標籤邏輯
            if is_main_id and not current_event_seen:
                line_color = 'red'
                line_style = '-'
                line_width = 2
                label_text = "" # 主事件通常不額外標文字，看紅線即可
                current_event_seen = True
            else:
                line_color = 'royalblue'
                line_style = '-'
                line_width = 1.5
                label_text = f"ID:{row['event_id']}"
                if is_main_id: label_text += " (Dup)"

            # 【關鍵修改】：使用 ax.plot 繪製垂直線，限制在 y_min 到 y_max 之間
            ax.plot([rel_time, rel_time], [y_min, y_max], 
                    color=line_color, linestyle=line_style, lw=line_width, alpha=0.8)
            
            # 如果有標籤文字 (鄰近事件或重複事件)，標註在虛線上方
            if label_text:
                ax.text(rel_time, y_max, label_text, color=line_color, 
                        fontsize=7, rotation=90, va='bottom', ha='center')

        # 側邊測站名稱標籤
        ax.text(-0.5, y_offset, full_name, ha='right', va='center', fontweight='bold', fontsize=9)
        valid_plot_count += 1
        ax.text(-0.5, y_offset, full_name, ha='right', va='center', fontweight='bold', fontsize=9)
        valid_plot_count += 1

    if valid_plot_count > 0:
        ax.set_title(f"Event Analysis | Current ID: {event_id}\nWindow Start: {t_start}", pad=20)
        ax.set_xlabel("Time (s)")
        ax.set_yticks([]) 
        ax.set_xlim(0, t_end - t_start)
        # 加上簡單的圖例
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='red', lw=2),
                        Line2D([0], [0], color='royalblue', lw=1.5, linestyle='--')]
        ax.legend(custom_lines, ['Current Event', 'Other Events'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"event_{event_id:03d}_distinction.png"), dpi=150)
    
    plt.close(fig)

# --- 地震關聯邏輯 (analyze_seismic_catalog 函數保持你提供的版本) ---
def analyze_seismic_catalog(file_path, time_window_sec=25, min_stations=3):
    try:
        df = pd.read_csv(file_path)
        print(df)
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
        return None

    # --- 修正處：檢查欄位名稱 ---
    # 如果 CSV 裡面叫 'start_time_str'，我們把它當作時間來源
    time_col = 'start_time_str' if 'start_time_str' in df.columns else 'start_time'
    
    if time_col not in df.columns:
        print(f"錯誤：CSV 中找不到時間欄位。現有欄位為: {df.columns.tolist()}")
        return None

    # 將時間轉換為 datetime 物件以便計算
    df['start_time_dt'] = pd.to_datetime(df[time_col], format='mixed', utc=True)
    df = df.sort_values('start_time_dt').reset_index(drop=True)
    
    # 2. 關聯標記邏輯
    df['event_id'] = -1
    current_event_id = 0
    if not df.empty:
        df.at[0, 'event_id'] = current_event_id
        for i in range(1, len(df)):
            # 使用轉換後的 datetime 物件計算秒差
            time_diff = (df.loc[i, 'start_time_dt'] - df.loc[i-1, 'start_time_dt']).total_seconds()
            if time_diff <= time_window_sec:
                df.at[i, 'event_id'] = current_event_id
            else:
                current_event_id += 1
                df.at[i, 'event_id'] = current_event_id

    # 3. 判定與分類
    event_counts = df.groupby('event_id')['station'].transform('nunique')
    
    def classify_event(count):
        if count >= 5: return f"Earthquake (Total {count} sta)"
        elif count >= min_stations: return f"Suspected (Total {count} sta)"
        else: return f"Isolated (Only {count} sta)"

    df['note'] = event_counts.apply(classify_event)

    # 4. 格式化輸出時間
    def format_microsecond(dt_obj):
        ts = dt_obj.strftime('%Y-%m-%d %H:%M:%S.%f')
        trimmed = ts.rstrip('0')
        if trimmed.endswith('.'):
            return trimmed + '00' # 依照您的需求補 .00
        return trimmed

    df['start_time_str'] = df['start_time_dt'].apply(format_microsecond)
    
    # 這裡保留繪圖需要的 event_id, note 等資訊
    output_columns = ['station', 'start_time_str', 'event_id', 'note']
    return df[output_columns]   

# --- 主程式執行區塊 ---
processed_df = analyze_seismic_catalog(INPUT_FILE)

if processed_df is not None:
    # 1. 統一複製並預處理時間，徹底解決 SettingWithCopyWarning
    catalog_df = processed_df.copy()
    catalog_df['dt_obj'] = pd.to_datetime(catalog_df['start_time_str'])

    # 2. 【核心修改】不再進行篩選，直接獲取編目中所有的 event_id
    # 不論是 Earthquake, Suspected 還是 Isolated 全部都會納入
    event_ids = sorted([int(eid) for eid in catalog_df['event_id'].unique()])
    
    print(f"從編目中找到以下所有事件 ID (全數繪製): {event_ids}")
    
    # 3. 進入繪圖迴圈
    print(f"開始為 {len(event_ids)} 個事件繪製波形圖...")
    for eid in event_ids:
        try:
            # 傳入完整的 catalog_df 供尋找 nearby picks
            # 傳入特定的 eid 進行主事件繪製
            plot_associated_event(eid, catalog_df, full_stream, EVENT_PLOT_DIR)
            print(f"  - Event {eid} 繪圖完成")
        except Exception as e:
            # 為了除錯方便，建議印出具體錯誤訊息
            print(f"  - Event {eid} 繪圖失敗: {str(e)}")

print(f"\n[系統通知] 所有事件繪圖作業已結束。")