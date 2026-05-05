import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seisbench.models as sbm
from obspy import read, UTCDateTime

# --- 1. 參數設定 ---
MSEED_PATH = "../Data/0827Yilan.mseed"
OUTPUT_DIR = "seismic_plots"
#CHUNK_LENGTH = 1800  # 每次讀取 30 分鐘 (避免記憶體溢位)
CHUNK_LENGTH = 60    # 每次讀取 1 分鐘 (單一事件)
OVERLAP = 60         # 時段間重疊 1 分鐘 (確保地震不被切斷)
STRIDE = 500         # 模型滑動步長 (500 samples = 5秒，重疊越高越準)
MODEL_TYPE = "stead" # 使用 EQTransformer

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 初始化模型 ---
print(f"正在載入 {MODEL_TYPE} 模型...")
model = sbm.EQTransformer.from_pretrained(MODEL_TYPE)
# model.to("cuda") # 如果有 GPU 請取消註釋

# --- 3. 獲取時間範圍 ---
print("正在掃描檔案資訊...")
st_header = read(MSEED_PATH, headonly=True)
global_start = min(tr.stats.starttime for tr in st_header)
global_end = max(tr.stats.endtime for tr in st_header)
print(f"資料範圍: {global_start} ~ {global_end}")

all_picks_data = []
all_detections_data = []

# --- 4. 滑動視窗處理流程 ---
current_start = global_start
while current_start < global_end:
    current_end = current_start + CHUNK_LENGTH
    print(f"\n>>> 處理時段: {current_start} ~ {current_end}")
    
    try:
        # 讀取區段資料
        stream = read(MSEED_PATH, starttime=current_start, endtime=current_end + OVERLAP)
        if len(stream) == 0:
            current_start += CHUNK_LENGTH
            continue

        # --- 資料清理 (重要：解決 Fragments 與 Misalignment 警告) ---
        stream.detrend("demean")
        stream.merge(method=1, fill_value=0) # 接合斷點
        stream.trim(current_start, current_end + OVERLAP, pad=True, fill_value=0) # 對齊時間
        
        # 確保採樣率為 100Hz (EQT 標準)
        if stream[0].stats.sampling_rate != 100.0:
            stream.resample(100.0)

        # --- 模型推論 (自動執行 60s 視窗切片) ---
        # 透過 stride 控制 60s 視窗每次移動的距離
        annotations = model.annotate(stream, batch_size=24, stride=STRIDE)
        output = model.classify(stream, batch_size=24, stride=STRIDE)

        # --- A. 繪製測站波形圖 ---
        stations = list(set([tr.stats.station for tr in stream]))
        for sta in stations:
            print(f"正在繪製測站 {sta} 的波形圖...")
            sta_st = stream.select(station=sta)
            sta_ann = annotations.select(station=sta)
            
            if len(sta_st) < 3 or len(sta_ann) < 3:
                continue

            fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"hspace": 0})
            offset = sta_ann[0].stats.starttime - sta_st[0].stats.starttime
            
            for j in range(len(sta_st)):
                axs[0].plot(sta_st[j].times(), sta_st[j].data, label=sta_st[j].stats.channel)
                if sta_ann[j].stats.channel[-1] != "N":
                    axs[1].plot(sta_ann[j].times() + offset, sta_ann[j].data, label=sta_ann[j].stats.channel)
            
            axs[0].set_title(f"Station: {sta} | Start: {current_start}")
            axs[0].legend(loc="upper right")
            axs[1].legend(loc="upper right")
            
            timestamp = current_start.strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(OUTPUT_DIR, f"{sta}_{timestamp}.png"), bbox_inches='tight')
            plt.close(fig)

        # --- B. 收集 Picks ---
        for p in output.picks:
            all_picks_data.append({
                "station": p.trace_id, "phase": p.phase,
                "peak_time": str(p.peak_time), "peak_value": p.peak_value,
                "start_time": str(p.start_time), "end_time": str(p.end_time)
            })

        # --- C. 收集 Detections ---
        for d in output.detections:
            all_detections_data.append({
                "station": d.trace_id, "start_time": str(d.start_time),
                "end_time": str(d.end_time), "peak_value": d.peak_value
            })

        # 強制清理記憶體
        del stream, annotations, output
        gc.collect()

    except Exception as e:
        print(f"區段 {current_start} 處理失敗: {e}")

    current_start += CHUNK_LENGTH

# --- 5. 輸出整合結果 (CSV) ---
if all_picks_data:
    df_picks = pd.DataFrame(all_picks_data).drop_duplicates(subset=["station", "phase", "peak_time"])
    
    # --- 新增：按照 start_time 排序 ---
    # 先確保 start_time 是時間格式以利正確排序，排序後再轉回字串格式（選用）
    df_picks['start_time'] = pd.to_datetime(df_picks['start_time'])
    df_picks = df_picks.sort_values(by='start_time').reset_index(drop=True)
    
    df_picks.to_csv(os.path.join(OUTPUT_DIR, "picks_results.csv"), index=False, encoding='utf-8-sig')

if all_detections_data:
    df_detections = pd.DataFrame(all_detections_data).drop_duplicates(subset=["station", "start_time"])
    
    # --- 新增：按照 start_time 排序 ---
    df_detections['start_time'] = pd.to_datetime(df_detections['start_time'])
    df_detections = df_detections.sort_values(by='start_time').reset_index(drop=True)
    
    df_detections.to_csv(os.path.join(OUTPUT_DIR, "detections_results.csv"), index=False, encoding='utf-8-sig')

# --- 6. 產生偵測統計報告 (summary_report.txt) ---
report_path = os.path.join(OUTPUT_DIR, "summary_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== 地震自動辨識統計報告 ===\n")
    f.write(f"處理時間: {UTCDateTime.now()}\n")
    f.write(f"資料來源: {MSEED_PATH}\n")
    f.write(f"時間範圍: {global_start} ~ {global_end}\n")
    f.write("-" * 30 + "\n\n")

    if not df_picks.empty:
        f.write(f"總計 Picks (P/S 波拾取): {len(df_picks)} 筆\n")
        f.write(f"總計 Detections (事件偵測): {len(df_detections)} 筆\n\n")

        # 統計各測站的數量
        f.write("各測站統計 (Station Statistics):\n")
        station_stats = df_picks.groupby(['station', 'phase']).size().unstack(fill_value=0)
        f.write(station_stats.to_string())
        f.write("\n\n")

        # 統計信心值分佈 (平均值)
        avg_conf = df_picks.groupby('phase')['peak_value'].mean()
        f.write("平均辨識信心度 (Average Confidence):\n")
        for phase, val in avg_conf.items():
            f.write(f"  {phase} phase: {val:.4f}\n")
    else:
        f.write("本次處理未偵測到任何地震事件。\n")

    f.write("\n=== 報告結束 ===")

print(f"偵測統計報告已列出於: {report_path}")