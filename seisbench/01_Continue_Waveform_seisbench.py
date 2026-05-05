import os
import gc
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seisbench.models as sbm # type: ignore
from obspy import read, UTCDateTime # type: ignore

# --- 1. 參數設定 ---
MSEED_PATH = "../Data/0504.mseed"
OUTPUT_DIR = "seismic_plots"
CHUNK_LENGTH = 1800  # 每段處理 30 分鐘
OVERLAP = 60         # 重疊 1 分鐘
MODEL_TYPE = "stead" 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. 初始化模型 ---
model = sbm.EQTransformer.from_pretrained(MODEL_TYPE)
# model.to("cuda") # 如果有 GPU 請開啟

# --- 3. 獲取時間範圍 ---
st_header = read(MSEED_PATH, headonly=True)
global_start = min(tr.stats.starttime for tr in st_header)
global_end = max(tr.stats.endtime for tr in st_header)

all_picks_data = []
all_detections_data = []

# --- 4. 滑動視窗處理流程 ---
current_start = global_start
while current_start < global_end:
    current_end = current_start + CHUNK_LENGTH
    print(f"\n>>> 正在處理區段: {current_start} ~ {current_end}")
    
    try:
        # 讀取特定時段資料
        stream = read(MSEED_PATH, starttime=current_start, endtime=current_end + OVERLAP)
        
        if len(stream) == 0:
            current_start += CHUNK_LENGTH
            continue

        # --- [關鍵新增] 資料預處理與清理 ---
        stream.detrend("demean")      # 去直流分量
        
        # A. 合併破碎片段：解決 "Fragments shorter than input" 警告
        # method=1 會嘗試插值接合，fill_value=0 則是在大缺口補零
        stream.merge(method=1, fill_value=0)
        
        # B. 強制對齊時間視窗：確保所有 Trace 起訖點完全一致
        stream.trim(current_start, current_end + OVERLAP, pad=True, fill_value=0)
        
        # C. 檢查採樣率 (EQTransformer 預期 100Hz，SeisBench 會自動轉，但手動轉更穩)
        # stream.interpolate(sampling_rate=100) 

        # --- 模型推論 ---
        annotations = model.annotate(stream)
        output = model.classify(stream)

        # D. 繪製並儲存圖片
        # 注意：stream 可能包含多個測站，每 3 條 trace (Z,N,E) 繪製一張圖
        stations = list(set([tr.stats.station for tr in stream]))
        for sta in stations:
            sta_st = stream.select(station=sta)
            sta_ann = annotations.select(station=sta)
            
            if len(sta_st) < 3 or len(sta_ann) < 3:
                continue # 跳過分量不齊全的測站

            timestamp = current_start.strftime("%Y%m%d_%H%M%S")
            fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"hspace": 0})
            
            offset = sta_ann[0].stats.starttime - sta_st[0].stats.starttime
            
            for j in range(len(sta_st)):
                axs[0].plot(sta_st[j].times(), sta_st[j].data, label=sta_st[j].stats.channel)
                if sta_ann[j].stats.channel[-1] != "N":
                    axs[1].plot(sta_ann[j].times() + offset, sta_ann[j].data, label=sta_ann[j].stats.channel)
            
            axs[0].set_title(f"Station: {sta} | Time: {current_start}")
            axs[0].legend(loc="upper right")
            axs[1].legend(loc="upper right")
            
            save_path = os.path.join(OUTPUT_DIR, f"{sta}_{timestamp}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)

        # E. 收集數據
        for p in output.picks:
            all_picks_data.append({
                "station": p.trace_id, "phase": p.phase,
                "peak_time": p.peak_time, "peak_value": p.peak_value,
                "start_time": p.start_time, "end_time": p.end_time
            })

        for d in output.detections:
            all_detections_data.append({
                "station": d.trace_id, "start_time": d.start_time,
                "end_time": d.end_time, "peak_value": d.peak_value
            })

        # 釋放記憶體
        del stream, annotations, output
        gc.collect()

    except Exception as e:
        print(f"處理區段 {current_start} 發生錯誤: {e}")

    current_start += CHUNK_LENGTH

# --- 5. 輸出整合結果 ---
if all_picks_data:
    df_picks = pd.DataFrame(all_picks_data).drop_duplicates(subset=["station", "phase", "peak_time"])
    df_picks.to_csv(os.path.join(OUTPUT_DIR, "picks_results.csv"), index=False, encoding='utf-8-sig')
    print(f"總計 Picks: {len(df_picks)}")

if all_detections_data:
    df_detections = pd.DataFrame(all_detections_data).drop_duplicates(subset=["station", "start_time"])
    df_detections.to_csv(os.path.join(OUTPUT_DIR, "detections_results.csv"), index=False, encoding='utf-8-sig')
    print(f"總計 Detections: {len(df_detections)}")

print("\n--- 任務全部完成 ---")