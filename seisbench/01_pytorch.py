import os
import gc
import pandas as pd
import obspy
from obspy import read
import seisbench
import seisbench.models as sbm

# 強制啟動備用倉庫
seisbench.use_backup_repository()

# --- 1. 參數設定 ---
MSEED_PATH = "../Data/211024_M68.mseed"
OUTPUT_DIR = "seismic_plots_pytorch"

# 微震特搜模式參數
CHUNK_LENGTH = 300   # 每次大範圍掃描 5 分鐘
OVERLAP = 10         # 【修正】時間重疊 10 秒即可（單位是秒！不是點數！）
STRIDE = 100         # 內層推論的滑動步長
PICK_THRESHOLD = 0.08

# 使用對微震更敏感的 PhaseNet
MODEL_CLASS = sbm.PhaseNet 
MODEL_TYPE = "instance" 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 初始化模型 ---
print(f"正在載入 {MODEL_CLASS.__name__} ({MODEL_TYPE}) 模型...")
model = MODEL_CLASS.from_pretrained(MODEL_TYPE)

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
        # 讀取區段資料 (只加上合理的 10 秒重疊)
        stream = read(MSEED_PATH, starttime=current_start, endtime=current_end + OVERLAP)
        if len(stream) == 0:
            current_start += CHUNK_LENGTH
            continue

        # 基本前置清理與濾波
        stream.detrend("demean")
        stream.detrend("linear")
        stream.filter("bandpass", freqmin=2.0, freqmax=25.0, zerophase=True)
        stream.merge(method=1, fill_value=0)
        stream.trim(current_start, current_end + OVERLAP, pad=True, fill_value=0)
        if stream[0].stats.sampling_rate != 100.0:
            stream.resample(100.0)

        # =================================================================
        # 【記憶體救星】內層迴圈：將大區段手動切成「1分鐘 (60秒)」的小碎塊送進 AI
        # =================================================================
        sub_chunk_size = 60  
        sub_overlap = 4      

        t_start = current_start
        while t_start < current_end:
            t_end = t_start + sub_chunk_size

            # 切割出超小片段 (View)
            stream2 = stream.slice(starttime=t_start, endtime=t_end + sub_overlap)
            
            if len(stream2) > 0:
                # 讓 AI 只處理這 60 秒的微小矩陣
                output = model.classify(stream2, stride=STRIDE, 
                                        p_threshold=PICK_THRESHOLD, 
                                        s_threshold=PICK_THRESHOLD)

                # 收集 Picks
                for p in output.picks:
                    all_picks_data.append({
                        "station": p.trace_id, "phase": p.phase,
                        "peak_time": str(p.peak_time), "peak_value": p.peak_value,
                        "start_time": str(p.start_time), "end_time": str(p.end_time)
                    })

                # 收集 Detections (僅限 EQTransformer)
                if hasattr(output, 'detections'):
                    for d in output.detections:
                        all_detections_data.append({
                            "station": d.trace_id, "start_time": str(d.start_time),
                            "end_time": str(d.end_time), "peak_value": d.peak_value
                        })
                
                del output
            
            t_start += sub_chunk_size 
                    
        # 【修正】刪除了原本放在這裡、會導致記憶體爆炸的重複二次 classify 程式碼

        # 強制清理大時段記憶體
        del stream
        gc.collect()

    except Exception as e:
        print(f"區段 {current_start} 處理失敗: {e}")

    current_start += CHUNK_LENGTH

# --- 5. 輸出整合結果 (CSV) ---
if all_picks_data:
    df_picks = pd.DataFrame(all_picks_data).drop_duplicates(subset=["station", "phase", "peak_time"])
    df_picks['start_time'] = pd.to_datetime(df_picks['start_time'])
    df_picks = df_picks.sort_values(by='start_time').reset_index(drop=True)
    df_picks.to_csv(os.path.join(OUTPUT_DIR, "picks_results.csv"), index=False, encoding='utf-8-sig')
    print(f"Picks 處理完成，共計 {len(df_picks)} 筆結果。")

if all_detections_data:
    df_detections = pd.DataFrame(all_detections_data).drop_duplicates(subset=["station", "start_time"])
    df_detections['start_time'] = pd.to_datetime(df_detections['start_time'])
    df_detections = df_detections.sort_values(by='start_time').reset_index(drop=True)
    df_detections.to_csv(os.path.join(OUTPUT_DIR, "detections_results.csv"), index=False, encoding='utf-8-sig')