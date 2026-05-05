"""
Smart Traffic Signal Optimizer — FastAPI Backend v3
- Reads vehicle_count_south.txt written by vehicle_detection.py
- Uses TF ML models for timing prediction (falls back to formula)
- Broadcasts real-time state via WebSocket to dashboard
Run: uvicorn main:app --reload --port 8000
"""
import asyncio, json, base64, os, warnings, subprocess
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import pandas as pd

import traffic_controller
import trainthemodel

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ── Optional ML ────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO = True
except ImportError:
    _YOLO = False

app = FastAPI(title="Traffic Signal API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE = os.path.dirname(os.path.abspath(__file__))
DIRS = ["north", "south", "east", "west"]

# ── Load ML models ─────────────────────────────────────────────────────────────
# ML models are now managed by traffic_controller module
# We delegate all prediction logic to traffic_controller.predict_signal_times()
print(f"✅ ML models from traffic_controller: {'✅' if traffic_controller.ml_initialized else '❌(formula)'}")

# ── Load YOLO ──────────────────────────────────────────────────────────────────
yolo = None
if _YOLO:
    try:
        yolo = YOLO("yolov8n.pt")
        print("✅ YOLOv8 loaded")
    except Exception as e:
        print(f"⚠️  YOLO: {e}")

VEHICLE_CLS = {"car","motorbike","bus","truck","bicycle","motorcycle"}
CONF = 0.45

# ── Shared state ───────────────────────────────────────────────────────────────
state = {
    "mode": "idle",
    "input_source": "manual",       # manual | cv
    "vehicle_counts": {d: 0 for d in DIRS},
    "detections":     {d: [] for d in DIRS},
    "timings":        {d: {"green": 30, "yellow": 5} for d in DIRS},
    "ambulance": False, "ambulance_dir": None,
    "active_direction": "north", "active_phase": "idle", "time_remaining": 0,
    "ml_used": False, "yolo_used": False,
    "last_updated": None,
}

# ── WebSocket manager ──────────────────────────────────────────────────────────
class WSMan:
    def __init__(self): self.sockets: list[WebSocket] = []
    async def connect(self, ws):
        await ws.accept(); self.sockets.append(ws)
    def disconnect(self, ws):
        if ws in self.sockets: self.sockets.remove(ws)
    async def broadcast(self, data):
        txt = json.dumps(data)
        dead = []
        for ws in self.sockets:
            try: await ws.send_text(txt)
            except: dead.append(ws)
        [self.disconnect(w) for w in dead]

mgr = WSMan()

# ── Helpers ────────────────────────────────────────────────────────────────────
def run_yolo(frame: np.ndarray, direction: str) -> dict:
    if yolo is None or frame is None:
        return {"count":0,"ambulance":False,"detections":[],"b64":""}
    res = yolo(frame, imgsz=640, verbose=False)[0]
    count, amb, dets = 0, False, []
    ann = frame.copy()
    for box in res.boxes:
        cf = float(box.conf)
        if cf < CONF: continue
        lbl = yolo.names[int(box.cls)]
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        if lbl in VEHICLE_CLS:
            count += 1; dets.append({"label":lbl,"conf":round(cf,2)})
            cv2.rectangle(ann,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(ann,f"{lbl} {cf:.2f}",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        if lbl=="truck" and cf>0.7:
            amb=True
            cv2.rectangle(ann,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(ann,"AMBULANCE",(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.putText(ann,f"{direction.upper()}: {count}",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
    _,buf = cv2.imencode(".jpg",ann,[cv2.IMWRITE_JPEG_QUALITY,70])
    return {"count":count,"ambulance":amb,"detections":dets,"b64":base64.b64encode(buf).decode()}

# ── Background tasks ───────────────────────────────────────────────────────────
sim_task: Optional[asyncio.Task] = None
cv_task:  Optional[asyncio.Task] = None
yolo_process: Optional[subprocess.Popen] = None

async def sim_loop():
    idx = 0
    while True:
        if state["mode"] == "idle":
            await asyncio.sleep(0.5); continue
        d = DIRS[idx]
        g = state["timings"][d]["green"]
        y = state["timings"][d]["yellow"]
        if g > 0:
            state.update(active_direction=d, active_phase="green")
            for t in range(g, 0, -1):
                if state["mode"]=="idle": return
                state["time_remaining"] = t
                await mgr.broadcast({**state,"event":"tick"})
                await asyncio.sleep(1)
        state["active_phase"] = "yellow"
        for t in range(y, 0, -1):
            if state["mode"]=="idle": return
            state["time_remaining"] = t
            await mgr.broadcast({**state,"event":"tick"})
            await asyncio.sleep(1)
        idx = (idx+1) % 4

async def cv_poll_loop():
    """Poll vehicle_count_south.txt every 3s and re-predict.
    
    This implements hybrid CV mode where:
    - South direction data comes from vehicle_count_south.txt (written by vehicle_detection.py)
    - North, East, and West direction data come from manual sliders in the dashboard
    - Combined data is sent to traffic_controller for unified prediction
    
    The loop runs continuously while state["input_source"] == "cv" and:
    1. Reads South vehicle count and ambulance status from file (via traffic_controller)
    2. Combines with manual N/E/W counts from state
    3. Delegates prediction to traffic_controller.predict_signal_times()
    4. Broadcasts updated state via WebSocket to dashboard
    """
    while state["input_source"] == "cv":
        # Delegate South direction file reading to traffic_controller
        s_count, amb = traffic_controller.get_vehicle_count_south()
        state["vehicle_counts"]["south"] = s_count
        state["ambulance"] = amb
        state["ambulance_dir"] = "south" if amb else None
        n = state["vehicle_counts"]["north"]
        e = state["vehicle_counts"]["east"]
        w = state["vehicle_counts"]["west"]
        
        # Create DataFrame for prediction
        now = datetime.now()
        user_data = pd.DataFrame([{
            "time_of_day": now.hour + now.minute/60,
            "day_of_week": now.isoweekday(),
            "vehicle_count_north": n,
            "vehicle_count_south": s_count,
            "vehicle_count_east": e,
            "vehicle_count_west": w
        }])
        
        # Delegate prediction to traffic_controller
        ambulance_dir = "south" if amb else None
        state["timings"] = traffic_controller.predict_signal_times(user_data, ambulance_dir)
        state["ml_used"] = traffic_controller.ml_initialized
        state["last_updated"] = datetime.now().isoformat()
        
        # Broadcast updated state to all connected WebSocket clients
        await mgr.broadcast({**state, "event":"cv_update"})
        await asyncio.sleep(3)

def restart_sim():
    global sim_task
    if sim_task and not sim_task.done(): sim_task.cancel()
    sim_task = asyncio.create_task(sim_loop())

# ── Pydantic schemas ───────────────────────────────────────────────────────────
class ManualReq(BaseModel):
    north: int; south: int; east: int; west: int; ambulance: bool = False

class CountsReq(BaseModel):
    north: int; south: int; east: int; west: int

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "traffic_dashboard.html"))

@app.get("/api/info")
async def info():
    return {
        "ml_available": traffic_controller.ml_initialized, "yolo_available": yolo is not None,
        "ml_used": state["ml_used"], "yolo_used": state["yolo_used"],
        "mode": state["mode"], "input_source": state["input_source"],
    }

@app.get("/api/state")
async def get_state():
    return state

@app.post("/api/set_counts")
async def set_counts(req: CountsReq):
    """Update vehicle counts and recalculate signal timings.
    
    This endpoint allows the dashboard to update vehicle counts for all directions
    and trigger a new signal timing prediction. Used during manual mode and hybrid CV mode.
    
    Delegation Strategy:
    - Accepts vehicle counts from dashboard
    - Delegates prediction to traffic_controller.predict_signal_times()
    - Broadcasts updated state via WebSocket to all connected clients
    """
    state["vehicle_counts"]["north"] = req.north
    state["vehicle_counts"]["east"] = req.east
    state["vehicle_counts"]["west"] = req.west
    if state["input_source"] != "cv":
        state["vehicle_counts"]["south"] = req.south
    
    n = state["vehicle_counts"]["north"]
    s = state["vehicle_counts"]["south"]
    e = state["vehicle_counts"]["east"]
    w = state["vehicle_counts"]["west"]
    
    # Create DataFrame for prediction
    now = datetime.now()
    user_data = pd.DataFrame([{
        "time_of_day": now.hour + now.minute/60,
        "day_of_week": now.isoweekday(),
        "vehicle_count_north": n,
        "vehicle_count_south": s,
        "vehicle_count_east": e,
        "vehicle_count_west": w
    }])
    
    # Delegate prediction to traffic_controller
    ambulance_dir = "south" if state["ambulance"] else None
    state["timings"] = traffic_controller.predict_signal_times(user_data, ambulance_dir)
    state["ml_used"] = traffic_controller.ml_initialized
    state["last_updated"] = datetime.now().isoformat()
    
    # Broadcast state update via WebSocket
    await mgr.broadcast({**state, "event": "counts_updated"})
    return {"status": "ok"}

@app.post("/api/update")
async def manual_update(req: ManualReq):
    """Handle manual mode updates from dashboard.
    
    This endpoint processes manual vehicle count input from the dashboard sliders.
    It cancels any active CV polling, updates vehicle counts, and triggers prediction.
    
    Delegation Strategy:
    - Cancels CV polling loop if active (switches to manual mode)
    - Delegates prediction to traffic_controller.predict_signal_times()
    - Restarts simulation loop with new timings
    - Broadcasts updated state via WebSocket to all connected clients
    """
    global cv_task
    if cv_task and not cv_task.done(): cv_task.cancel()
    state["input_source"] = "manual"
    state["vehicle_counts"] = {"north":req.north,"south":req.south,"east":req.east,"west":req.west}
    state["ambulance"] = req.ambulance
    state["ambulance_dir"] = "south" if req.ambulance else None
    
    # Create DataFrame for prediction
    now = datetime.now()
    user_data = pd.DataFrame([{
        "time_of_day": now.hour + now.minute/60,
        "day_of_week": now.isoweekday(),
        "vehicle_count_north": req.north,
        "vehicle_count_south": req.south,
        "vehicle_count_east": req.east,
        "vehicle_count_west": req.west
    }])
    
    # Delegate prediction to traffic_controller
    ambulance_dir = "south" if req.ambulance else None
    state["timings"] = traffic_controller.predict_signal_times(user_data, ambulance_dir)
    state["ml_used"] = traffic_controller.ml_initialized
    state["mode"] = "ambulance" if req.ambulance else "running"
    state["last_updated"] = datetime.now().isoformat()
    state["yolo_used"] = False
    restart_sim()
    
    # Broadcast state update via WebSocket
    await mgr.broadcast({**state,"event":"update"})
    return {"status":"ok","timings":state["timings"],"ml_used":state["ml_used"]}

@app.post("/api/cv/start")
async def start_cv():
    """Start reading from vehicle_count_south.txt (launches vehicle_detection.py).
    
    This endpoint activates hybrid CV mode where:
    - South direction data comes from vehicle_count_south.txt (written by vehicle_detection.py)
    - North, East, and West direction data come from manual sliders
    
    Hybrid CV Mode Logic:
    1. Launches vehicle_detection.py subprocess if not already running
    2. Starts cv_poll_loop() background task to read South data every 3 seconds
    3. Combines South (from file) with N/E/W (from sliders) for prediction
    4. Restarts simulation loop with new mode
    5. Broadcasts state update via WebSocket
    """
    global cv_task, yolo_process
    state["input_source"] = "cv"
    state["mode"] = "running"
    
    # Launch vehicle_detection.py subprocess for South direction CV
    if yolo_process is None or yolo_process.poll() is not None:
        yolo_process = subprocess.Popen(["python", "vehicle_detection.py"])
        
    if cv_task and not cv_task.done(): cv_task.cancel()
    cv_task = asyncio.create_task(cv_poll_loop())
    restart_sim()
    
    # Broadcast state update via WebSocket
    await mgr.broadcast({**state,"event":"cv_started"})
    return {"status":"cv_started","info":"vehicle_detection.py launched in background"}

@app.post("/api/detect/upload")
async def detect_upload(direction: str = "south", file: UploadFile = File(...)):
    """Handle image upload for YOLO-based vehicle detection.
    
    This endpoint processes uploaded images using YOLO to detect vehicles and
    recalculate signal timings for the specified direction.
    
    Delegation Strategy:
    - Runs YOLO detection on uploaded image (local processing)
    - Updates vehicle count for specified direction
    - Delegates prediction to traffic_controller.predict_signal_times()
    - Restarts simulation loop with new timings
    - Broadcasts updated state via WebSocket to all connected clients
    """
    if yolo is None:
        return JSONResponse({"error":"YOLO not available"},status_code=503)
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    result = run_yolo(frame, direction)
    state["vehicle_counts"][direction] = result["count"]
    state["detections"][direction]  = result["detections"]
    state["yolo_used"] = True
    
    # Create DataFrame for prediction
    now = datetime.now()
    n,s,e,w = [state["vehicle_counts"][d] for d in DIRS]
    user_data = pd.DataFrame([{
        "time_of_day": now.hour + now.minute/60,
        "day_of_week": now.isoweekday(),
        "vehicle_count_north": n,
        "vehicle_count_south": s,
        "vehicle_count_east": e,
        "vehicle_count_west": w
    }])
    
    # Delegate prediction to traffic_controller
    ambulance_dir = "south" if state["ambulance"] else None
    state["timings"] = traffic_controller.predict_signal_times(user_data, ambulance_dir)
    state["ml_used"] = traffic_controller.ml_initialized
    state["mode"] = "running"
    state["last_updated"] = datetime.now().isoformat()
    restart_sim()
    
    # Broadcast state update via WebSocket
    await mgr.broadcast({**state,"event":"upload_detect"})
    return {"direction":direction,"count":result["count"],
            "ambulance":result["ambulance"],"detections":result["detections"],
            "frame_b64":result["b64"],"timings":state["timings"]}

@app.post("/api/reset")
async def reset():
    global sim_task, cv_task, yolo_process
    if yolo_process is not None:
        yolo_process.terminate()
        yolo_process = None
        
    state.update(mode="idle",input_source="manual",active_phase="idle",
                 time_remaining=0,ambulance=False,ambulance_dir=None,
                 ml_used=False,yolo_used=False,
                 vehicle_counts={d:0 for d in DIRS},
                 detections={d:[] for d in DIRS},
                 timings={d:{"green":30,"yellow":5} for d in DIRS},
                 last_updated=datetime.now().isoformat())
    for t in [sim_task, cv_task]:
        if t and not t.done(): t.cancel()
    await mgr.broadcast({**state,"event":"reset"})
    return {"status":"reset"}

@app.post("/api/train")
async def train_ml():
    """Train the ML models using trainthemodel.py"""
    success = trainthemodel.train_models()
    if success:
        # Re-initialize models in traffic_controller
        traffic_controller.init_ml()
        return {"status": "success", "message": "Models trained successfully."}
    return JSONResponse({"error": "Failed to train models. Check if dataset exists."}, status_code=500)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time state updates.
    
    WebSocket Broadcast Strategy:
    - Accepts WebSocket connections from dashboard clients
    - Sends initial state on connection
    - Broadcasts state updates on every change (manual update, CV update, tick, reset)
    - Automatically removes disconnected clients from broadcast list
    - Provides real-time synchronization between backend and dashboard
    """
    await mgr.connect(ws)
    await ws.send_text(json.dumps({**state,"event":"init"}))
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        mgr.disconnect(ws)

@app.on_event("startup")
async def startup():
    global sim_task
    sim_task = asyncio.create_task(sim_loop())
    print(f"🚀 Ready | ML:{'✅' if traffic_controller.ml_initialized else '❌(formula)'} | YOLO:{'✅' if yolo else '❌'}")
