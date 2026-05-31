# -*- coding: utf-8 -*-
"""
inference_api.py — FastAPI REST API
════════════════════════════════════
Menerima hasil deteksi mood dari inference_camera.py
Menyimpan ke database SQLite (mood_checkins.db)
Mengekspos endpoint untuk frontend React (tim FS)

Install:
    pip install fastapi uvicorn sqlalchemy anthropic

Jalankan:
    uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload

Docs: http://localhost:8000/docs
"""

import json
from datetime import datetime, timezone
from collections import Counter

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Modul AI suggestion (pisah file — toggle di payload)
from ai_suggestion import get_suggestion

# ============================================================
# DATABASE
# ============================================================

DATABASE_URL = "sqlite:///./mood_checkins.db"
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base         = declarative_base()


class CheckinRecord(Base):
    __tablename__ = "checkins"

    id                = Column(Integer,  primary_key=True, index=True)
    user_id           = Column(String,   default="default", index=True)
    mood              = Column(String,   nullable=False)
    confidence        = Column(Float,    nullable=False)
    emoji             = Column(String,   default="")
    all_probs_json    = Column(Text,     default="{}")
    # ai_suggestion disimpan sebagai JSON string agar tips ikut tersimpan
    ai_suggestion_json = Column(Text,   default="{}")
    checkin_time      = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)
print("[DB] mood_checkins.db siap.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================
# SCHEMAS (Pydantic)
# ============================================================

class CheckinRequest(BaseModel):
    user_id:        str   = "default"
    mood:           str
    confidence:     float
    emoji:          str   = ""
    all_probs:      dict  = {}
    # Toggle AI suggestion — default True, frontend boleh set False
    use_ai_comment: bool  = True


class AISuggestionSchema(BaseModel):
    enabled:  bool
    source:   str          # "ai" | "fallback"
    mood:     str
    message:  str
    tips:     list[str]


class CheckinData(BaseModel):
    id:            int
    user_id:       str
    mood:          str
    confidence:    float
    emoji:         str
    scores:        dict
    ai_suggestion: AISuggestionSchema | None
    checkin_time:  str


class CheckinResponse(BaseModel):
    status:  str
    data:    CheckinData
    message: str

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Facial Expression Recognition API",
    description="Daily Mood Check-in — kamera → model → API → DB → React (tim FS)",
    version="2.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Info Endpoints ────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "service":   "FER API",
        "version":   "2.2",
        "docs":      "http://localhost:8000/docs",
        "endpoints": [
            "POST /checkin",
            "GET  /checkins",
            "GET  /checkins/latest",
            "GET  /stats",
            "GET  /emotions",
            "GET  /health",
        ],
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status":    "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/emotions", tags=["Info"])
def list_emotions():
    """Daftar emosi yang didukung model — referensi tim FS."""
    return {
        "status": "success",
        "data": [
            {"id": "angry",   "emoji": "😠", "label": "Marah"},
            {"id": "happy",   "emoji": "😄", "label": "Bahagia"},
            {"id": "sad",     "emoji": "😢", "label": "Sedih"},
            {"id": "neutral", "emoji": "😐", "label": "Netral"},
        ],
    }

# ── Check-in Endpoint ─────────────────────────────────────────────────────────

@app.post("/checkin", tags=["Check-in"])
def submit_checkin(payload: CheckinRequest, db: Session = Depends(get_db)):
    """
    Dipanggil oleh inference_camera.py setelah scan 4 detik selesai.

    Body JSON:
    ```json
    {
        "user_id":        "default",
        "mood":           "happy",
        "confidence":     87.5,
        "emoji":          "😄",
        "all_probs":      {"angry": 2.1, "happy": 87.5, "sad": 3.2, "neutral": 7.2},
        "use_ai_comment": true
    }
    ```

    Response JSON:
    ```json
    {
        "status": "success",
        "data": {
            "id": 1,
            "user_id": "default",
            "mood": "happy",
            "confidence": 87.5,
            "emoji": "😄",
            "scores": { "angry": 2.1, "happy": 87.5, "sad": 3.2, "neutral": 7.2 },
            "ai_suggestion": {
                "enabled": true,
                "source":  "ai",
                "mood":    "happy",
                "message": "Energi positifmu luar biasa, sebarkan semangat itu!",
                "tips":    ["tip1", "tip2", "tip3"]
            },
            "checkin_time": "2025-05-28T10:30:00+00:00"
        },
        "message": "Check-in disimpan! Mood: HAPPY 😄"
    }
    ```

    Jika `use_ai_comment=false`, field `ai_suggestion` akan bernilai `null`.
    """
    # Validasi mood
    valid_moods = ["angry", "happy", "sad", "neutral"]
    if payload.mood not in valid_moods:
        raise HTTPException(
            status_code=422,
            detail=f"Mood '{payload.mood}' tidak valid. Pilih salah satu: {valid_moods}"
        )

    # Generate AI suggestion (dari ai_suggestion.py)
    suggestion = get_suggestion(
        mood=payload.mood,
        confidence=payload.confidence,
        enabled=payload.use_ai_comment,
        use_ai=True,
    )

    # Simpan ke DB
    rec = CheckinRecord(
        user_id            = payload.user_id,
        mood               = payload.mood,
        confidence         = payload.confidence,
        emoji              = payload.emoji,
        all_probs_json     = json.dumps(payload.all_probs),
        ai_suggestion_json = json.dumps(suggestion) if suggestion else "null",
        checkin_time       = datetime.now(timezone.utc),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    print(f"[DB] Saved → id={rec.id} | mood={rec.mood} | conf={rec.confidence:.1f}% | ai={'yes' if suggestion else 'no'}")

    return {
        "status": "success",
        "data": {
            "id":            rec.id,
            "user_id":       rec.user_id,
            "mood":          rec.mood,
            "confidence":    rec.confidence,
            "emoji":         rec.emoji,
            "scores":        payload.all_probs,
            "ai_suggestion": suggestion,          # dict atau None
            "checkin_time":  rec.checkin_time.isoformat(),
        },
        "message": f"Check-in disimpan! Mood: {rec.mood.upper()} {rec.emoji}",
    }

# ── History Endpoints ─────────────────────────────────────────────────────────

def _parse_record(r: CheckinRecord) -> dict:
    """Convert ORM record ke dict JSON-friendly."""
    raw_suggestion = r.ai_suggestion_json or "null"
    try:
        suggestion = json.loads(raw_suggestion)
    except Exception:
        suggestion = None

    return {
        "id":            r.id,
        "user_id":       r.user_id,
        "mood":          r.mood,
        "confidence":    r.confidence,
        "emoji":         r.emoji,
        "scores":        json.loads(r.all_probs_json or "{}"),
        "ai_suggestion": suggestion,
        "checkin_time":  r.checkin_time.isoformat(),
    }


@app.get("/checkins", tags=["History"])
def get_checkins(
    user_id: str = "default",
    limit:   int = 20,
    db: Session = Depends(get_db),
):
    """Riwayat check-in terbaru — untuk ditampilkan di React."""
    recs = (
        db.query(CheckinRecord)
        .filter(CheckinRecord.user_id == user_id)
        .order_by(CheckinRecord.checkin_time.desc())
        .limit(limit)
        .all()
    )
    return {
        "status": "success",
        "data":   [_parse_record(r) for r in recs],
    }


@app.get("/checkins/latest", tags=["History"])
def get_latest(user_id: str = "default", db: Session = Depends(get_db)):
    """Check-in paling terakhir — untuk dashboard React saat pertama load."""
    r = (
        db.query(CheckinRecord)
        .filter(CheckinRecord.user_id == user_id)
        .order_by(CheckinRecord.checkin_time.desc())
        .first()
    )
    if not r:
        raise HTTPException(status_code=404, detail="Belum ada check-in untuk user ini.")

    return {
        "status": "success",
        "data":   _parse_record(r),
    }

# ── Analytics Endpoint ────────────────────────────────────────────────────────

@app.get("/stats", tags=["Analytics"])
def get_stats(user_id: str = "default", db: Session = Depends(get_db)):
    """Statistik mood — untuk chart di React."""
    recs = (
        db.query(CheckinRecord)
        .filter(CheckinRecord.user_id == user_id)
        .all()
    )

    if not recs:
        return {
            "status": "success",
            "data": {
                "total_checkins":   0,
                "most_common_mood": None,
                "mood_counts":      {},
                "avg_confidence":   0.0,
            },
        }

    moods  = [r.mood for r in recs]
    counts = dict(Counter(moods))
    avg    = round(sum(r.confidence for r in recs) / len(recs), 1)

    return {
        "status": "success",
        "data": {
            "total_checkins":   len(recs),
            "most_common_mood": max(counts, key=counts.get),
            "mood_counts":      counts,
            "avg_confidence":   avg,
        },
    }

# ── Delete Endpoint ───────────────────────────────────────────────────────────

@app.delete("/checkins/{checkin_id}", tags=["History"])
def delete_checkin(checkin_id: int, db: Session = Depends(get_db)):
    """Hapus satu record berdasarkan id."""
    r = db.query(CheckinRecord).filter(CheckinRecord.id == checkin_id).first()
    if not r:
        raise HTTPException(status_code=404, detail=f"Record id={checkin_id} tidak ditemukan.")
    db.delete(r)
    db.commit()
    return {
        "status":  "success",
        "message": f"Record id={checkin_id} berhasil dihapus.",
    }

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("  FER FastAPI Server  |  http://localhost:8000")
    print("  Swagger UI          |  http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8000, reload=True)