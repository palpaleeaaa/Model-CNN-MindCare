# -*- coding: utf-8 -*-
"""
ai_suggestion.py
════════════════
Modul saran AI per mood — bisa dipanggil dari inference_api.py
atau diimport langsung oleh modul lain.

Cara pakai:
    from ai_suggestion import get_suggestion

    # Dengan AI (butuh ANTHROPIC_API_KEY di env)
    result = get_suggestion("happy", confidence=87.5, use_ai=True)

    # Tanpa AI (pakai fallback statis)
    result = get_suggestion("sad", confidence=72.0, use_ai=False)

    # Disable total (return None — frontend tidak render section ini)
    result = get_suggestion("neutral", confidence=60.0, enabled=False)

Return format:
    {
        "enabled":  True,
        "source":   "ai" | "fallback",
        "mood":     "happy",
        "message":  "Energi positifmu luar biasa! ...",
        "tips":     ["tip1", "tip2", "tip3"]   # hanya dari fallback
    }
    atau None jika enabled=False
"""

import os

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

# ============================================================
# FALLBACK STATIS — dipakai jika AI tidak tersedia / error
# Struktur: message (1 kalimat) + tips (3 butir aksi nyata)
# ============================================================

_FALLBACK: dict[str, dict] = {
    "angry": {
        "message": "Tarik napas dalam, lepaskan perlahan — kamu bisa melewati ini. 💙",
        "tips": [
            "Hitung mundur dari 10 sambil tarik napas.",
            "Minum segelas air putih, lalu berjalan sebentar.",
            "Tulis apa yang bikin marah di catatan, bukan ke orang lain.",
        ],
    },
    "sad": {
        "message": "Perasaanmu valid. Satu langkah kecil hari ini sudah lebih dari cukup. 🌤️",
        "tips": [
            "Hubungi satu orang yang kamu percaya hari ini.",
            "Keluar sebentar dan hirup udara segar.",
            "Lakukan hal kecil yang biasanya kamu nikmati.",
        ],
    },
    "happy": {
        "message": "Energi positifmu luar biasa — sebarkan semangat itu! ✨",
        "tips": [
            "Tandai momen ini di jurnal harianmu.",
            "Bagikan kebaikanmu ke satu orang hari ini.",
            "Gunakan energi ini untuk mulai sesuatu yang tertunda.",
        ],
    },
    "neutral": {
        "message": "Hari yang baik untuk memulai sesuatu baru. Semangat! 🚀",
        "tips": [
            "Tentukan satu prioritas utama untuk hari ini.",
            "Luangkan 5 menit untuk stretching ringan.",
            "Coba satu hal baru yang sudah lama ingin kamu coba.",
        ],
    },
}

# ============================================================
# PROMPT PER MOOD — dipakai saat AI aktif
# Mengembalikan JSON agar tips juga bisa di-generate AI
# ============================================================

_AI_PROMPTS: dict[str, str] = {
    "angry": (
        "Mood pengguna terdeteksi MARAH (confidence {confidence:.1f}%). "
        "Balas HANYA dalam format JSON berikut tanpa tanda ```:\n"
        '{{"message": "<satu kalimat menenangkan max 20 kata, bahasa Indonesia>", '
        '"tips": ["<aksi nyata 1>", "<aksi nyata 2>", "<aksi nyata 3>"]}}\n'
        "Tips harus singkat, konkret, dan menenangkan."
    ),
    "sad": (
        "Mood pengguna terdeteksi SEDIH (confidence {confidence:.1f}%). "
        "Balas HANYA dalam format JSON berikut tanpa tanda ```:\n"
        '{{"message": "<satu kalimat hangat dan supportif max 20 kata, bahasa Indonesia>", '
        '"tips": ["<aksi nyata 1>", "<aksi nyata 2>", "<aksi nyata 3>"]}}\n'
        "Tips harus singkat, konkret, dan menghibur."
    ),
    "happy": (
        "Mood pengguna terdeteksi BAHAGIA (confidence {confidence:.1f}%). "
        "Balas HANYA dalam format JSON berikut tanpa tanda ```:\n"
        '{{"message": "<satu kalimat merayakan energi positif max 20 kata, bahasa Indonesia>", '
        '"tips": ["<aksi nyata 1>", "<aksi nyata 2>", "<aksi nyata 3>"]}}\n'
        "Tips harus singkat, konkret, dan mempertahankan semangat."
    ),
    "neutral": (
        "Mood pengguna terdeteksi NETRAL (confidence {confidence:.1f}%). "
        "Balas HANYA dalam format JSON berikut tanpa tanda ```:\n"
        '{{"message": "<satu kalimat motivasi ringan max 20 kata, bahasa Indonesia>", '
        '"tips": ["<aksi nyata 1>", "<aksi nyata 2>", "<aksi nyata 3>"]}}\n'
        "Tips harus singkat, konkret, dan memulai hari dengan positif."
    ),
}


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def get_suggestion(
    mood: str,
    confidence: float = 0.0,
    enabled: bool = True,
    use_ai: bool = True,
) -> dict | None:
    """
    Parameters
    ----------
    mood        : "angry" | "happy" | "sad" | "neutral"
    confidence  : skor 0–100 dari model
    enabled     : False  → return None (frontend skip render section ini)
    use_ai      : True   → coba panggil Anthropic API dulu, fallback jika gagal
                  False  → langsung pakai fallback statis

    Returns
    -------
    dict atau None
    """
    if not enabled:
        return None

    # Normalisasi mood — jika unknown pakai neutral
    mood = mood.lower()
    if mood not in _FALLBACK:
        mood = "neutral"

    if use_ai:
        result = _call_anthropic(mood, confidence)
        if result:
            return result

    # Fallback statis
    fb = _FALLBACK[mood]
    return {
        "enabled": True,
        "source":  "fallback",
        "mood":    mood,
        "message": fb["message"],
        "tips":    fb["tips"],
    }


# ============================================================
# INTERNAL — Anthropic API call
# ============================================================

def _call_anthropic(mood: str, confidence: float) -> dict | None:
    """
    Panggil Anthropic API. Return dict jika sukses, None jika gagal.
    Semua exception ditangkap supaya tidak crash inference_api.
    """
    if not _ANTHROPIC_AVAILABLE:
        print("[ai_suggestion] anthropic package tidak ter-install → fallback")
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[ai_suggestion] ANTHROPIC_API_KEY tidak di-set → fallback")
        return None

    prompt_template = _AI_PROMPTS.get(mood, _AI_PROMPTS["neutral"])
    prompt = prompt_template.format(confidence=confidence)

    try:
        import json as _json
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()

        # Bersihkan fence jika model tetap menambahkan ```
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = _json.loads(raw)

        return {
            "enabled": True,
            "source":  "ai",
            "mood":    mood,
            "message": parsed.get("message", _FALLBACK[mood]["message"]),
            "tips":    parsed.get("tips",    _FALLBACK[mood]["tips"]),
        }

    except Exception as e:
        print(f"[ai_suggestion] API error: {e} → fallback")
        return None


# ============================================================
# QUICK TEST — jalankan langsung: python ai_suggestion.py
# ============================================================

if __name__ == "__main__":
    import json

    test_cases = [
        ("happy",   92.5, True,  True),
        ("sad",     78.3, True,  False),   # pakai fallback
        ("angry",   65.0, True,  True),
        ("neutral", 55.0, False, True),    # disabled → None
    ]

    for mood, conf, enabled, use_ai in test_cases:
        result = get_suggestion(mood, conf, enabled=enabled, use_ai=use_ai)
        print(f"\n{'='*50}")
        print(f"mood={mood} | conf={conf} | enabled={enabled} | use_ai={use_ai}")
        print(json.dumps(result, ensure_ascii=False, indent=2))