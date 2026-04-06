"""
Nano-RAG — OpenRouter embeddings + cosinus Python pur + JSON.
Zéro dépendance supplémentaire.
"""
import json
import math
import logging
import os
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

OPENROUTER_KEY = (
    os.environ.get("OPENROUTER_KEY")
    or os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)
_EMBED_URL   = "https://openrouter.ai/api/v1/embeddings"
_EMBED_MODEL = "openai/text-embedding-3-small"

MEMORY_DIR   = Path(__file__).parent / "memory"
MEMORY_FILE  = MEMORY_DIR / "memory.json"
CERVEAU_FILE = Path(__file__).parent / "cerveau.md"
RAG_INDEX    = MEMORY_DIR / "rag_index.json"

CHUNK_SIZE    = 300   # caractères par chunk
CHUNK_OVERLAP = 50    # chevauchement pour ne pas couper le sens


# ── Helpers ──────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    """Appel OpenRouter embeddings avec logging complet pour debug."""
    resp = requests.post(
        _EMBED_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": _EMBED_MODEL, "input": text[:8000]},
        timeout=20,
    )
    if not resp.ok:
        logger.error(f"[RAG] embed ERREUR HTTP {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
    data = resp.json()
    vec = data["data"][0]["embedding"]
    logger.info(f"[RAG] embed OK — dim={len(vec)} premier={vec[0]:.4f}")
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _chunk_chars(text: str) -> list[str]:
    """
    Découpe par caractères avec overlap — fonctionne sur n'importe quel texte
    (JSON sérialisé, markdown, texte brut).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _entries_to_texts(entries) -> list[str]:
    """
    Convertit n'importe quelle structure JSON (list, dict, nested)
    en chunks texte. Ne dépend d'aucune clé spécifique.
    """
    texts = []
    if isinstance(entries, list):
        for item in entries:
            raw = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
            texts.extend(_chunk_chars(raw))
    elif isinstance(entries, dict):
        texts.extend(_chunk_chars(json.dumps(entries, ensure_ascii=False)))
    else:
        texts.extend(_chunk_chars(str(entries)))
    return texts


# ── Indexation ────────────────────────────────────────────────────────────────

def index_memory() -> str:
    """
    Lit memory.json + cerveau.md, chunk par caractères, embed, sauvegarde.
    Robuste à toute structure JSON.
    """
    texts: list[str] = []

    # memory.json — structure inconnue → sérialiser chaque item intégralement
    if MEMORY_FILE.exists():
        try:
            raw = MEMORY_FILE.read_text(encoding="utf-8")
            entries = json.loads(raw)
            batch = _entries_to_texts(entries)
            logger.info(f"[RAG] memory.json → {len(batch)} chunks")
            texts.extend(batch)
        except Exception as exc:
            logger.warning(f"[RAG] memory.json lecture échouée: {exc}")

    # cerveau.md — chunk par caractères aussi
    if CERVEAU_FILE.exists():
        try:
            md = CERVEAU_FILE.read_text(encoding="utf-8")
            batch = _chunk_chars(md)
            logger.info(f"[RAG] cerveau.md → {len(batch)} chunks")
            texts.extend(batch)
        except Exception as exc:
            logger.warning(f"[RAG] cerveau.md lecture échouée: {exc}")

    if not texts:
        return "❌ Rien à indexer : memory.json vide et cerveau.md absent."

    logger.info(f"[RAG] Total à embedder : {len(texts)} chunks")

    index: list[dict] = []
    errors = 0
    for i, txt in enumerate(texts):
        try:
            vec = _embed(txt)
            index.append({"text": txt, "embedding": vec})
        except Exception as exc:
            logger.warning(f"[RAG] chunk {i} embed échoué: {exc}")
            errors += 1

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RAG_INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    tmp.replace(RAG_INDEX)

    msg = f"✅ Index RAG mis à jour — {len(index)} chunks indexés"
    if errors:
        msg += f" ({errors} échecs)"
    logger.info(f"[RAG] {msg}")
    return msg


# ── Requête ───────────────────────────────────────────────────────────────────

def rag_query(question: str, top_k: int = 3) -> str:
    """
    Embed la question, cosinus sur l'index, retourne les top_k chunks.
    Toujours retourne au moins le top-1 pour éviter les faux négatifs.
    """
    if not RAG_INDEX.exists():
        logger.warning("[RAG] rag_index.json absent — lance 'indexe ma mémoire'")
        return ""
    try:
        index: list[dict] = json.loads(RAG_INDEX.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"[RAG] lecture index: {exc}")
        return ""
    if not index:
        return ""

    try:
        q_vec = _embed(question)
    except Exception as exc:
        logger.error(f"[RAG] embed question échoué: {exc}")
        return ""

    scored = sorted(
        [(e["text"], _cosine(q_vec, e["embedding"])) for e in index],
        key=lambda x: x[1],
        reverse=True,
    )

    logger.info(f"[RAG] top scores: {[round(s,3) for _,s in scored[:5]]}")

    # Seuil souple : 0.1 suffit, et on force toujours au moins le top-1
    top = [text for text, score in scored[:top_k] if score > 0.1]
    if not top and scored:
        top = [scored[0][0]]
        logger.info(f"[RAG] score faible ({scored[0][1]:.3f}) — top-1 forcé")

    return "\n---\n".join(top)
