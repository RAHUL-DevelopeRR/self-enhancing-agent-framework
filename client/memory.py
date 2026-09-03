import sqlite3
import json
import time
import math
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

class EpisodicMemory:
    """Manages persistent episodes and learned lessons using SQLite and local zero-token vector similarity."""

    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                prompt TEXT NOT NULL,
                draft_content TEXT,
                critic_feedback TEXT,
                final_content TEXT,
                model_used TEXT,
                provider TEXT,
                critic_score REAL,
                success BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                trigger_pattern TEXT NOT NULL,
                lesson_text TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                success_count INTEGER DEFAULT 1,
                failure_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.commit()

    def _local_embed_bow(self, text: str) -> Dict[str, float]:
        """Zero-token local bag-of-words / TF representation for vector cosine similarity."""
        words = re.findall(r"\w+", text.lower())
        vec: Dict[str, float] = {}
        for w in words:
            if len(w) > 2:
                vec[w] = vec.get(w, 0.0) + 1.0
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def _cosine_similarity(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        intersection = set(v1.keys()) & set(v2.keys())
        return sum(v1[k] * v2[k] for k in intersection)

    def record_episode(self, episode_id: str, task_type: str, prompt: str, draft: str, 
                       critic_feedback: str, final_content: str, model: str, provider: str, 
                       score: float, success: bool):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT OR REPLACE INTO episodes 
            (id, task_type, prompt, draft_content, critic_feedback, final_content, model_used, provider, critic_score, success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (episode_id, task_type, prompt, draft, critic_feedback, final_content, model, provider, score, success))
            conn.commit()

    def add_or_update_lesson(self, lesson_id: str, task_type: str, trigger_pattern: str, 
                             lesson_text: str, success: bool = True):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT success_count, failure_count FROM lessons WHERE id = ?", (lesson_id,))
            row = cursor.fetchone()
            if row:
                s_count = row[0] + (1 if success else 0)
                f_count = row[1] + (0 if success else 1)
                confidence = s_count / float(s_count + f_count)
                cursor.execute("""
                UPDATE lessons SET success_count = ?, failure_count = ?, confidence = ? WHERE id = ?
                """, (s_count, f_count, confidence, lesson_id))
            else:
                cursor.execute("""
                INSERT INTO lessons (id, task_type, trigger_pattern, lesson_text, confidence, success_count, failure_count)
                VALUES (?, ?, ?, ?, 1.0, 1, 0)
                """, (lesson_id, task_type, trigger_pattern, lesson_text))
            conn.commit()

    def retrieve_relevant_lessons(self, query: str, task_type: Optional[str] = None, top_k: int = 4) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if task_type:
                cursor.execute("SELECT id, task_type, trigger_pattern, lesson_text, confidence FROM lessons WHERE task_type = ?", (task_type,))
            else:
                cursor.execute("SELECT id, task_type, trigger_pattern, lesson_text, confidence FROM lessons")
            rows = cursor.fetchall()

        if not rows:
            return []

        q_vec = self._local_embed_bow(query)
        scored = []
        for r in rows:
            lesson_vec = self._local_embed_bow(f"{r[2]} {r[3]}")
            sim = self._cosine_similarity(q_vec, lesson_vec)
            scored.append({
                "id": r[0],
                "task_type": r[1],
                "trigger": r[2],
                "lesson": r[3],
                "confidence": r[4],
                "similarity": sim
            })

        scored.sort(key=lambda x: (x["confidence"] * 0.4 + x["similarity"] * 0.6), reverse=True)
        return scored[:top_k]
