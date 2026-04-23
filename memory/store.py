"""
AKBASCORE V13.0 — Persistent Memory
SQLite-backed immortal memory with vector similarity search.

Every experience is stored permanently on disk.
Sleep cycles prune weak memories; important ones are strengthened.
"""

import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional
import json

from config.hardware import MemoryConfig


class MemoryRecord:
    """A single stored memory."""
    __slots__ = ['id', 'content', 'embedding', 'importance', 'timestamp',
                 'access_count', 'emotion', 'layer', 'source']

    def __init__(self, id, content, embedding, importance, timestamp,
                 access_count, emotion, layer, source):
        self.id = id
        self.content = content
        self.embedding = embedding
        self.importance = importance
        self.timestamp = timestamp
        self.access_count = access_count
        self.emotion = emotion
        self.layer = layer
        self.source = source


class PermanentMemory:
    """
    Immortal SQLite memory store.
    
    Features:
    - Persistent storage (survives reboots)
    - Vector similarity search (cosine)
    - Importance-weighted recall
    - Pruning of weak/stale memories
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                content       TEXT    NOT NULL,
                embedding     BLOB,
                importance    REAL    DEFAULT 0.5,
                timestamp     TEXT    NOT NULL,
                access_count  INTEGER DEFAULT 0,
                emotion       REAL    DEFAULT 0.0,
                layer         INTEGER DEFAULT 1,
                source        TEXT    DEFAULT 'internal'
            );

            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_timestamp  ON memories(timestamp DESC);

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        ''')
        self.conn.commit()

    # ----------------------------------------------------------------
    # WRITE
    # ----------------------------------------------------------------

    def save(self, content: str, embedding: np.ndarray, importance: float,
             emotion: float = 0.0, layer: int = 1, source: str = 'internal') -> int:
        """Store a new memory. Returns the new memory ID."""
        cur = self.conn.execute(
            '''INSERT INTO memories (content, embedding, importance, timestamp, emotion, layer, source)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (content[:1000], embedding.tobytes(), float(importance),
             datetime.now().isoformat(), float(emotion), layer, source)
        )
        self.conn.commit()
        return cur.lastrowid

    def update_importance(self, memory_id: int, delta: float = 0.05):
        """Strengthen a memory (called on recall)."""
        self.conn.execute(
            '''UPDATE memories
               SET importance    = MIN(1.0, importance + ?),
                   access_count  = access_count + 1
               WHERE id = ?''',
            (delta, memory_id)
        )
        self.conn.commit()

    # ----------------------------------------------------------------
    # READ
    # ----------------------------------------------------------------

    def recall_top(self, limit: int = 100) -> List[MemoryRecord]:
        """Recall most important memories."""
        rows = self.conn.execute(
            '''SELECT * FROM memories
               ORDER BY importance DESC, access_count DESC
               LIMIT ?''', (limit,)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def recall_recent(self, limit: int = 50) -> List[MemoryRecord]:
        """Recall most recent memories."""
        rows = self.conn.execute(
            '''SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?''', (limit,)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[MemoryRecord, float]]:
        """
        Cosine similarity search across all stored embeddings.
        Returns (memory, similarity_score) pairs, sorted by relevance.
        """
        rows = self.conn.execute(
            'SELECT * FROM memories WHERE embedding IS NOT NULL'
        ).fetchall()

        results = []
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        for row in rows:
            try:
                emb = np.frombuffer(row['embedding'], dtype=np.float32)
                if emb.shape != query_embedding.shape:
                    continue
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                sim = float(np.dot(q_norm, emb_norm))
                results.append((self._row_to_record(row), sim))
            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return self.conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]

    # ----------------------------------------------------------------
    # PRUNE
    # ----------------------------------------------------------------

    def prune_weak(self, threshold: Optional[float] = None) -> int:
        """Delete memories below importance threshold."""
        t = threshold or self.config.PRUNING_THRESHOLD
        cur = self.conn.execute('DELETE FROM memories WHERE importance < ?', (t,))
        self.conn.commit()
        return cur.rowcount

    def prune_overflow(self) -> int:
        """If memory count exceeds MAX_MEMORIES, delete least important."""
        count = self.count()
        if count <= self.config.MAX_MEMORIES:
            return 0
        excess = count - self.config.MAX_MEMORIES
        self.conn.execute(
            '''DELETE FROM memories WHERE id IN (
               SELECT id FROM memories ORDER BY importance ASC, access_count ASC
               LIMIT ?)''', (excess,)
        )
        self.conn.commit()
        return excess

    # ----------------------------------------------------------------
    # INTERNAL
    # ----------------------------------------------------------------

    def _row_to_record(self, row) -> MemoryRecord:
        emb = None
        if row['embedding']:
            try:
                emb = np.frombuffer(row['embedding'], dtype=np.float32)
            except Exception:
                pass
        return MemoryRecord(
            id=row['id'], content=row['content'], embedding=emb,
            importance=row['importance'], timestamp=row['timestamp'],
            access_count=row['access_count'], emotion=row['emotion'],
            layer=row['layer'], source=row['source']
        )

    def close(self):
        self.conn.close()
