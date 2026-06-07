import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path


DB_PATH = "logs/queries.db"


class QueryLogger:
    """
    Logs every prediction to SQLite database.

    Why SQLite:
    - Zero setup, no separate server needed
    - Perfect for single-instance applications
    - Easy to query with standard SQL
    - Feeds into monitoring and error analysis

    Schema captures everything needed for:
    - Performance analysis (latency fields)
    - Error analysis (predicted vs actual if known)
    - Usage patterns (intent distribution)
    - Debugging (full query and response logged)
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    predicted_intent TEXT NOT NULL,
                    similarity_score REAL,
                    is_clear INTEGER,
                    entities_json TEXT,
                    retrieval_time_ms REAL,
                    llm_time_ms REAL,
                    total_time_ms REAL,
                    retrieved_examples_json TEXT
                )
            """)
            conn.commit()

    def log(self,
            user_input: str,
            predicted_intent: str,
            similarity_score: float,
            is_clear: bool,
            entities: dict,
            retrieval_time_ms: float,
            llm_time_ms: float,
            total_time_ms: float,
            retrieved_examples: list[str]) -> None:
        """Log a single prediction to database."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO queries (
                    timestamp, user_input, predicted_intent,
                    similarity_score, is_clear, entities_json,
                    retrieval_time_ms, llm_time_ms, total_time_ms,
                    retrieved_examples_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                user_input,
                predicted_intent,
                similarity_score,
                int(is_clear),
                json.dumps(entities),
                retrieval_time_ms,
                llm_time_ms,
                total_time_ms,
                json.dumps(retrieved_examples)
            ))
            conn.commit()

    def get_recent(self, limit: int = 10) -> list[dict]:
        """Get most recent predictions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM queries
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """Get summary statistics from logs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_queries,
                    AVG(total_time_ms) as avg_total_ms,
                    AVG(retrieval_time_ms) as avg_retrieval_ms,
                    AVG(llm_time_ms) as avg_llm_ms,
                    AVG(similarity_score) as avg_similarity
                FROM queries
            """)
            row = cursor.fetchone()

            cursor2 = conn.execute("""
                SELECT predicted_intent, COUNT(*) as count
                FROM queries
                GROUP BY predicted_intent
                ORDER BY count DESC
            """)
            intent_counts = {row[0]: row[1] for row in cursor2.fetchall()}

            return {
                "total_queries": row[0],
                "avg_total_ms": round(row[1] or 0, 1),
                "avg_retrieval_ms": round(row[2] or 0, 1),
                "avg_llm_ms": round(row[3] or 0, 1),
                "avg_similarity": round(row[4] or 0, 3),
                "intent_distribution": intent_counts
            }


# ── Quick Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # Use test database
    test_db = "logs/test_queries.db"
    logger = QueryLogger(db_path=test_db)

    print("Testing QueryLogger...\n")

    # Log 3 sample predictions
    test_data = [
        ("Book a flight to Delhi", "book_flight", 0.87, True,
         {"location": "Delhi"}, 12.5, 800.0, 815.0,
         ["Book a flight to Delhi", "I want to fly to Mumbai"]),

        ("Order me a pizza", "order_food", 0.92, True,
         {"food_item": "pizza"}, 10.0, 650.0, 662.0,
         ["Order me a pizza", "Get me some food"]),

        ("xyzzy random gibberish", "unclear", 0.22, False,
         {}, 8.0, 0.0, 8.0,
         ["some example"]),
    ]

    for data in test_data:
        logger.log(*data)

    print("✅ Logged 3 predictions\n")

    # Retrieve recent
    recent = logger.get_recent(limit=3)
    print(f"Recent queries: {len(recent)}")
    for r in recent:
        print(f"  [{r['timestamp'][:19]}] "
              f"'{r['user_input'][:30]}' → {r['predicted_intent']} "
              f"({r['total_time_ms']:.0f}ms)")

    # Get stats
    stats = logger.get_stats()
    print(f"\nStats:")
    print(f"  Total queries:    {stats['total_queries']}")
    print(f"  Avg total ms:     {stats['avg_total_ms']}")
    print(f"  Avg LLM ms:       {stats['avg_llm_ms']}")
    print(f"  Avg similarity:   {stats['avg_similarity']}")
    print(f"  Intent dist:      {stats['intent_distribution']}")

    # Cleanup test db
    os.remove(test_db)
    print("\n✅ All logger tests passed")