"""Batch SQL operations for performance.

Uses executemany() instead of loops for bulk operations.
"""

from typing import List, Dict, Any
import sqlite3


def batch_insert(
    conn: sqlite3.Connection, table: str, rows: List[Dict[str, Any]], batch_size: int = 1000
) -> int:
    """Batch insert rows using executemany()."""
    if not rows:
        return 0

    columns = list(rows[0].keys())
    placeholders = ",".join(["?" for _ in columns])
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"

    cursor = conn.cursor()
    data = [[row.get(col) for col in columns] for row in rows]

    total_inserted = 0
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        cursor.executemany(sql, batch)
        total_inserted += len(batch)

    conn.commit()
    return total_inserted


def batch_update(
    conn: sqlite3.Connection,
    table: str,
    updates: List[Dict[str, Any]],
    where_column: str,
    batch_size: int = 1000,
) -> int:
    """Batch update rows using executemany()."""
    if not updates:
        return 0

    cursor = conn.cursor()
    total_updated = 0

    for i in range(0, len(updates), batch_size):
        batch = updates[i : i + batch_size]
        for row in batch:
            where_value = row.pop(where_column)
            cols = ",".join([f"{k}=?" for k in row.keys()])
            sql = f"UPDATE {table} SET {cols} WHERE {where_column}=?"
            values = list(row.values()) + [where_value]
            cursor.execute(sql, values)
            total_updated += 1

    conn.commit()
    return total_updated


__all__ = ["batch_insert", "batch_update"]
