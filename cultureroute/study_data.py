"""Loader, validator, and prepared-view builder for CultureRoute synthetic research data.

Loads SmartTrip (PS-01.db) and WanderMatch (PS-11.db) read-only from the ZIP archives,
verifies identical shared tables, confirms all 8 study checkpoints, and provides
canonical prepared views at their documented grains.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import sqlite3
import tempfile
import zipfile
from typing import Dict, Generator, List, Set, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent
SMART_TRIP_ZIP = ROOT_DIR / "SmartTrip (1).zip"
WANDER_MATCH_ZIP = ROOT_DIR / "WanderMatch.zip"

SMART_DB_ENTRY = "SmartTrip/data/PS-01.db"
WANDER_DB_ENTRY = "WanderMatch/data/PS-11.db"

CULTURAL_CATEGORIES = ("heritage", "museum", "religious", "food")


@dataclass(frozen=True)
class AuditReport:
    smart_tables_count: int
    smart_rows_count: int
    wander_tables_count: int
    wander_rows_count: int
    shared_tables_count: int
    shared_rows_count: int
    distinct_tables_count: int
    distinct_rows_count: int
    india_cities_count: int
    india_attractions_count: int
    study_trips_count: int
    matched_interactions_count: int
    selected_stops_count: int
    cultural_attractions_count: int
    cultural_interactions_count: int
    cultural_selected_stops_count: int
    distinct_saved_pois: int
    distinct_planned_pois: int
    distinct_inner_pois: int
    saved_only_pois: int
    planned_only_pois: int
    full_outer_pois: int
    india_events_count: int
    events_missing_footfall: int
    complete_weeks_count: int
    forecast_last_week_mae: float
    forecast_four_week_mae: float


def _scoped_selected_sql(expression: str = "count(*)") -> str:
    return f"""
        SELECT {expression}
        FROM itinerary_items ii
        JOIN itineraries i ON i.itinerary_id = ii.itinerary_id
        JOIN trips t ON t.trip_id = i.trip_id
        JOIN cities c ON c.city_id = t.destination_city_id
        JOIN activities_poi a ON a.poi_id = ii.entity_id
        WHERE c.country_code = 'IN'
          AND t.status <> 'cancelled'
          AND i.is_active = 1
          AND ii.status = 'confirmed'
          AND ii.item_type = 'poi'
          AND ii.entity_type = 'poi'
          AND a.city_id = t.destination_city_id
    """


# Persistent extracted SQLite cache paths for maximum query throughput
CACHE_DIR = ROOT_DIR / "data"
CACHE_SMART_DB = CACHE_DIR / ".cache_smart.db"
CACHE_WANDER_DB = CACHE_DIR / ".cache_wander.db"


def _ensure_extracted_cache() -> Tuple[Path, Path]:
    """Ensure both study databases are extracted once to persistent disk cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    smart_needs_extract = not CACHE_SMART_DB.exists()
    if not smart_needs_extract:
        smart_needs_extract = CACHE_SMART_DB.stat().st_mtime < SMART_TRIP_ZIP.stat().st_mtime

    if smart_needs_extract:
        with zipfile.ZipFile(SMART_TRIP_ZIP) as z1:
            CACHE_SMART_DB.write_bytes(z1.read(SMART_DB_ENTRY))

    wander_needs_extract = not CACHE_WANDER_DB.exists()
    if not wander_needs_extract:
        wander_needs_extract = CACHE_WANDER_DB.stat().st_mtime < WANDER_MATCH_ZIP.stat().st_mtime

    if wander_needs_extract:
        with zipfile.ZipFile(WANDER_MATCH_ZIP) as z2:
            CACHE_WANDER_DB.write_bytes(z2.read(WANDER_DB_ENTRY))

    return CACHE_SMART_DB, CACHE_WANDER_DB


@contextmanager
def open_study_databases(
    smart_db_path: Optional[Path] = None,
    wander_db_path: Optional[Path] = None
) -> Generator[Tuple[sqlite3.Connection, sqlite3.Connection], None, None]:
    """Open persistent extracted SQLite databases in read-only mode with zero unzipping overhead."""
    if smart_db_path and wander_db_path:
        smart_path, wander_path = smart_db_path, wander_db_path
    else:
        if not SMART_TRIP_ZIP.exists():
            raise FileNotFoundError(f"Missing SmartTrip archive at {SMART_TRIP_ZIP}")
        if not WANDER_MATCH_ZIP.exists():
            raise FileNotFoundError(f"Missing WanderMatch archive at {WANDER_MATCH_ZIP}")
        smart_path, wander_path = _ensure_extracted_cache()

    smart_conn: sqlite3.Connection | None = None
    wander_conn: sqlite3.Connection | None = None

    try:
        smart_conn = sqlite3.connect(f"{smart_path.as_uri()}?mode=ro", uri=True)
        wander_conn = sqlite3.connect(f"{wander_path.as_uri()}?mode=ro", uri=True)
        smart_conn.row_factory = sqlite3.Row
        wander_conn.row_factory = sqlite3.Row

        # Attach WanderMatch database as 'wm' to smart_conn for joined queries
        smart_conn.execute("ATTACH DATABASE ? AS wm", (f"{wander_path.as_uri()}?mode=ro",))

        yield smart_conn, wander_conn

    finally:
        if smart_conn:
            try:
                smart_conn.close()
            except Exception:
                pass
        if wander_conn:
            try:
                wander_conn.close()
            except Exception:
                pass


@lru_cache(maxsize=1)
def verify_and_audit_study() -> AuditReport:
    """Run full verification across the 24 distinct tables and 8 checkpoints."""
    with open_study_databases() as (smart, wander):
        smart_tables = {
            r[0] for r in smart.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        wander_tables = {
            r[0] for r in wander.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        shared_tables = smart_tables & wander_tables

        if len(smart_tables) != 18 or len(wander_tables) != 14 or len(shared_tables) != 8:
            raise AssertionError(
                f"Expected 18/14/8 tables, got {len(smart_tables)}/{len(wander_tables)}/{len(shared_tables)}"
            )

        # Verify shared tables identical by schema and row content
        for tbl in sorted(shared_tables):
            schema_smart = smart.execute(f'PRAGMA table_info("{tbl}")').fetchall()
            schema_wander = wander.execute(f'PRAGMA table_info("{tbl}")').fetchall()
            if schema_smart != schema_wander:
                raise AssertionError(f"Schema mismatch in shared table '{tbl}'")

            pk = next(r[1] for r in schema_smart if r[5] == 1)
            rows_smart = smart.execute(f'SELECT * FROM "{tbl}" ORDER BY "{pk}"').fetchall()
            rows_wander = wander.execute(f'SELECT * FROM "{tbl}" ORDER BY "{pk}"').fetchall()
            if rows_smart != rows_wander:
                raise AssertionError(f"Content mismatch in shared table '{tbl}'")

        total_smart_rows = sum(
            int(smart.execute(f'SELECT count(*) FROM "{n}"').fetchone()[0]) for n in smart_tables
        )
        total_wander_rows = sum(
            int(wander.execute(f'SELECT count(*) FROM "{n}"').fetchone()[0]) for n in wander_tables
        )
        shared_rows = sum(
            int(smart.execute(f'SELECT count(*) FROM "{n}"').fetchone()[0]) for n in shared_tables
        )

        distinct_tables = smart_tables | wander_tables
        distinct_rows = total_smart_rows + total_wander_rows - shared_rows

        if (total_smart_rows, total_wander_rows, shared_rows) != (32627, 27428, 11327):
            raise AssertionError(
                f"Unexpected row counts: {total_smart_rows}, {total_wander_rows}, {shared_rows}"
            )
        if distinct_rows != 48728:
            raise AssertionError(f"Distinct rows expected 48,728, got {distinct_rows}")

        # Core Study Checkpoints
        city_count = int(smart.execute("SELECT count(*) FROM cities WHERE country_code='IN'").fetchone()[0])
        poi_count = int(smart.execute(
            "SELECT count(*) FROM activities_poi a JOIN cities c USING(city_id) WHERE c.country_code='IN'"
        ).fetchone()[0])
        trip_count = int(smart.execute(
            "SELECT count(*) FROM trips t JOIN cities c ON c.city_id=t.destination_city_id "
            "WHERE c.country_code='IN' AND t.status<>'cancelled'"
        ).fetchone()[0])

        interaction_where = (
            "FROM wm.user_interactions u "
            "JOIN activities_poi a ON u.entity_type='poi' AND u.entity_id=a.poi_id "
            "JOIN cities c ON c.city_id=a.city_id WHERE c.country_code='IN'"
        )
        interaction_count = int(smart.execute(f"SELECT count(*) {interaction_where}").fetchone()[0])
        selected_count = int(smart.execute(_scoped_selected_sql()).fetchone()[0])

        cultural_pred = "('heritage','museum','religious','food')"
        cultural_poi = int(smart.execute(
            f"SELECT count(*) FROM activities_poi a JOIN cities c USING(city_id) "
            f"WHERE c.country_code='IN' AND a.poi_category IN {cultural_pred}"
        ).fetchone()[0])
        cultural_interactions = int(smart.execute(
            f"SELECT count(*) {interaction_where} AND a.poi_category IN {cultural_pred}"
        ).fetchone()[0])
        cultural_selected = int(smart.execute(
            _scoped_selected_sql() + f" AND a.poi_category IN {cultural_pred}"
        ).fetchone()[0])

        if (city_count, poi_count, trip_count, interaction_count, selected_count) != (45, 675, 411, 6850, 3191):
            raise AssertionError("Study counts do not match specification baseline (45, 675, 411, 6850, 3191)")
        if (cultural_poi, cultural_interactions, cultural_selected) != (211, 2144, 961):
            raise AssertionError("Cultural share counts do not match baseline (211, 2144, 961)")

        # Saved vs Planned Set Operations
        saved = {
            r[0] for r in smart.execute(
                f"SELECT DISTINCT u.entity_id {interaction_where} AND u.interaction_type='save'"
            )
        }
        planned = {
            r[0] for r in smart.execute(_scoped_selected_sql("DISTINCT ii.entity_id"))
        }

        # Indian Events & Footfall
        events_row = smart.execute(
            "SELECT count(*), sum(e.expected_footfall IS NULL) "
            "FROM events_festivals e JOIN cities c USING(city_id) WHERE c.country_code='IN'"
        ).fetchone()
        events_count = int(events_row[0])
        events_missing = int(events_row[1])

        # Weekly cultural forecast series
        timestamps = smart.execute(
            f"SELECT u.occurred_at {interaction_where} AND a.poi_category IN {cultural_pred}"
        ).fetchall()
        from collections import Counter
        weekly: Counter = Counter()
        for (ts,) in timestamps:
            d = datetime.fromisoformat(ts).date()
            weekly[d - timedelta(days=d.weekday())] += 1
        weeks = sorted(weekly)
        complete_weeks = weeks[:-1]
        values = [weekly[w] for w in complete_weeks]
        last_week_mae = sum(abs(values[i] - values[i - 1]) for i in range(len(values) - 12, len(values))) / 12
        four_week_mae = sum(abs(values[i] - sum(values[i - 4:i]) / 4) for i in range(len(values) - 12, len(values))) / 12

        return AuditReport(
            smart_tables_count=len(smart_tables),
            smart_rows_count=total_smart_rows,
            wander_tables_count=len(wander_tables),
            wander_rows_count=total_wander_rows,
            shared_tables_count=len(shared_tables),
            shared_rows_count=shared_rows,
            distinct_tables_count=len(distinct_tables),
            distinct_rows_count=distinct_rows,
            india_cities_count=city_count,
            india_attractions_count=poi_count,
            study_trips_count=trip_count,
            matched_interactions_count=interaction_count,
            selected_stops_count=selected_count,
            cultural_attractions_count=cultural_poi,
            cultural_interactions_count=cultural_interactions,
            cultural_selected_stops_count=cultural_selected,
            distinct_saved_pois=len(saved),
            distinct_planned_pois=len(planned),
            distinct_inner_pois=len(saved & planned),
            saved_only_pois=len(saved - planned),
            planned_only_pois=len(planned - saved),
            full_outer_pois=len(saved | planned),
            india_events_count=events_count,
            events_missing_footfall=events_missing,
            complete_weeks_count=len(complete_weeks),
            forecast_last_week_mae=round(last_week_mae, 2),
            forecast_four_week_mae=round(four_week_mae, 2),
        )


@lru_cache(maxsize=1)
def _cached_study_cities_tuple() -> tuple:
    from cultureroute.study_db import DEFAULT_STUDY_DB, open_study_cache
    if DEFAULT_STUDY_DB.exists():
        with open_study_cache() as conn:
            rows = conn.execute("""
                SELECT city_id, name, state, lat, lng, timezone, population, primary_language
                FROM cities
                WHERE country_code = 'IN'
                ORDER BY name ASC
            """).fetchall()
            return tuple(dict(r) for r in rows)
    with open_study_databases() as (smart, _):
        rows = smart.execute("""
            SELECT city_id, name, state, lat, lng, timezone, population, primary_language
            FROM cities
            WHERE country_code = 'IN'
            ORDER BY name ASC
        """).fetchall()
        return tuple(dict(r) for r in rows)


def get_study_cities() -> List[Dict]:
    """Retrieve the 45 Indian study cities with basic metadata."""
    return [dict(c) for c in _cached_study_cities_tuple()]


@lru_cache(maxsize=128)
def _cached_city_attractions_tuple(city_id: str) -> tuple:
    from cultureroute.study_db import DEFAULT_STUDY_DB, open_study_cache
    if DEFAULT_STUDY_DB.exists():
        with open_study_cache() as conn:
            matrix_origins = {
                r[0] for r in conn.execute(
                    "SELECT DISTINCT origin_poi_id FROM poi_travel_matrix WHERE origin_poi_id IN "
                    "(SELECT poi_id FROM activities_poi WHERE city_id = ?)", (city_id,)
                ).fetchall()
            }
            rows = conn.execute("""
                SELECT poi_id, city_id, name, poi_category, lat, lng, typical_duration_minutes,
                       entry_cost, currency, opens_at, closes_at, accessibility, description
                FROM activities_poi
                WHERE city_id = ?
                ORDER BY name ASC
            """, (city_id,)).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["in_travel_matrix"] = d["poi_id"] in matrix_origins
                results.append(d)
            return tuple(results)

    with open_study_databases() as (smart, _):
        matrix_origins = {
            r[0] for r in smart.execute(
                "SELECT DISTINCT origin_poi_id FROM poi_travel_matrix WHERE origin_poi_id IN "
                "(SELECT poi_id FROM activities_poi WHERE city_id = ?)", (city_id,)
            ).fetchall()
        }
        rows = smart.execute("""
            SELECT poi_id, city_id, name, poi_category, lat, lng, typical_duration_minutes,
                   entry_cost, currency, opens_at, closes_at, accessibility, description
            FROM activities_poi
            WHERE city_id = ?
            ORDER BY name ASC
        """, (city_id,)).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["in_travel_matrix"] = d["poi_id"] in matrix_origins
            results.append(d)
        return tuple(results)


def get_city_attractions(city_id: str) -> List[Dict]:
    """Retrieve the 15 attractions for a study city, marking travel-matrix availability."""
    return [dict(a) for a in _cached_city_attractions_tuple(city_id)]