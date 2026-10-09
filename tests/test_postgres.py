"""Runs only when DATABASE_URL points at PostgreSQL (CI service container); proves the SQL library is dialect-portable."""
import os

import pytest
from conftest import SMALL_GEN

from riskplatform.analytics import run_all
from riskplatform.db import get_engine, load_tables
from riskplatform.generate import generate

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL")


def test_sql_library_on_postgres(tmp_path):
    eng = get_engine()
    load_tables(eng, generate(SMALL_GEN))
    res = run_all(eng, tmp_path)
    assert len(res) >= 10 and all(len(d) > 0 for d in res.values())
