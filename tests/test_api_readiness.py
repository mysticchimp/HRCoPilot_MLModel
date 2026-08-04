"""API readiness gate: /score must not run before lifespan warm completes."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


def test_score_returns_503_when_models_not_ready():
    import api.main as main

    with patch.object(main, "_models_ready", False), patch.object(
        main, "_embedding_model", None
    ):
        with pytest.raises(HTTPException) as exc:
            main._require_ready()
        assert exc.value.status_code == 503


def test_health_reports_warming_until_ready():
    import api.main as main

    with patch.object(main, "_models_ready", False), patch.object(
        main, "_embedding_model", None
    ), patch.object(main, "_startup_rss_mb", None):
        # Avoid loading real models via TestClient lifespan — call health directly.
        body = main.health()
        assert body["models_ready"] is False
        assert body["status"] == "warming"
