import unittest
from unittest.mock import Mock, patch

from backend import readiness


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        readiness._last_recovery_attempt = 0.0

    @patch.dict("os.environ", {}, clear=False)
    def test_recovery_without_control_plane_uses_bounded_retry(self):
        with patch.dict(
            "os.environ",
            {
                "QDRANT_RECOVERY_WEBHOOK_URL": "",
                "QDRANT_RESTART_COMMAND": "",
            },
        ):
            result = readiness._run_qdrant_recovery()

        self.assertFalse(result["attempted"])
        self.assertEqual(result["method"], "retry")

    @patch("backend.readiness.requests.post")
    def test_recovery_webhook_is_called_without_exposing_credentials(self, post):
        response = Mock()
        response.raise_for_status.return_value = None
        post.return_value = response

        with patch.dict(
            "os.environ",
            {
                "QDRANT_RECOVERY_WEBHOOK_URL": "https://recovery.example.test/qdrant",
                "QDRANT_RECOVERY_WEBHOOK_TOKEN": "secret-value",
                "QDRANT_RESTART_COMMAND": "",
            },
        ):
            result = readiness._run_qdrant_recovery()

        self.assertTrue(result["attempted"])
        self.assertEqual(result["method"], "webhook")
        self.assertNotIn("secret-value", str(result))
        post.assert_called_once()

    @patch("backend.readiness.requests.post")
    @patch("backend.readiness.requests.get")
    def test_qdrant_cloud_recovery_restarts_an_active_cluster(self, get, post):
        get_response = Mock()
        get_response.json.return_value = {"cluster": {"state": {"phase": "HEALTHY"}}}
        get_response.raise_for_status.return_value = None
        post_response = Mock()
        post_response.raise_for_status.return_value = None
        get.return_value = get_response
        post.return_value = post_response

        with patch.dict(
            "os.environ",
            {
                "QDRANT_CLOUD_MANAGEMENT_KEY": "management-secret",
                "QDRANT_CLOUD_ACCOUNT_ID": "account-id",
                "QDRANT_CLOUD_CLUSTER_ID": "cluster-id",
                "QDRANT_RECOVERY_WEBHOOK_URL": "",
                "QDRANT_RESTART_COMMAND": "",
            },
        ):
            result = readiness._run_qdrant_recovery()

        self.assertTrue(result["attempted"])
        self.assertEqual(result["method"], "qdrant_cloud_restart")
        self.assertTrue(post.call_args.args[0].endswith("/restart"))
        self.assertNotIn("management-secret", str(result))

    @patch("backend.readiness.check_local_knowledge")
    @patch("backend.readiness.check_langfuse")
    @patch("backend.readiness.check_chat_storage")
    @patch("backend.readiness.check_embeddings")
    @patch("backend.readiness.check_gemini")
    @patch("backend.readiness.check_qdrant")
    def test_critical_failure_blocks_workspace(
        self,
        qdrant,
        gemini,
        embeddings,
        chat_storage,
        langfuse,
        local_knowledge,
    ):
        def result(check_id, status, critical):
            return {
                "id": check_id,
                "label": check_id,
                "status": status,
                "message": "test",
                "critical": critical,
                "duration_ms": 1,
            }

        qdrant.return_value = [
            result("qdrant", "failed", True),
            result("qdrant_collections", "failed", True),
        ]
        gemini.return_value = result("gemini", "passed", True)
        embeddings.return_value = result("embeddings", "passed", True)
        local_knowledge.return_value = result("local_knowledge", "passed", True)
        chat_storage.return_value = result("chat_storage", "warning", False)
        langfuse.return_value = result("langfuse", "warning", False)

        payload = readiness.run_readiness_checks()

        self.assertFalse(payload["can_continue"])
        self.assertEqual(payload["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
