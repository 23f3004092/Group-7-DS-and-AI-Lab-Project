import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_downloader_module(file_path: Path):
    """Load a download_data.py file dynamically into a separate module namespace."""
    spec = importlib.util.spec_from_file_location("download_data_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDownloadData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_script = PROJECT_ROOT / "scripts" / "download_data.py"
        cls.module = load_downloader_module(cls.data_script)

    def test_get_api_key_from_cli(self):
        """Verify that an explicit CLI argument takes precedence for API key."""
        key = self.module.get_api_key(cli_key="cli_secret_key")
        self.assertEqual(key, "cli_secret_key")

    @patch.dict(os.environ, {"DATA_GOV_KEY": "env_secret_key"}, clear=True)
    def test_get_api_key_from_env(self):
        """Verify that environment variable DATA_GOV_KEY is used when CLI arg is not passed."""
        # Temporarily patch out .env check or let env var take priority
        key = self.module.get_api_key(cli_key=None)
        self.assertEqual(key, "env_secret_key")

    def test_get_api_key_from_dotenv_file(self):
        """Verify that get_api_key correctly reads DATA_GOV_KEY from .env at project root."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            dotenv_path = tmp_root / ".env"
            dotenv_path.write_text(
                "# Sample .env file\n"
                "OTHER_VAR=123\n"
                "DATA_GOV_KEY=\"test_dotenv_key_579b464db6\"\n",
                encoding="utf-8",
            )
            with patch.object(self.module, "PROJECT_ROOT", tmp_root), patch.dict(os.environ, {}, clear=True):
                # Remove any KCC/DATA_GOV keys from os.environ
                for k in ["DATA_GOV_KEY", "KCC_API_KEY", "API_KEY"]:
                    os.environ.pop(k, None)
                key = self.module.get_api_key(cli_key=None)
                self.assertEqual(key, "test_dotenv_key_579b464db6")

    def test_parse_months_valid(self):
        """Test valid month specifications ("1-12", "1,3,7", "6", ""/None)."""
        self.assertEqual(self.module._parse_months("1-4"), [1, 2, 3, 4])
        self.assertEqual(self.module._parse_months("1, 3, 7-9"), [1, 3, 7, 8, 9])
        self.assertEqual(self.module._parse_months("6"), [6])
        self.assertEqual(self.module._parse_months(""), [None])
        self.assertEqual(self.module._parse_months("all"), [None])

    def test_parse_years_valid(self):
        """Test valid year specifications ("2020-2025", "2020,2023", "2025", ""/None)."""
        self.assertEqual(self.module._parse_years("2020-2023"), ["2020", "2021", "2022", "2023"])
        self.assertEqual(self.module._parse_years("2020, 2023, 2025"), ["2020", "2023", "2025"])
        self.assertEqual(self.module._parse_years("2025"), ["2025"])
        self.assertEqual(self.module._parse_years(""), [None])



    def test_parse_months_invalid(self):
        """Test invalid month specifications raise ValueError."""
        with self.assertRaises(ValueError):
            self.module._parse_months("0-5")
        with self.assertRaises(ValueError):
            self.module._parse_months("13")
        with self.assertRaises(ValueError):
            self.module._parse_months("abc")

    def test_kcc_fetch_page_json_params(self):
        """Verify _kcc_fetch_page constructs exact query parameters for format=json."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"records": [{"id": 1}], "total": "1"}
        mock_session.get.return_value = mock_response

        res = self.module._kcc_fetch_page(
            session=mock_session,
            api_key="my_key",
            state="UTTAR PRADESH",
            year="2025",
            month=7,
            offset=10,
            limit=50,
            fmt="json",
        )

        mock_session.get.assert_called_once()
        url, kwargs = mock_session.get.call_args
        self.assertEqual(url[0], "https://api.data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f")
        self.assertEqual(
            kwargs["params"],
            {
                "api-key": "my_key",
                "format": "json",
                "offset": 10,
                "limit": 50,
                "filters[StateName]": "UTTAR PRADESH",
                "filters[year]": "2025",
                "filters[month]": "7",
            },
        )
        self.assertEqual(res, {"records": [{"id": 1}], "total": "1"})

    def test_kcc_fetch_page_xml_and_csv(self):
        """Verify _kcc_fetch_page returns raw text response when format is xml or csv."""
        for fmt in ["xml", "csv"]:
            mock_session = MagicMock()
            mock_response = MagicMock()
            expected_text = f"<{fmt}>mock data</{fmt}>" if fmt == "xml" else "col1,col2\n1,2"
            mock_response.text = expected_text
            mock_session.get.return_value = mock_response

            res = self.module._kcc_fetch_page(
                session=mock_session,
                api_key="my_key",
                state="ALL",  # ALL should omit filters[StateName]
                year="2025",
                month=1,
                offset=0,
                limit=10,
                fmt=fmt,
            )
            _, kwargs = mock_session.get.call_args
            self.assertEqual(kwargs["params"]["format"], fmt)
            self.assertNotIn("filters[StateName]", kwargs["params"])
            self.assertEqual(res, expected_text)

    def test_kcc_download_batch_xml_format(self):
        """Verify _kcc_download_batch saves XML file when fmt='xml'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir)
            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "<response><row><query>Test Query</query></row></response>"
            mock_session.get.return_value = mock_response

            count = self.module._kcc_download_batch(
                session=mock_session,
                api_key="key",
                state="UTTAR PRADESH",
                year="2025",
                month=5,
                page_size=100,
                dest=dest,
                fmt="xml",
                start_offset=0,
                max_limit=10,
            )

            self.assertEqual(count, 1)
            out_file = dest / "kcc_uttar_pradesh_2025_month_05.xml"
            self.assertTrue(out_file.exists())
            self.assertIn("<query>Test Query</query>", out_file.read_text(encoding="utf-8"))

    def test_kcc_download_batch_json_pagination_and_no_month(self):
        """Verify _kcc_download_batch paginates correctly when fmt='json' and month is None."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir)
            mock_session = MagicMock()

            # First call (limit=1) returns total=3
            resp_first = MagicMock()
            resp_first.json.return_value = {"total": "3", "records": [{"id": 1}]}

            # Subsequent pagination calls
            resp_page1 = MagicMock()
            resp_page1.json.return_value = {"total": "3", "records": [{"id": 1, "q": "a"}, {"id": 2, "q": "b"}]}
            resp_page2 = MagicMock()
            resp_page2.json.return_value = {"total": "3", "records": [{"id": 3, "q": "c"}]}

            mock_session.get.side_effect = [resp_first, resp_page1, resp_page2]

            with patch.object(self.module, "KCC_REQUEST_DELAY_S", 0):
                count = self.module._kcc_download_batch(
                    session=mock_session,
                    api_key="key",
                    state="UTTAR PRADESH",
                    year="2025",
                    month=None,  # No month specified -> all months
                    page_size=2,
                    dest=dest,
                    fmt="json",
                )

            self.assertEqual(count, 3)
            out_file = dest / "kcc_uttar_pradesh_2025.jsonl"
            self.assertTrue(out_file.exists())
            lines = out_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(json.loads(lines[0])["id"], 1)
            self.assertEqual(json.loads(lines[2])["id"], 3)


    def test_download_kcc_end_to_end(self):
        """Verify download_kcc main logic handles key loading and format options."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            raw_dir = tmp_root / "data" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            mock_session = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "<data><row>mock</row></data>"
            mock_session.get.return_value = mock_response

            with patch.object(self.module, "RAW_DIR", raw_dir), \
                 patch.object(self.module, "PROJECT_ROOT", tmp_root), \
                 patch("requests.Session", return_value=mock_session):
                
                success = self.module.download_kcc(
                    api_key="test_api_key",
                    state="PUNJAB",
                    year="2025",
                    months_spec="6",
                    fmt="xml",
                    offset=0,
                    limit=5,
                )
                self.assertTrue(success)
                out_file = raw_dir / "kcc" / "kcc_punjab_2025_month_06.xml"
                self.assertTrue(out_file.exists())
                self.assertEqual(out_file.read_text(encoding="utf-8"), "<data><row>mock</row></data>")


if __name__ == "__main__":
    unittest.main()
