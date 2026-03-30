"""Auto-updater that checks GitHub releases for new versions."""

import json
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
from packaging.version import Version

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QProgressDialog
from PyQt6.QtCore import Qt

GITHUB_REPO = "kavanagh21/anipip"
RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


class UpdateChecker(QThread):
    """Background thread to check for updates."""

    update_available = pyqtSignal(str, str, str)  # version, download_url, release_notes
    no_update = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, current_version: str):
        super().__init__()
        self.current_version = current_version

    def run(self):
        try:
            req = Request(RELEASES_URL, headers={"Accept": "application/vnd.github.v3+json"})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            tag = data.get("tag_name", "")
            latest_version = tag.lstrip("v")

            if Version(latest_version) > Version(self.current_version):
                # Find the right asset for this platform
                download_url = self._find_asset_url(data.get("assets", []))
                release_notes = data.get("body", "")
                if download_url:
                    self.update_available.emit(latest_version, download_url, release_notes)
                else:
                    self.error.emit(
                        f"Version {latest_version} is available but no installer "
                        f"was found for your platform."
                    )
            else:
                self.no_update.emit()

        except URLError as e:
            self.error.emit(f"Could not connect to GitHub: {e.reason}")
        except Exception as e:
            self.error.emit(f"Error checking for updates: {e}")

    def _find_asset_url(self, assets: list) -> str | None:
        if sys.platform == "win32":
            suffix = "-Windows-Setup.exe"
        elif sys.platform == "darwin":
            suffix = "-macOS.dmg"
        else:
            return None

        for asset in assets:
            name = asset.get("name", "")
            if name.endswith(suffix):
                return asset.get("browser_download_url")
        return None


class UpdateDownloader(QThread):
    """Background thread to download the update installer."""

    progress = pyqtSignal(int, int)  # bytes_downloaded, total_bytes
    finished = pyqtSignal(str)  # local file path
    error = pyqtSignal(str)

    def __init__(self, url: str, filename: str):
        super().__init__()
        self.url = url
        self.filename = filename

    def run(self):
        try:
            req = Request(self.url)
            with urlopen(req, timeout=300) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                tmp_dir = tempfile.mkdtemp(prefix="anipip_update_")
                file_path = os.path.join(tmp_dir, self.filename)

                downloaded = 0
                chunk_size = 256 * 1024

                with open(file_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(downloaded, total)

            self.finished.emit(file_path)

        except Exception as e:
            self.error.emit(f"Download failed: {e}")


def launch_installer(file_path: str) -> None:
    """Launch the downloaded installer and exit the app."""
    if sys.platform == "win32":
        subprocess.Popen([file_path], shell=False)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", file_path])
    sys.exit(0)
