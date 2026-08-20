"""Deletion. The Recycle Bin is the undo -- there is no quarantine folder."""

from __future__ import annotations

import logging
import os
from typing import Sequence

from send2trash import send2trash

from app import db
from app.thumbs import discard_thumb

log = logging.getLogger(__name__)


def delete_images(conn, image_ids: Sequence[int]) -> dict:
    """send2trash each file, then drop its rows.

    send2trash fails on network drives and on removable media with no Recycle
    Bin. Report that to the user rather than falling back to a real delete.
    """
    deleted: list[int] = []
    failed: list[dict] = []
    freed = 0

    rows = db.images_by_id(conn, list(image_ids))
    for image_id in image_ids:
        row = rows.get(int(image_id))
        if row is None:
            failed.append({"id": image_id, "path": None, "error": "not in the index"})
            continue
        try:
            send2trash(os.fspath(row["path"]))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            log.warning("could not trash %s: %s", row["path"], exc)
            failed.append({"id": int(image_id), "path": row["path"], "error": str(exc)})
            continue
        deleted.append(int(image_id))
        freed += int(row["size"])
        discard_thumb(int(image_id))

    db.forget_images(conn, deleted)
    return {"deleted": len(deleted), "freed_bytes": freed, "failed": failed}
