#!/usr/bin/env python3
import sqlite3
import tempfile
import os
import time
import json
from types import SimpleNamespace

import torch
import numpy as np

import clustering

 
class DummyYoloWrapper:
    def __init__(self):
        # Provide a dummy callable at self.model.model and detection_layer_entry
        def model_forward(images):
            B = images.shape[0]
            # three heads: simulate (B, C, H, W) with small sizes
            return [torch.randn(B, 8, 4, 4), torch.randn(B, 16, 2, 2), torch.randn(B, 32, 1, 1)]

        self.model = SimpleNamespace(model=model_forward)
        self.detection_layer_entry = [SimpleNamespace(features=None) for _ in range(2)]

    def _set_pooled_features(self, images):
        B = images.shape[0]
        for i in range(len(self.detection_layer_entry)):
            self.detection_layer_entry[i].features = torch.randn(B, 8)

    def predict_and_store_raw(self, images, img_ids, emb_conn, model_name, **kwargs):
        # reuse clustering.tensors_to_heads_blob etc.
        B = images.shape[0]
        rows = []
        cur = emb_conn.cursor()
        for i in range(B):
            # create three dummy head tensors per image
            heads = [torch.randn(8, 4, 4), torch.randn(16, 2, 2), torch.randn(32, 1, 1)]
            head_blob, shapes_json, dtypes_json = clustering.tensors_to_heads_blob(heads, compress=True)
            features_blob = clustering.array_to_blob(np.random.randn(8).astype(np.float32))
            created_ts = int(time.time())
            rows.append((int(img_ids[i]), model_name, head_blob, shapes_json, dtypes_json, json.dumps({}), features_blob, created_ts))
        cur.executemany(
            "INSERT OR REPLACE INTO predictions_raw_heads (img_id, model_name, head_blob, head_shapes, head_dtypes, anchors_info, features, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        emb_conn.commit()
        return {"ok": True}


def main():
    # create DB in tmp
    fd, db_path = tempfile.mkstemp(prefix="pred_raw_", suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS predictions_raw_heads (
                id INTEGER PRIMARY KEY,
                img_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                head_blob BLOB NOT NULL,
                head_shapes TEXT NOT NULL,
                head_dtypes TEXT NOT NULL,
                anchors_info TEXT,
                features BLOB,
                created_ts INTEGER,
                UNIQUE(img_id, model_name)
            )"""
    )
    conn.commit()

    B = 2
    images = torch.randn(B, 3, 32, 32)
    img_ids = [1001, 1002]

    wrapper = DummyYoloWrapper()
    wrapper._set_pooled_features(images)
    res = wrapper.predict_and_store_raw(images=images, img_ids=img_ids, emb_conn=conn, model_name="yolo_dummy")

    cur.execute("SELECT COUNT(*) FROM predictions_raw_heads")
    count = cur.fetchone()[0]
    print(f"Inserted rows: {count}")
    print(f"DB path: {db_path}")
    conn.close()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import sqlite3
import tempfile
import os
import time
import json
from types import SimpleNamespace

import torch
import numpy as np

import clustering


class DummyYoloWrapper:
    def __init__(self):
        # Provide a dummy callable at self.model.model and detection_layer_entry
        def model_forward(images):
            B = images.shape[0]
            # three heads: simulate (B, C, H, W) with small sizes
            return [torch.randn(B, 8, 4, 4), torch.randn(B, 16, 2, 2), torch.randn(B, 32, 1, 1)]

        self.model = SimpleNamespace(model=model_forward)
        self.detection_layer_entry = [SimpleNamespace(features=None) for _ in range(2)]

    def _set_pooled_features(self, images):
        B = images.shape[0]
        for i in range(len(self.detection_layer_entry)):
            self.detection_layer_entry[i].features = torch.randn(B, 8)

    def predict_and_store_raw(self, images, img_ids, emb_conn, model_name, **kwargs):
        # reuse clustering.tensors_to_heads_blob etc.
        B = images.shape[0]
        rows = []
        cur = emb_conn.cursor()
        for i in range(B):
            # create three dummy head tensors per image
            heads = [torch.randn(8, 4, 4), torch.randn(16, 2, 2), torch.randn(32, 1, 1)]
            head_blob, shapes_json, dtypes_json = clustering.tensors_to_heads_blob(heads, compress=True)
            features_blob = clustering.array_to_blob(np.random.randn(8).astype(np.float32))
            created_ts = int(time.time())
            rows.append((int(img_ids[i]), model_name, head_blob, shapes_json, dtypes_json, json.dumps({}), features_blob, created_ts))
        cur.executemany(
            "INSERT OR REPLACE INTO predictions_raw_heads (img_id, model_name, head_blob, head_shapes, head_dtypes, anchors_info, features, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        emb_conn.commit()
        return {"ok": True}


def main():
    # create DB in tmp
    fd, db_path = tempfile.mkstemp(prefix="pred_raw_", suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS predictions_raw_heads (
                id INTEGER PRIMARY KEY,
                img_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                head_blob BLOB NOT NULL,
                head_shapes TEXT NOT NULL,
                head_dtypes TEXT NOT NULL,
                anchors_info TEXT,
                features BLOB,
                created_ts INTEGER,
                UNIQUE(img_id, model_name)
            )"""
    )
    conn.commit()

    B = 2
    images = torch.randn(B, 3, 32, 32)
    img_ids = [1001, 1002]

    wrapper = DummyYoloWrapper()
    wrapper._set_pooled_features(images)
    res = wrapper.predict_and_store_raw(images=images, img_ids=img_ids, emb_conn=conn, model_name="yolo_dummy")

    cur.execute("SELECT COUNT(*) FROM predictions_raw_heads")
    count = cur.fetchone()[0]
    print(f"Inserted rows: {count}")
    print(f"DB path: {db_path}")
    conn.close()


if __name__ == "__main__":
    main()
