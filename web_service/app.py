"""Flask web service for the Area Target processing pipeline."""

import os
import shutil
import threading
import uuid
import zipfile
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_file, send_from_directory

app = Flask(__name__, static_folder="static")

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/pipeline_uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/pipeline_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory job store
jobs = {}


def find_scan_root(extract_dir):
    """Find the directory containing model.obj and poses.json inside extracted zip."""
    for root, dirs, files in os.walk(extract_dir):
        if "model.obj" in files and "poses.json" in files:
            return root
    return None


def run_pipeline(job_id, zip_path):
    """Run the optimized pipeline in a background thread."""
    job = jobs[job_id]
    extract_dir = os.path.join(UPLOAD_DIR, job_id, "extracted")
    output_dir = os.path.join(OUTPUT_DIR, job_id)

    try:
        # Extract zip
        job["status"] = "extracting"
        job["step"] = "解压 ZIP 文件"
        job["progress"] = 5
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        scan_root = find_scan_root(extract_dir)
        if scan_root is None:
            job["status"] = "failed"
            job["error"] = "ZIP 中未找到 model.obj 和 poses.json"
            return

        from processing_pipeline.optimized_pipeline import OptimizedPipeline

        optimizer_url = os.environ.get("MODEL_OPTIMIZER_URL", "http://model_optimizer:3000")
        pipeline = OptimizedPipeline(optimizer_url=optimizer_url)

        # Step 1: Input validation
        job["status"] = "processing"
        job["step"] = "1/4 输入验证"
        job["progress"] = 15

        # Step 2: Model optimization
        job["step"] = "2/4 模型优化"
        job["progress"] = 35

        # Step 3: Feature extraction
        job["step"] = "3/4 特征提取"
        job["progress"] = 60

        # Step 4: Asset bundling
        job["step"] = "4/4 资产打包"
        job["progress"] = 80

        os.makedirs(output_dir, exist_ok=True)
        pipeline.run(scan_root, output_dir)

        # Create downloadable zip
        result_zip = os.path.join(OUTPUT_DIR, f"{job_id}.zip")
        shutil.make_archive(result_zip.replace(".zip", ""), "zip", output_dir)

        job["status"] = "completed"
        job["step"] = "完成"
        job["progress"] = 100
        job["result_zip"] = result_zip
        job["finished_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["progress"] = 0


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".zip"):
        return jsonify({"error": "请上传 ZIP 文件"}), 400

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    zip_path = os.path.join(job_dir, "upload.zip")
    f.save(zip_path)

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "step": "等待处理",
        "progress": 0,
        "error": None,
        "result_zip": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }

    t = threading.Thread(target=run_pipeline, args=(job_id, zip_path), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(job)


@app.route("/api/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "completed":
        return jsonify({"error": "资产包未就绪"}), 404
    return send_file(job["result_zip"], as_attachment=True, download_name=f"asset_bundle_{job_id}.zip")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
