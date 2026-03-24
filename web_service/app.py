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
_jobs_lock = threading.Lock()


def _update_job(job_id, **kwargs):
    """Thread-safe helper to update job fields."""
    with _jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


def find_scan_root(extract_dir):
    """Find the directory containing model.obj and poses.json inside extracted zip."""
    for root, dirs, files in os.walk(extract_dir):
        if "model.obj" in files and "poses.json" in files:
            return root
    return None

def safe_extract(zf, extract_dir, max_size=500 * 1024 * 1024):
    """Safely extract a ZIP file, rejecting path traversal and enforcing size limits.

    Args:
        zf: An open zipfile.ZipFile object.
        extract_dir: Target directory for extraction.
        max_size: Maximum total uncompressed size in bytes (default 500MB).

    Raises:
        ValueError: If a ZIP entry contains path traversal or total size exceeds max_size.
    """
    real_extract_dir = os.path.realpath(extract_dir)
    total_size = 0

    for entry in zf.infolist():
        # Check for path traversal
        target_path = os.path.realpath(os.path.join(extract_dir, entry.filename))
        if not target_path.startswith(real_extract_dir + os.sep) and target_path != real_extract_dir:
            raise ValueError(
                f"Path traversal detected in ZIP entry: {entry.filename}"
            )

        # Accumulate uncompressed size
        total_size += entry.file_size
        if total_size > max_size:
            raise ValueError(
                f"ZIP extraction would exceed size limit of {max_size} bytes "
                f"(accumulated {total_size} bytes)"
            )

        # Extract this single entry safely
        zf.extract(entry, extract_dir)

        # Post-check: verify actual written path (防 TOCTOU)
        actual_path = os.path.realpath(os.path.join(extract_dir, entry.filename))
        if not actual_path.startswith(real_extract_dir + os.sep) and actual_path != real_extract_dir:
            # Remove the extracted file and raise
            if os.path.exists(actual_path):
                os.remove(actual_path)
            raise ValueError(
                f"Post-extraction path traversal detected: {entry.filename}"
            )




def run_pipeline(job_id, zip_path, uv_unwrap=False):
    """Run the optimized pipeline in a background thread."""
    extract_dir = os.path.join(UPLOAD_DIR, job_id, "extracted")
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    work_dir = None

    try:
        # Extract zip
        _update_job(job_id, status="extracting", step="解压 ZIP 文件", progress=5)
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            safe_extract(zf, extract_dir)

        scan_root = find_scan_root(extract_dir)
        if scan_root is None:
            _update_job(job_id, status="failed", error="ZIP 中未找到 model.obj 和 poses.json")
            return

        # Optional: UV unwrap (xatlas re-unwrap + texture re-projection)
        if uv_unwrap:
            _update_job(job_id, status="processing", step="0/4 UV 纹理展开 (xatlas)", progress=8)
            from processing_pipeline.uv_unwrap import uv_unwrap_scan
            uv_stats = uv_unwrap_scan(scan_root)
            _update_job(job_id, progress=12)

        from processing_pipeline.optimized_pipeline import OptimizedPipeline

        optimizer_url = os.environ.get("MODEL_OPTIMIZER_URL", "http://model_optimizer:3000")
        pipeline = OptimizedPipeline(optimizer_url=optimizer_url)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Input validation
        _update_job(job_id, status="processing", step="1/4 输入验证")
        scan_input = pipeline.validate_input(scan_root)
        _update_job(job_id, progress=15)

        # Step 2: Model optimization
        _update_job(job_id, step="2/4 模型优化")
        import tempfile
        import trimesh
        work_dir = tempfile.mkdtemp(prefix="pipeline_")
        glb_path = pipeline.optimize_model(scan_input, work_dir)
        _update_job(job_id, progress=35)

        # Load GLB once and convert to trimesh mesh
        scene = trimesh.load(glb_path)
        if isinstance(scene, trimesh.Scene):
            mesh_tri = scene.to_geometry()
        else:
            mesh_tri = scene

        # Step 3: Feature extraction
        _update_job(job_id, step="3/4 特征提取")
        features = pipeline.build_feature_database(
            mesh_tri, scan_input.images, scan_input.intrinsics
        )
        _update_job(job_id, progress=60)

        # Step 4: Asset bundling
        _update_job(job_id, step="4/4 资产打包")
        pipeline.export_asset_bundle(glb_path, mesh_tri, features, output_dir)
        _update_job(job_id, progress=80)

        # Create downloadable zip
        result_zip = os.path.join(OUTPUT_DIR, f"{job_id}.zip")
        shutil.make_archive(result_zip.replace(".zip", ""), "zip", output_dir)

        _update_job(
            job_id,
            status="completed",
            step="完成",
            progress=100,
            result_zip=result_zip,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        _update_job(job_id, status="failed", error=str(e), progress=0)
    finally:
        if work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)



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

    # 验证文件内容是否为有效 ZIP
    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        os.rmdir(job_dir)
        return jsonify({"error": "上传的文件不是有效的 ZIP 格式"}), 400

    # 读取 UV 展开选项
    uv_unwrap = request.form.get("uv_unwrap", "0") == "1"

    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "step": "等待处理",
        "progress": 0,
        "error": None,
        "result_zip": None,
        "uv_unwrap": uv_unwrap,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }

    t = threading.Thread(target=run_pipeline, args=(job_id, zip_path, uv_unwrap), daemon=True)
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
