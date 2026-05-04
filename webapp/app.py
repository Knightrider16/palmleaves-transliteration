"""
Kerala Archives Research Portal — Flask app.

Run:
    realesgran_venv/Scripts/python.exe -m webapp.app

The app exposes:
    /            login
    /dashboard   home with recent additions
    /search      archive document search
    /projects    transliteration tool (palm-leaf manuscripts)
    /gallery     image grid
    /about       static about page
    /contact     static contact page
"""
from __future__ import annotations
import os
from functools import wraps
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_from_directory, abort,
)
from werkzeug.utils import secure_filename

import re as _re
from markupsafe import Markup, escape

from . import store, transliterate

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

UPLOAD_DIR = ROOT / "static" / "uploads"
GALLERY_DIR = ROOT / "static" / "gallery"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "ARCHIVES_SECRET", "dev-secret-change-me"
)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.template_filter("highlight")
def highlight_filter(text: str, tokens) -> Markup:
    """
    Wrap any case-insensitive occurrence of a token (or its synonyms)
    with <mark>. Tokens may be a list of strings or a single string.
    """
    if not text:
        return Markup("")
    if isinstance(tokens, str):
        tokens = [tokens]
    terms = [t for t in (tokens or []) if t and len(t) >= 3]
    if not terms:
        return Markup(escape(text))
    pattern = _re.compile(
        r"(" + "|".join(_re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")",
        flags=_re.IGNORECASE,
    )
    out = []
    last = 0
    for m in pattern.finditer(text):
        out.append(escape(text[last:m.start()]))
        out.append("<mark>")
        out.append(escape(m.group(0)))
        out.append("</mark>")
        last = m.end()
    out.append(escape(text[last:]))
    return Markup("".join(out))


def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if not session.get("user"):
            return redirect(url_for("login", next=request.path))
        return view(*a, **kw)
    return wrapped


# ---- auth -----------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        u = (request.form.get("username") or "").strip()
        p = request.form.get("password") or ""
        if store.check_password(u, p):
            session["user"] = u
            nxt = request.args.get("next") or url_for("dashboard")
            return redirect(nxt)
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ---- pages ----------------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():
    recent = store.recent_archives(limit=4)
    return render_template(
        "dashboard.html",
        active="dashboard",
        recent=recent,
        stats=store.stats(),
    )


@app.route("/api/search-suggest")
@login_required
def api_search_suggest():
    q = (request.args.get("q") or "").strip()
    return jsonify({"suggestions": store.suggest(q, limit=6)})


@app.route("/archive/<int:archive_id>")
@login_required
def archive_detail(archive_id: int):
    a = store.get_archive(archive_id)
    if a is None:
        abort(404)
    # Find related items via the same semantic engine
    related = []
    try:
        from . import embeddings
        embeddings.index(store.all_archives())
        seed = ". ".join([a["title"], a["description"] or "",
                          " ".join(a["tags"])])
        hits = embeddings.nearest(seed, k=4, exclude_ids={a["id"]})
        for aid, score in hits:
            r = store.get_archive(aid)
            if r:
                r["score"] = round(float(score), 3)
                related.append(r)
    except Exception:
        pass
    cover_idx = (archive_id - 1) % 4 + 1
    return render_template(
        "archive_detail.html",
        active="search",
        a=a,
        related=related,
        cover_idx=cover_idx,
    )


@app.route("/search")
@login_required
def search():
    q = (request.args.get("q") or "").strip()
    if q:
        result = store.search_with_related(q)
        recent = []
    else:
        result = {"direct": [], "related": [], "tokens": [], "expanded": []}
        recent = store.recent_archives(limit=20)
    return render_template(
        "search.html",
        active="search",
        q=q,
        result=result,
        recent=recent,
    )


@app.route("/projects")
@login_required
def projects():
    return render_template("projects.html", active="projects")


@app.route("/projects/transliteration", methods=["GET"])
@login_required
def transliteration():
    samples = []
    samples_dir = ROOT / "static" / "samples"
    if samples_dir.is_dir():
        samples = sorted(
            f.name for f in samples_dir.iterdir()
            if f.suffix.lower() in ALLOWED_EXTS
        )[:6]
    return render_template(
        "transliteration.html",
        active="projects",
        models=transliterate.available_models(),
        samples=samples,
    )


@app.route("/api/transliterate", methods=["POST"])
@login_required
def api_transliterate():
    model_name = (request.form.get("model") or "").strip()
    if model_name not in transliterate.available_models():
        return jsonify({"error": "Select a model."}), 400

    file = request.files.get("image")
    sample = (request.form.get("sample") or "").strip()

    src_path = None
    if file and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTS:
            return jsonify({"error": "Unsupported image format."}), 400
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe = secure_filename(file.filename) or f"upload{ext}"
        dest = UPLOAD_DIR / safe
        file.save(dest)
        src_path = dest
        image_url = url_for("static", filename=f"uploads/{safe}")
    elif sample:
        safe = secure_filename(sample)
        cand = ROOT / "static" / "samples" / safe
        if not cand.is_file():
            return jsonify({"error": "Sample not found."}), 400
        src_path = cand
        image_url = url_for("static", filename=f"samples/{safe}")
    else:
        return jsonify({"error": "Upload an image or pick a sample."}), 400

    try:
        lines = transliterate.run(model_name, str(src_path))
    except Exception as e:
        app.logger.exception("transliteration failed")
        return jsonify({"error": f"Inference failed: {e}"}), 500

    text = "\n".join(" ".join(line) for line in lines if line)
    return jsonify({
        "image_url": image_url,
        "model": model_name,
        "lines": lines,
        "text": text,
    })


@app.route("/gallery")
@login_required
def gallery():
    items = []
    if GALLERY_DIR.is_dir():
        items = sorted(
            f.name for f in GALLERY_DIR.iterdir()
            if f.suffix.lower() in ALLOWED_EXTS
        )
    return render_template("gallery.html", active="gallery", items=items)


@app.route("/about")
@login_required
def about():
    return render_template("about.html", active="about")


@app.route("/contact")
@login_required
def contact():
    sent = request.args.get("sent") == "1"
    return render_template("contact.html", active="contact", sent=sent)


@app.route("/contact", methods=["POST"])
@login_required
def contact_submit():
    name = (request.form.get("name") or "").strip()
    email = (request.form.get("email") or "").strip()
    message = (request.form.get("message") or "").strip()
    if not (name and email and message):
        flash("All fields are required.", "error")
        return redirect(url_for("contact"))
    store.record_contact(name, email, message)
    return redirect(url_for("contact", sent=1))


# ---- error handlers -------------------------------------------------------

@app.errorhandler(413)
def too_large(_e):
    return ("Upload too large (max 16 MB).", 413)


def main():
    store.init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "static" / "samples").mkdir(parents=True, exist_ok=True)

    # Warm the embedding model + index so the first search request is fast.
    try:
        from . import embeddings
        print("Warming semantic-search embeddings (~3 sec on first run)...")
        embeddings.warm()
        embeddings.index(store.all_archives())
        print("Embeddings ready.")
    except Exception as e:
        app.logger.warning("Could not warm embeddings: %s — "
                           "search will fall back to tag overlap.", e)

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
