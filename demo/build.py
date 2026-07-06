"""Inyecta demo/data/bundle.json dentro de la plantilla app.html -> index.html.

El resultado es un unico archivo autocontenido (sin dependencias externas) que
abre en cualquier navegador y sirve como fuente del Artifact.

    python demo/build.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
tpl = (HERE / "app.html").read_text(encoding="utf-8")
data = (HERE / "data" / "bundle.json").read_text(encoding="utf-8")

# Seguridad: evita que un eventual "</script>" dentro de un nombre cierre el tag.
# JSON.parse interpreta "<\/" como "</", asi que el dato no se altera.
data_safe = data.replace("</", "<\\/")

out = tpl.replace("__DATA__", data_safe)
dst = HERE / "index.html"
dst.write_text(out, encoding="utf-8")
print(f"OK -> {dst}  ({len(out)/1e6:.2f} MB)")
