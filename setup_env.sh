#!/usr/bin/env bash
# =============================================================
# Setup del entorno — Proyecto 2 Data Mining UTEC (Mac M1 Pro)
# Uso:  cd proyecto_2/codigo && bash setup_env.sh
# Crea el venv ".venv-yelpdm" (Yelp Data Mining) y registra el
# kernel de Jupyter "Python (yelp-dm)".
# =============================================================
set -e
cd "$(dirname "$0")"

echo "== 1/4 Buscando Python compatible (3.10–3.12; PySpark 3.5 no soporta 3.13+) =="
PY=""
for cand in python3.12 python3.11 python3.10; do
  if command -v $cand >/dev/null 2>&1; then PY=$cand; break; fi
done
if [ -z "$PY" ]; then
  V=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0")
  case "$V" in 3.10|3.11|3.12) PY=python3 ;; esac
fi
if [ -z "$PY" ]; then
  echo "ERROR: no encontré Python 3.10–3.12."
  echo "  Solución (Homebrew):  brew install python@3.12"
  exit 1
fi
echo "   OK: usando $PY ($($PY --version))"

echo "== 2/4 Verificando Java (PySpark necesita JDK; recomendado 17) =="
if /usr/libexec/java_home -v 17 >/dev/null 2>&1; then
  export JAVA_HOME=$(/usr/libexec/java_home -v 17)
  echo "   OK: Java 17 en $JAVA_HOME"
elif /usr/libexec/java_home >/dev/null 2>&1; then
  export JAVA_HOME=$(/usr/libexec/java_home)
  echo "   AVISO: usaré el Java por defecto ($JAVA_HOME). Si Spark falla:"
  echo "     brew install --cask temurin@17   y vuelve a correr este script."
else
  echo "ERROR: no hay Java instalado. Instala JDK 17 (nativo arm64):"
  echo "   brew install --cask temurin@17"
  exit 1
fi

echo "== 3/4 Creando venv .venv-yelpdm e instalando dependencias =="
$PY -m venv .venv-yelpdm
# Dejar JAVA_HOME fijado cada vez que se active el venv:
JAVA_EXPORT="export JAVA_HOME=\"$JAVA_HOME\""
grep -Fqx "$JAVA_EXPORT" .venv-yelpdm/bin/activate || echo "$JAVA_EXPORT" >> .venv-yelpdm/bin/activate
source .venv-yelpdm/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "== 4/4 Registrando kernel de Jupyter =="
python -m ipykernel install --user --name yelpdm --display-name "Python (yelp-dm)"

echo ""
echo "=============================================="
echo "LISTO. Para trabajar:"
echo "  source .venv-yelpdm/bin/activate"
echo "  jupyter lab"
echo "y en los notebooks elige el kernel: Python (yelp-dm)"
echo "=============================================="
python - <<'EOF'
import sys, pyspark, numpy, pandas, duckdb
print(f"Python  {sys.version.split()[0]}")
print(f"PySpark {pyspark.__version__}")
print(f"numpy   {numpy.__version__} | pandas {pandas.__version__} | duckdb {duckdb.__version__}")
EOF
