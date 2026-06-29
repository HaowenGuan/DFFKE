SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
DFFKE_DIR="${DFFKE_DIR:-$(CDPATH= cd "$SCRIPT_DIR/.." && pwd)}"
cd "$DFFKE_DIR" || exit 1
mkdir -p logs

nohup python3 "$DFFKE_DIR/main.py" --device_id 0 --config_file DFFKE.yaml 2>&1 | ts '[%Y-%m-%d %H:%M:%S]' > logs/example.log &
