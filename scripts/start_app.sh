#!/usr/bin/env bash

# AI Systems Lab backend + frontend development launcher.
#
# Usage:
#   ./scripts/start_app.sh start
#   ./scripts/start_app.sh start --detached
#   ./scripts/start_app.sh stop
#   ./scripts/start_app.sh restart
#   ./scripts/start_app.sh status

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${PROJECT_ROOT}/.run"
PID_FILE="${RUNTIME_DIR}/ai-systems-lab.pids"
LOG_DIR="${PROJECT_ROOT}/logs"

DEFAULT_BACKEND_PORT="${BACKEND_PORT:-8000}"
DEFAULT_FRONTEND_PORT="${FRONTEND_PORT:-3000}"
HOST="${APP_HOST:-127.0.0.1}"
DETACHED=false
STRICT_PORTS=false

BACKEND_PID=""
FRONTEND_PID=""
REMOVE_PID_FILE=false

print_usage() {
    cat <<'EOF'
AI Systems Lab local application launcher

Usage:
  ./scripts/start_app.sh start [options]
  ./scripts/start_app.sh stop
  ./scripts/start_app.sh restart [options]
  ./scripts/start_app.sh status

Options for start/restart:
  --backend-port PORT    Backend base port (default: 8000)
  --frontend-port PORT   Frontend base port (default: 3000)
  --host HOST            Bind host (default: 127.0.0.1)
  --detached             Run in background and write logs to logs/
  --strict-ports         Fail instead of selecting the next free port
  -h, --help             Show this help

When a requested port is busy, the launcher selects the next available port
unless --strict-ports is provided. Existing processes are never terminated.
EOF
}

die() {
    printf 'Hata: %s\n' "$1" >&2
    exit 1
}

is_valid_port() {
    [[ "$1" =~ ^[0-9]+$ ]] && ((10#$1 >= 1 && 10#$1 <= 65535))
}

port_in_use() {
    local port="$1"

    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"${port}" -sTCP:LISTEN -t >/dev/null 2>&1
        return $?
    fi

    "${PYTHON_BIN}" -c \
        'import socket, sys; s=socket.socket(); s.settimeout(0.15); result=s.connect_ex(("127.0.0.1", int(sys.argv[1]))); s.close(); raise SystemExit(result == 0)' \
        "${port}"
}

next_free_port() {
    local port="$1"

    ((port <= 65535)) || die "Boş port bulunamadı."
    while port_in_use "${port}"; do
        ((port += 1))
        ((port <= 65535)) || die "Boş port bulunamadı."
    done

    printf '%s' "${port}"
}

pid_is_running() {
    [[ "$1" =~ ^[0-9]+$ ]] && kill -0 "$1" >/dev/null 2>&1
}

read_pid_value() {
    local key="$1"
    [[ -f "${PID_FILE}" ]] || return 0
    awk -F= -v wanted="${key}" '$1 == wanted { print substr($0, index($0, "=") + 1) }' "${PID_FILE}"
}

write_pid_file() {
    local backend_port="$1"
    local frontend_port="$2"

    mkdir -p "${RUNTIME_DIR}"
    umask 077
    {
        printf 'BACKEND_PID=%s\n' "${BACKEND_PID}"
        printf 'FRONTEND_PID=%s\n' "${FRONTEND_PID}"
        printf 'BACKEND_PORT=%s\n' "${backend_port}"
        printf 'FRONTEND_PORT=%s\n' "${frontend_port}"
    } > "${PID_FILE}"
}

terminate_pid() {
    local label="$1"
    local pid="$2"

    if ! pid_is_running "${pid}"; then
        return 0
    fi

    printf '%s durduruluyor (PID %s)...\n' "${label}" "${pid}"
    kill "${pid}" >/dev/null 2>&1 || true

    for _ in {1..20}; do
        pid_is_running "${pid}" || return 0
        sleep 0.25
    done

    printf '%s zorla durduruluyor...\n' "${label}"
    kill -KILL "${pid}" >/dev/null 2>&1 || true
}

stop_from_pid_file() {
    if [[ ! -f "${PID_FILE}" ]]; then
        printf 'Çalışan launcher kaydı bulunamadı.\n'
        return 0
    fi

    local backend_pid
    local frontend_pid
    backend_pid="$(read_pid_value BACKEND_PID)"
    frontend_pid="$(read_pid_value FRONTEND_PID)"

    terminate_pid "Backend" "${backend_pid}"
    terminate_pid "Frontend" "${frontend_pid}"
    rm -f "${PID_FILE}"
    printf 'Uygulama durduruldu.\n'
}

cleanup_foreground() {
    local exit_code=$?
    trap - EXIT INT TERM

    if [[ -n "${BACKEND_PID}" ]]; then
        terminate_pid "Backend" "${BACKEND_PID}"
    fi
    if [[ -n "${FRONTEND_PID}" ]]; then
        terminate_pid "Frontend" "${FRONTEND_PID}"
    fi
    if [[ "${REMOVE_PID_FILE}" == true ]]; then
        rm -f "${PID_FILE}"
    fi

    exit "${exit_code}"
}

wait_for_url() {
    local url="$1"
    local label="$2"

    for _ in {1..60}; do
        if curl -fsS --max-time 1 "${url}" >/dev/null 2>&1; then
            printf '✓ %s hazır: %s\n' "${label}" "${url}"
            return 0
        fi
        sleep 0.5
    done

    printf '⚠ %s 30 saniye içinde doğrulanamadı: %s\n' "${label}" "${url}" >&2
    return 1
}

start_app() {
    local backend_port="$1"
    local frontend_port="$2"

    if [[ -f "${PID_FILE}" ]]; then
        local old_backend_pid
        local old_frontend_pid
        old_backend_pid="$(read_pid_value BACKEND_PID)"
        old_frontend_pid="$(read_pid_value FRONTEND_PID)"
        if pid_is_running "${old_backend_pid}" || pid_is_running "${old_frontend_pid}"; then
            die "Uygulama zaten çalışıyor. Önce './scripts/start_app.sh stop' çalıştırın."
        fi
        rm -f "${PID_FILE}"
    fi

    local allowed_origins
    allowed_origins="[\"http://${HOST}:${frontend_port}\",\"http://localhost:${frontend_port}\",\"http://127.0.0.1:${frontend_port}\",\"http://${HOST}:${backend_port}\",\"http://localhost:${backend_port}\"]"

    local api_url="http://${HOST}:${backend_port}"
    local frontend_url="http://${HOST}:${frontend_port}"
    local backend_log="${LOG_DIR}/backend.log"
    local frontend_log="${LOG_DIR}/frontend.log"

    mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}"

    printf '\nAI Systems Lab başlatılıyor...\n'
    printf '  Backend : %s\n' "${api_url}"
    printf '  Frontend: %s\n\n' "${frontend_url}"

    if [[ "${DETACHED}" == true ]]; then
        (
            cd "${PROJECT_ROOT}"
            exec nohup env \
                BACKEND_HOST="${HOST}" \
                BACKEND_PORT="${backend_port}" \
                BACKEND_RELOAD=true \
                BACKEND_WORKERS=1 \
                ALLOWED_ORIGINS="${allowed_origins}" \
                "${PYTHON_BIN}" -m uvicorn backend.main:app --host "${HOST}" --port "${backend_port}" --reload
        ) > "${backend_log}" 2>&1 < /dev/null &
        BACKEND_PID=$!

        (
            cd "${PROJECT_ROOT}/frontend"
            exec nohup env \
                API_URL="${api_url}" \
                NEXT_PUBLIC_API_URL="${api_url}" \
                npm run dev -- --hostname "${HOST}" --port "${frontend_port}"
        ) > "${frontend_log}" 2>&1 < /dev/null &
        FRONTEND_PID=$!
    else
        (
            cd "${PROJECT_ROOT}"
            exec env \
                BACKEND_HOST="${HOST}" \
                BACKEND_PORT="${backend_port}" \
                BACKEND_RELOAD=true \
                BACKEND_WORKERS=1 \
                ALLOWED_ORIGINS="${allowed_origins}" \
                "${PYTHON_BIN}" -m uvicorn backend.main:app --host "${HOST}" --port "${backend_port}" --reload
        ) &
        BACKEND_PID=$!

        (
            cd "${PROJECT_ROOT}/frontend"
            exec env \
                API_URL="${api_url}" \
                NEXT_PUBLIC_API_URL="${api_url}" \
                npm run dev -- --hostname "${HOST}" --port "${frontend_port}"
        ) &
        FRONTEND_PID=$!
    fi

    write_pid_file "${backend_port}" "${frontend_port}"
    REMOVE_PID_FILE=true
    if [[ "${DETACHED}" != true ]]; then
        trap cleanup_foreground EXIT INT TERM
    fi

    local ready=true
    wait_for_url "${api_url}/health" "Backend" || ready=false
    wait_for_url "${frontend_url}/" "Frontend" || ready=false

    if [[ "${ready}" != true ]]; then
        printf 'Uygulama başlatılamadı. Logları kontrol edin: %s ve %s\n' "${backend_log}" "${frontend_log}" >&2
        if [[ "${DETACHED}" == true ]]; then
            terminate_pid "Backend" "${BACKEND_PID}"
            terminate_pid "Frontend" "${FRONTEND_PID}"
            rm -f "${PID_FILE}"
        else
            cleanup_foreground
        fi
        return 1
    fi

    printf '\nUygulama hazır.\n'
    printf '  Frontend: %s\n' "${frontend_url}"
    printf '  Backend : %s\n' "${api_url}"
    printf '  API docs: %s/docs\n' "${api_url}"

    if [[ "${DETACHED}" == true ]]; then
        printf '  Loglar  : %s, %s\n' "${backend_log}" "${frontend_log}"
        printf 'Durdurmak için: ./scripts/start_app.sh stop\n'
        return 0
    fi

    printf '\nÇıkmak için Ctrl-C kullanın.\n'
    while pid_is_running "${BACKEND_PID}" && pid_is_running "${FRONTEND_PID}"; do
        sleep 1
    done

    printf '\nBir servis sonlandı; diğer servisler durduruluyor.\n' >&2
    return 1
}

show_status() {
    if [[ ! -f "${PID_FILE}" ]]; then
        printf 'Uygulama çalışmıyor.\n'
        return 1
    fi

    local backend_pid frontend_pid backend_port frontend_port
    backend_pid="$(read_pid_value BACKEND_PID)"
    frontend_pid="$(read_pid_value FRONTEND_PID)"
    backend_port="$(read_pid_value BACKEND_PORT)"
    frontend_port="$(read_pid_value FRONTEND_PORT)"

    printf 'Backend : %s (PID %s) %s\n' "${backend_port}" "${backend_pid}" \
        "$(pid_is_running "${backend_pid}" && printf 'çalışıyor' || printf 'çalışmıyor')"
    printf 'Frontend: %s (PID %s) %s\n' "${frontend_port}" "${frontend_pid}" \
        "$(pid_is_running "${frontend_pid}" && printf 'çalışıyor' || printf 'çalışmıyor')"
}

require_dependencies() {
    if [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
        PYTHON_BIN="${PROJECT_ROOT}/venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        die "Python 3 bulunamadı. Önce scripts/setup_dev.sh çalıştırın."
    fi

    command -v npm >/dev/null 2>&1 || die "npm bulunamadı."
    [[ -d "${PROJECT_ROOT}/frontend/node_modules" ]] || die "Frontend bağımlılıkları yok. Önce cd frontend && npm install çalıştırın."
    "${PYTHON_BIN}" -c 'import uvicorn' >/dev/null 2>&1 || die "uvicorn bulunamadı. Python bağımlılıklarını kurun."
    command -v curl >/dev/null 2>&1 || die "curl bulunamadı."
}

resolve_ports() {
    local backend_port="$1"
    local frontend_port="$2"

    is_valid_port "${backend_port}" || die "Geçersiz backend portu: ${backend_port}"
    is_valid_port "${frontend_port}" || die "Geçersiz frontend portu: ${frontend_port}"

    if port_in_use "${backend_port}"; then
        [[ "${STRICT_PORTS}" == true ]] && die "Backend portu kullanımda: ${backend_port}"
        backend_port="$(next_free_port "$((10#${backend_port} + 1))")"
        printf 'Bilgi: Backend için boş port seçildi: %s\n' "${backend_port}"
    fi

    if [[ "${frontend_port}" == "${backend_port}" ]]; then
        [[ "${STRICT_PORTS}" == true ]] && die "Backend ve frontend aynı portu kullanamaz."
        frontend_port="$((10#${frontend_port} + 1))"
    fi

    if port_in_use "${frontend_port}"; then
        [[ "${STRICT_PORTS}" == true ]] && die "Frontend portu kullanımda: ${frontend_port}"
        frontend_port="$(next_free_port "$((10#${frontend_port} + 1))")"
        printf 'Bilgi: Frontend için boş port seçildi: %s\n' "${frontend_port}"
    fi

    START_BACKEND_PORT="${backend_port}"
    START_FRONTEND_PORT="${frontend_port}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
fi

COMMAND="${1:-start}"
if [[ $# -gt 0 ]]; then
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend-port)
            [[ $# -ge 2 ]] || die "--backend-port bir değer bekliyor."
            DEFAULT_BACKEND_PORT="$2"
            shift 2
            ;;
        --frontend-port)
            [[ $# -ge 2 ]] || die "--frontend-port bir değer bekliyor."
            DEFAULT_FRONTEND_PORT="$2"
            shift 2
            ;;
        --host)
            [[ $# -ge 2 ]] || die "--host bir değer bekliyor."
            HOST="$2"
            shift 2
            ;;
        --detached)
            DETACHED=true
            shift
            ;;
        --strict-ports)
            STRICT_PORTS=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            die "Bilinmeyen seçenek: $1"
            ;;
    esac
done

case "${COMMAND}" in
    start|restart)
        require_dependencies
        if [[ "${COMMAND}" == restart ]]; then
            stop_from_pid_file
        fi
        resolve_ports "${DEFAULT_BACKEND_PORT}" "${DEFAULT_FRONTEND_PORT}"
        start_app "${START_BACKEND_PORT}" "${START_FRONTEND_PORT}"
        ;;
    stop)
        stop_from_pid_file
        ;;
    status)
        show_status
        ;;
    help)
        print_usage
        ;;
    *)
        die "Bilinmeyen komut: ${COMMAND}. --help kullanın."
        ;;
esac
