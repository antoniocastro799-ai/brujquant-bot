"""Servidor web ligero para mantener el bot activo 24/7 en Render."""
import os
import sys
import time
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone

# Cargar variables locales si existen
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

import brujquant_bot

# Estado global para monitoreo web
bot_state = {
    "status": "INITIALIZING",
    "mode": "UNKNOWN",
    "symbol": os.getenv("TRADING_SYMBOL", "XRP/USDT:USDT"),
    "timeframe": os.getenv("TRADING_TIMEFRAME", "15m"),
    "last_cycle_utc": "Never",
    "last_result": {},
    "start_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    "cycles_count": 0,
    "last_error": None
}


def run_bot_worker():
    """Hilo de fondo que ejecuta la logica de trading de forma continua."""
    is_live = os.getenv("DRY_RUN", "true").lower() == "false"
    symbol = os.getenv("TRADING_SYMBOL", "XRP/USDT:USDT")
    timeframe = os.getenv("TRADING_TIMEFRAME", "15m")
    check_interval = int(os.getenv("CHECK_INTERVAL_SEC", "60"))

    mode_str = "LIVE_BINANCE" if is_live else "DRY_RUN"
    bot_state["mode"] = mode_str
    bot_state["status"] = "RUNNING"
    brujquant_bot.logger.info("Worker iniciado. Modo: %s Par: %s", mode_str, symbol)

    try:
        agent = brujquant_bot.BrujQuantDayTraderAgent(dry_run=not is_live)
    except Exception as e:
        bot_state["status"] = "AGENT_INIT_FAILED"
        bot_state["last_error"] = str(e)
        brujquant_bot.logger.error("Fallo al inicializar agente: %s", e)
        return

    while True:
        try:
            res = agent.run_live_cycle(symbol=symbol, timeframe=timeframe)
            bot_state["cycles_count"] += 1
            bot_state["last_cycle_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            bot_state["last_result"] = res
            bot_state["status"] = "HEALTHY_LISTENING"
            brujquant_bot.logger.info("Ciclo #%d: %s", bot_state["cycles_count"], json.dumps(res))
        except Exception as e:
            bot_state["last_error"] = str(e)
            brujquant_bot.logger.error("Error en ciclo: %s", e)

        time.sleep(check_interval)


class StatusHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silenciar logs ruidosos de peticiones http para no saturar consola
        return

    def do_GET(self):
        if self.path in ("/health", "/ping"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
            return

        if self.path == "/json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(bot_state, indent=2).encode("utf-8"))
            return

        # Pagina principal con status visual
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        mode_badge = f'<span style="background: {"#10b981" if bot_state["mode"] == "LIVE_BINANCE" else "#f59e0b"}; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold;">{bot_state["mode"]}</span>'

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BrujQuant Bot Monitor</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; display: flex; justify-content: center; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 25px; max-width: 600px; width: 100%; border: 1px solid #334155; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        h1 {{ margin-top: 0; color: #38bdf8; font-size: 24px; }}
        .row {{ display: flex; justify-content: space-between; margin: 12px 0; border-bottom: 1px solid #334155; padding-bottom: 8px; }}
        .label {{ color: #94a3b8; font-weight: 500; }}
        .value {{ font-weight: 600; text-align: right; }}
        pre {{ background: #0f172a; padding: 12px; border-radius: 8px; overflow-x: auto; font-size: 13px; color: #a5f3fc; border: 1px solid #334155; }}
        .footer {{ font-size: 12px; color: #64748b; margin-top: 20px; text-align: center; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>⚡ BrujQuant Trading Bot</h1>
        <div class="row"><span class="label">Estado:</span><span class="value">{bot_state['status']}</span></div>
        <div class="row"><span class="label">Modo:</span><span class="value">{mode_badge}</span></div>
        <div class="row"><span class="label">S&iacute;mbolo:</span><span class="value">{bot_state['symbol']} ({bot_state['timeframe']})</span></div>
        <div class="row"><span class="label">Ciclos Completados:</span><span class="value">{bot_state['cycles_count']}</span></div>
        <div class="row"><span class="label">&Uacute;ltima Revisi&oacute;n:</span><span class="value">{bot_state['last_cycle_utc']}</span></div>
        <div class="row"><span class="label">Iniciado el:</span><span class="value">{bot_state['start_time_utc']}</span></div>
        
        <p style="margin-top: 20px; color: #94a3b8; font-size: 14px;">Resultado del &uacute;ltimo ciclo:</p>
        <pre>{json.dumps(bot_state['last_result'], indent=2)}</pre>
        
        <div class="footer">Servidor Activo 24/7 en Render &bull; Ping Keep-Alive habilitado</div>
    </div>
</body>
</html>"""
        self.wfile.write(html.encode("utf-8"))


def main():
    port = int(os.getenv("PORT", "10000"))
    
    # Iniciar hilo del bot de trading en segundo plano
    t = threading.Thread(target=run_bot_worker, daemon=True)
    t.start()

    # Iniciar servidor HTTP
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, StatusHTTPHandler)
    brujquant_bot.logger.info("Servidor HTTP listo en puerto %d", port)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
