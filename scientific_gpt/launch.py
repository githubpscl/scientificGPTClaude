"""
launch.py — starts Scientific GPT and exposes it via an ngrok tunnel.

Usage:
    python launch.py [--token YOUR_NGROK_TOKEN]

Free ngrok account → https://dashboard.ngrok.com/get-started/your-authtoken
The token only needs to be set once; it is stored in ~/.ngrok2/ngrok.yml.
"""

import argparse
import subprocess
import sys
import time
import os

from pyngrok import ngrok, conf


def main():
    parser = argparse.ArgumentParser(description="Launch Scientific GPT with a public URL")
    parser.add_argument("--token", default=os.getenv("NGROK_AUTHTOKEN"), help="ngrok auth token")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port (default 8501)")
    args = parser.parse_args()

    if args.token:
        ngrok.set_auth_token(args.token)

    # Start Streamlit in a subprocess
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--server.address", "localhost",
    ]
    print(f"🚀 Starting Streamlit on port {args.port}…")
    proc = subprocess.Popen(streamlit_cmd)

    # Give Streamlit a moment to bind
    time.sleep(3)

    # Open ngrok tunnel
    print("🌐 Opening ngrok tunnel…")
    try:
        tunnel = ngrok.connect(args.port, "http")
        public_url = tunnel.public_url
        # Prefer HTTPS
        if public_url.startswith("http://"):
            public_url = "https://" + public_url[len("http://"):]
    except Exception as e:
        print(f"\n⚠️  ngrok tunnel failed: {e}")
        print("    → Run without tunnel: venv/Scripts/streamlit run app.py")
        proc.wait()
        return

    print("\n" + "═" * 60)
    print(f"  🔬 Scientific GPT is LIVE")
    print(f"  🔗 Public URL : {public_url}")
    print(f"  💻 Local URL  : http://localhost:{args.port}")
    print("═" * 60)
    print("  Press Ctrl+C to stop.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down…")
        ngrok.kill()
        proc.terminate()


if __name__ == "__main__":
    main()
