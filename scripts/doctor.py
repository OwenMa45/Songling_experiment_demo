"""Can run with bare Python before installing this project."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from piper_titration.diagnostics import main
raise SystemExit(main())
