from pathlib import Path
import sys

applications_path = Path(__file__).parents[3].resolve() / "applications"
sys.path.insert(0, str((applications_path / "performUQ" / "UCSD_UQ").resolve()))
import mainscript_tmcmc