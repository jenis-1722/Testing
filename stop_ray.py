##hello This is testing
"""
Stop Ray processes for a specific developer
Safely terminates local Ray processes without affecting other users

Usage:
    stop_ray.py [--userid USERID] [--venv VENV_PATH] [--force]
"""

import argparse
import os
import sys
import subprocess
import time
import signal
import getpass
import socket
from typing import List, Set, Tuple
from pathlib import Path

class RayStopper:
    """Safely stop Ray processes for a specific developer"""
    
    def __init__(self, userid: str = None, venv: str = None, force: bool = False):
        self.userid = userid or self._get_userid()
        self.venv = venv
        self.force = force
        self.found_processes = False
        
        # Set developer userid for project_settings
        os.environ['VAULTIQ_DEVELOPER_USERID'] = self.userid
        
        # Track PIDs we've already processed
        self.processed_pids = set()
        
        # psutil will be imported after venv activation
        self.psutil = None
        
    def _get_userid(self) -> str:
        """Get the userid, requiring it for ec2-user"""
        current_user = getpass.getuser()
        
        if current_user == 'ec2-user':
            print("ERROR: You are using a shared account (ec2-user).")
            print("Please specify a userid: stop_ray.py --userid <userid>")
            sys.exit(1)
            
        return current_user
    
    def _activate_venv(self):
        """Activate virtual environment if needed"""
        if os.environ.get("VIRTUAL_ENV"):
            print(f"[*] Already in virtual environment: {os.environ['VIRTUAL_ENV']}")
            return

        if self.venv:
            venv_path = Path(self.venv)
        else:
            # Try to get from project_settings
            try:
                # Add current directory to path to find project_settings
                if os.getcwd() not in sys.path:
                    sys.path.insert(0, os.getcwd())
                
                from project_settings import settings
                venv_path = Path(settings.VAULTIQ_VENV)
                print(f"[*] Using VaultIQ default venv: {venv_path}")
            except (ImportError, AttributeError) as e:
                print(f"[ERROR] Not in a virtual environment and cannot find default: {e}")
                print("[ERROR] Please specify --venv /path/to/venv")
                sys.exit(1)

        activate_script = venv_path / 'bin' / 'activate_this.py'
        if activate_script.exists():
            exec(open(activate_script).read(), {'__file__': str(activate_script)})
            print(f"[✓] Activated virtual environment: {venv_path}")
        else:
            print(f"[ERROR] Cannot activate virtual environment at {venv_path}")
            print(f"[ERROR] activate_this.py not found at {activate_script}")
            sys.exit(1)
    
    def _import_dependencies(self):
        """Import dependencies after venv activation"""
        try:
            import psutil
            self.psutil = psutil
            print(f"[✓] psutil version {psutil.__version__} found")
        except ImportError:
            print("[ERROR] psutil is not installed in this environment")
            print("[ERROR] Please install it with: pip install psutil")
            sys.exit(1)
    
    def _load_settings(self):
        """Load settings from project_settings module"""
        try:
            # Add current directory to path if needed
            if os.getcwd() not in sys.path:
                sys.path.insert(0, os.getcwd())
                
            from project_settings import settings
            self.ray_gcs_port = settings.RAY_GCS_PORT
            self.ray_dashboard_port = settings.RAY_DASHBOARD_PORT
            self.ray_client_server_port = settings.RAY_CLIENT_SERVER_PORT
            self.ray_tmpdir = settings.RAY_TMPDIR
            self.ray_namespace = settings.RAY_NAMESPACE
            fastapi_url = str(getattr(settings, "FASTAPI_URL_BASE", "http://localhost:8000") or "")
            try:
                self.app_port = int(fastapi_url.rstrip("/").rsplit(":", 1)[-1])
            except (ValueError, IndexError):
                self.app_port = 8000
        except ImportError as e:
            print(f"[ERROR] Cannot import project_settings: {e}")
            print(f"[ERROR] Current directory: {os.getcwd()}")
            print(f"[ERROR] Python path: {sys.path}")
            sys.exit(1)
        
    def _print_header(self):
        """Print header information"""
        print("\n" + "=" * 50)
        print(f"STOPPING RAY PROCESSES FOR: {self.userid}")
        print("=" * 50)
        print("Configuration:")
        print(f"  RAY_GCS_PORT: {self.ray_gcs_port}")
        print(f"  RAY_DASHBOARD_PORT: {self.ray_dashboard_port}")
        print(f"  RAY_TMPDIR: {self.ray_tmpdir}")
        print()
        
    def _get_processes_by_port(self, port: int) -> List[int]:
        """Find processes listening on a specific TCP port."""
        pids = []
        try:
            # Only listeners own the service port. Client/observer processes,
            # such as `ray status`, can briefly hold connections to the port.
            result = subprocess.run(
                ['lsof', '-nP', '-t', f'-iTCP:{port}', '-sTCP:LISTEN'],
                capture_output=True,
                text=True
            )
            if result.stdout:
                pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
        except (subprocess.CalledProcessError, ValueError):
            pass
        return pids

    def _is_ray_status_observer(self, cmdline_parts: List[str]) -> bool:
        """Return True for Ray status/monitor commands that do not own Ray."""
        tokens = []
        for part in cmdline_parts or []:
            tokens.extend(str(part).strip().split())

        strip_chars = "'\"()[]{};"
        normalized = [token.strip(strip_chars) for token in tokens if token.strip(strip_chars)]
        lowered = [token.lower() for token in normalized]

        for idx, token in enumerate(lowered[:-1]):
            if Path(token).name == 'ray' and lowered[idx + 1] == 'status':
                return True

        observer_scripts = {'ray_status.py', 'ray_gpu_status.sh'}
        return any(Path(token).name in observer_scripts for token in lowered)

    def _port_has_listener(self, port: int) -> bool:
        """Best-effort listener probe even when PID ownership is hidden.

        In some environments `lsof`/`psutil` cannot attribute host-side
        listeners to a visible PID from the current namespace, but the port is
        still occupied. Probe loopback addresses directly so verification
        doesn't falsely report the port as free.
        """
        candidates = [("127.0.0.1", port), ("::1", port)]
        for host, port_number in candidates:
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            try:
                with socket.socket(family, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.25)
                    if sock.connect_ex((host, port_number)) == 0:
                        return True
            except OSError:
                continue
        return False
    
    def _get_process_info(self, pid: int) -> dict:
        """Get information about a process"""
        try:
            proc = self.psutil.Process(pid)
            return {
                'pid': pid,
                'name': proc.name(),
                'user': proc.username(),
                'cmdline': ' '.join(proc.cmdline()[:5])  # First 5 args for brevity
            }
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
            return None
    
    def _is_my_process(self, pid: int) -> bool:
        """Check if a process belongs to the current user"""
        try:
            proc = self.psutil.Process(pid)
            # Check if process belongs to current user
            return proc.username() == getpass.getuser()
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
            return False
    
    def _kill_processes(self, pids: List[int], description: str) -> bool:
        """Kill a list of processes"""
        if not pids:
            return False
            
        # Filter to only processes we own
        my_pids = [pid for pid in pids if self._is_my_process(pid)]
        
        if not my_pids:
            return False
            
        print(f"\n{description}:")
        
        # Show process details
        for pid in my_pids:
            if pid in self.processed_pids:
                continue
            info = self._get_process_info(pid)
            if info:
                print(f"    PID {info['pid']}: {info['name']} ({info['cmdline'][:60]}...)")
        
        new_pids = [pid for pid in my_pids if pid not in self.processed_pids]
        if not new_pids:
            print("    (Already processed)")
            return False
            
        print(f"  Terminating {len(new_pids)} process(es)...")
        
        # First try SIGTERM
        for pid in new_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                self.processed_pids.add(pid)
            except (ProcessLookupError, PermissionError):
                pass
        
        # Wait a bit
        time.sleep(2)
        
        # Force kill if needed
        still_alive = []
        for pid in new_pids:
            try:
                proc = self.psutil.Process(pid)
                if proc.is_running():
                    still_alive.append(pid)
            except self.psutil.NoSuchProcess:
                pass
        
        if still_alive:
            print(f"  Force killing {len(still_alive)} stubborn process(es)...")
            for pid in still_alive:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        
        return True
    
    def stop_by_ports(self):
        """Stop Ray processes by finding what's using Ray ports"""
        print("Checking Ray ports...")
        
        ports_to_check = [
            ('GCS', self.ray_gcs_port),
            ('Dashboard', self.ray_dashboard_port),
            ('Client Server', self.ray_client_server_port)
        ]
        
        for name, port in ports_to_check:
            if port:
                pids = self._get_processes_by_port(int(port))
                if pids:
                    if self._kill_processes(pids, f"Found processes on {name} port {port}"):
                        self.found_processes = True
    
    def stop_by_command_pattern(self):
        """Stop Ray processes by matching command line patterns"""
        print("\nSearching for Ray processes by command patterns...")
        
        ray_patterns = [
            'ray::', 'raylet', 'gcs_server', 'plasma_store',
            'dashboard', 'ray_client_server', 'runtime_env_agent'
        ]
        
        matching_pids = []
        
        for proc in self.psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
            try:
                # Skip if not our process
                if proc.info['username'] != getpass.getuser():
                    continue
                    
                cmdline_parts = proc.info['cmdline'] or []
                if self._is_ray_status_observer(cmdline_parts):
                    continue

                cmdline = ' '.join(cmdline_parts)
                name = proc.info['name'] or ''
                
                # Check if this is a Ray process
                is_ray = False
                
                # Check process name
                for pattern in ray_patterns:
                    if pattern in name:
                        is_ray = True
                        break
                
                # Check command line for our specific ports/paths
                if not is_ray and cmdline:
                    if (str(self.ray_gcs_port) in cmdline or
                        str(self.ray_dashboard_port) in cmdline or
                        self.ray_tmpdir in cmdline or
                        self.ray_namespace in cmdline):
                        
                        for pattern in ['ray', 'python']:
                            if pattern in cmdline.lower():
                                is_ray = True
                                break
                
                if is_ray:
                    matching_pids.append(proc.info['pid'])
                    
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
        
        if matching_pids:
            if self._kill_processes(matching_pids, "Found Ray processes"):
                self.found_processes = True
    
    def stop_by_ray_command(self):
        """Try using 'ray stop' if available"""
        if not self.force:
            return
            
        print("\nAttempting 'ray stop' command...")
        try:
            # Set Ray address for the stop command
            env = os.environ.copy()
            env['RAY_ADDRESS'] = f"127.0.0.1:{self.ray_gcs_port}"
            
            result = subprocess.run(
                ['ray', 'stop', '--force'],
                capture_output=True,
                text=True,
                env=env,
                timeout=5
            )
            
            if result.returncode == 0:
                print("  Successfully executed 'ray stop'")
                self.found_processes = True
            else:
                print(f"  'ray stop' failed: {result.stderr}")
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            print("  Could not execute 'ray stop'")

    def stop_app_server(self):
        """Stop the local uvicorn/gunicorn app server and any reload-child processes."""
        print("\nChecking local app server...")

        app_pids = []

        # Find by port first (catches both parent reloader and single-process runs).
        for pid in self._get_processes_by_port(self.app_port):
            if self._is_my_process(pid):
                app_pids.append(pid)

        # Also scan by cmdline: the uvicorn reloader parent doesn't always bind the
        # port itself — the child worker does — so the parent can survive Ctrl+C
        # without showing up in lsof.
        for proc in self.psutil.process_iter(['pid', 'cmdline', 'username']):
            try:
                if proc.info['username'] != getpass.getuser():
                    continue
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'backend.app:app' in cmdline and proc.info['pid'] not in app_pids:
                    app_pids.append(proc.info['pid'])
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue

        if app_pids:
            if self._kill_processes(app_pids, f"Found app server on port {self.app_port}"):
                self.found_processes = True
        else:
            print(f"  No app server found on port {self.app_port}")

    def stop_log_guard(self) -> bool:
        """Stop this developer profile's log guard after Ray has stopped."""
        for proc in self.psutil.process_iter(['cmdline', 'username']):
            try:
                if proc.info['username'] != getpass.getuser():
                    continue
                parts = proc.info['cmdline'] or []
                if 'lifecycle.ray_log_guard' not in parts or '--userid' not in parts:
                    continue
                idx = parts.index('--userid')
                if idx + 1 >= len(parts) or parts[idx + 1] != self.userid:
                    continue
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except self.psutil.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            except self.psutil.NoSuchProcess:
                continue
            except (self.psutil.AccessDenied, self.psutil.TimeoutExpired):
                return False
        return True

    def clean_temp_files(self):
        """Clean up Ray temporary files"""
        print("\nCleaning temporary files...")
        
        if os.path.exists(self.ray_tmpdir):
            try:
                # Only remove if it's our temp directory
                if self.userid in self.ray_tmpdir:
                    import shutil
                    shutil.rmtree(self.ray_tmpdir, ignore_errors=True)
                    print(f"  Removed temp directory: {self.ray_tmpdir}")
            except Exception as e:
                print(f"  Could not remove temp directory: {e}")
    
    def verify_stopped(self) -> Tuple[bool, List[str]]:
        """Verify that Ray processes are stopped"""
        issues = []
        
        # Check ports
        ports_to_check = [
            ('GCS', self.ray_gcs_port),
            ('Dashboard', self.ray_dashboard_port),
            ('Client Server', self.ray_client_server_port)
        ]
        
        for name, port in ports_to_check:
            if not port:
                continue
            visible_pids = self._get_processes_by_port(int(port))
            if visible_pids:
                issues.append(f"Port {port} ({name}) is still in use")
                continue
            if self._port_has_listener(int(port)):
                issues.append(
                    f"Port {port} ({name}) is still listening but the owner is not visible"
                )
        
        # Check for any remaining Ray processes
        remaining = 0
        for proc in self.psutil.process_iter(['name', 'cmdline', 'username']):
            try:
                if proc.info['username'] != getpass.getuser():
                    continue
                    
                cmdline_parts = proc.info['cmdline'] or []
                if self._is_ray_status_observer(cmdline_parts):
                    continue

                cmdline = ' '.join(cmdline_parts).lower()
                name = (proc.info['name'] or '').lower()
                if ('ray' in name or
                    ('ray' in cmdline and str(self.ray_gcs_port) in cmdline)):
                    remaining += 1
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                pass
        
        if remaining > 0:
            issues.append(f"{remaining} Ray process(es) still running")

        remaining_app_pids = [
            pid for pid in self._get_processes_by_port(self.app_port)
            if self._is_my_process(pid)
        ]
        if remaining_app_pids:
            issues.append(f"App server is still running on port {self.app_port}")
        
        return len(issues) == 0, issues
    
    def run(self):
        """Main execution"""
        # Activate venv first
        self._activate_venv()
        
        # Import dependencies after venv activation
        self._import_dependencies()
        
        # Now load settings (needs project_settings module)
        self._load_settings()

        from project_settings import settings
        from shared.ray_connection import cluster_mode

        if cluster_mode(settings) == "shared" and getpass.getuser() != "vaultiq_user":
            print("Shared Ray is managed by vaultiq_user; skipping developer Ray stop")
            return 0
        
        self._print_header()
        
        # Try different methods to stop Ray
        self.stop_by_ports()
        self.stop_by_command_pattern()
        
        if self.force:
            self.stop_by_ray_command()

        # Stop the app server last so it isn't restarted by uvicorn reload.
        self.stop_app_server()
        
        # Print summary
        print("\n" + "=" * 50)
        if self.found_processes:
            print("Finished stopping Ray processes")
        else:
            print("No Ray processes found")
        print("=" * 50)
        
        # Verification
        print("\nVerification:")
        all_stopped, issues = self.verify_stopped()
        if all_stopped:
            if self.stop_log_guard() is False:
                all_stopped = False
                issues.append("Ray log guard is still running")
            else:
                self.clean_temp_files()
        
        if all_stopped:
            print("  ✓ All Ray processes stopped successfully")
            print(f"  ✓ Port {self.ray_gcs_port} is free")
            print(f"  ✓ Port {self.ray_dashboard_port} is free")
        else:
            print("  ⚠ WARNING: Some issues remain:")
            for issue in issues:
                print(f"    - {issue}")
            
            if not self.force:
                print("\n  Tip: Try running with --force to attempt additional cleanup")
        
        return 0 if all_stopped else 1


def main():
    parser = argparse.ArgumentParser(
        description='Stop Ray processes for a specific user',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script safely stops Ray processes for a specific user without
affecting other users' Ray instances.

Examples:
  # Stop Ray for current user
  %(prog)s
  
  # Stop Ray for specific user (required when running as ec2-user)
  %(prog)s --userid vaultiq_developer
  
  # Specify custom virtual environment
  %(prog)s --venv /path/to/venv
  
  # Force stop with additional cleanup
  %(prog)s --force

Safety features:
  - Only stops processes owned by the current system user
  - Never affects other users' Ray instances
  - Verifies all processes are stopped
  - Cleans up temporary files
        """
    )
    
    parser.add_argument(
        '--userid',
        help='User ID (required for ec2-user)'
    )
    
    parser.add_argument(
        '--venv',
        help='Path to virtual environment (optional if already in venv)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force stop using additional methods including ray stop command'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    # Redirect output if quiet mode
    if args.quiet:
        import io
        sys.stdout = io.StringIO()
    
    try:
        stopper = RayStopper(args.userid, args.venv, args.force)
        exit_code = stopper.run()
        
        # If quiet mode, only print if there were issues
        if args.quiet:
            output = sys.stdout.getvalue()
            sys.stdout = sys.__stdout__
            if exit_code != 0:
                print(output)
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
