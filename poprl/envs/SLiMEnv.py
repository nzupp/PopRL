"""
SLiM gym env
"""
import gymnasium as gym
import subprocess
import time
from pathlib import Path

class SLiMEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, slim_file, task, timeout=10.0):
        super().__init__()
        self.slim_file = slim_file
        self.task = task
        self.timeout = timeout
        self.process = None
        self.previous_log_data = ""
        self.current_log_data = ""
        self.step_count = 0
        self.context = {}
        self.previous_state = None
        self.action_space = task.action_space
        self.observation_space = task.observation_space

    def _cleanup_files(self):
        """Remove flag, state, and completion files between episodes"""
        try:
            for filename in ['flag.txt', 'state.txt', 'generation_complete.txt']:
                file_path = Path(filename)
                
                if file_path.exists():
                    
                    if self._wait_for_file_release(str(file_path), timeout=1.0):
                        file_path.unlink()
                    
                    else:
                        print(f"Warning: {filename} is still in use.")
        
        except Exception as e:
            print(f"Error cleaning up files: {e}")

    def _wait_for_file_release(self, filepath, timeout=1.0):
        """Poll until file is readable or timeout is reached"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            
            try:
                with open(filepath, 'r+'):
                    return True
            
            except (IOError, PermissionError):
                time.sleep(0.01)
        
        return False

    def _make_flag(self, param_string):
        """Atomically write flag file to signal new params to SLiM process"""
        try:
            temp_file = Path('flag.txt.tmp')
            flag_file = Path('flag.txt')
            temp_file.write_text(param_string)
            temp_file.rename(flag_file)
            return True
        
        except Exception as e:
            print(f"Error writing flag file: {e}")
            return False

    def step(self, action):
        """Poll for state from SLiM, compute reward, and signal next action via flag file"""
        self.step_count += 1
        param_string = self.task.process_action(action)

        terminated = False
        truncated = False
        reward = 0
        current_state = None

        complete_path = Path('generation_complete.txt')
        state_path = Path('state.txt')
        flag_path = Path('flag.txt')

        start_time = time.time()
        while True:
            
            if complete_path.exists():
                self.close()
                return current_state or self.task.get_initial_state(self.context), reward, True, False, {"completed": True}

            
            if self.process.poll() is not None:
                
                if complete_path.exists():
                    self.close()
                    return current_state or self.task.get_initial_state(self.context), reward, True, False, {"completed": True}
                
                if current_state is not None:
                    self.close()
                    return current_state, reward, True, False, {"completed": True}
                
                break

            if time.time() - start_time > self.timeout:
                print("TIMEOUT - killing episode")
                self.close()
                self.reset()
                return self.task.get_initial_state(self.context), -1, True, False, {"error": "timeout"}

            if state_path.exists() and not flag_path.exists():
                
                if self._wait_for_file_release('state.txt'):
                    
                    with open('state.txt', 'r') as f:
                        self.current_log_data = f.read().strip()

                    # Diff against previous log to get only new generation data
                    state_data = self.current_log_data[len(self.previous_log_data):].strip()
                    current_state = self.task.process_state(state_data, self.context, self.step_count)

                    if self.previous_state is not None:
                        reward = self.task.calculate_reward(current_state, self.context)

                    self.previous_state = current_state
                    self.previous_log_data = self.current_log_data
                    self._make_flag(param_string)

                    return current_state, reward, terminated, truncated, {}

            time.sleep(0.01)

        if current_state is not None:
            return current_state, reward, True, False, {"episode_complete": True}

        # Process died unexpectedly, recover and reset
        return_code = self.process.returncode
        stdout, stderr = "", ""
        try:
            stdout, stderr = self.process.communicate(timeout=1)
        
        except:
            pass

        print(f"Process ended unexpectedly at step {self.step_count}")
        initial_state, _ = self.reset()
        
        return initial_state, -1, True, False, {
            "error": "process_ended",
            "return_code": return_code,
            "step_count": self.step_count,
            "stdout": stdout,
            "stderr": stderr
        }

    def reset(self, seed=None):
        """Kill existing SLiM process, clean up files, and launch fresh simulation"""
        super().reset(seed=seed)
        
        if self.process is not None:
            self.close()
        
        self._cleanup_files()
        self.previous_log_data = ""
        self.current_log_data = ""
        self.step_count = 0
        self.context = {}
        self.previous_state = None
        
        try:
            self.process = subprocess.Popen(
                ["slim", self.slim_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
        
        except Exception as e:
            print(f"Error starting SLiM process: {e}")
            self.process = None
        
        initial_state = self.task.get_initial_state(self.context)
        self.previous_state = initial_state
        
        return initial_state, {}

    def close(self):
        """Kill SLiM process and clean up IPC files"""
        if self.process:
            
            try:
                self.process.kill()
                self.process.wait(timeout=5)
            
            except:
                print("Process kill timeout")

        time.sleep(0.05)
        self._cleanup_files()