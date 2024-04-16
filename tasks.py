import os
import subprocess
from pathlib import Path

from invoke import task

project_root_path = Path(__file__).parent.resolve()


@task
def download_output(c):
    subprocess.run(f'scp -r onolab:~/workspace/sspo-exercise/output {project_root_path}', shell=True)

@task
def exec_matlab(c, file_name: str, jp_locale: bool = False):
    if jp_locale:
        subprocess.run(f"""ssh onolab 'cd ~/workspace/sspo-exercise/matlab && exec-matlab-jp {file_name}' && poetry run inv download-output > /dev/null""", shell=True)
    else:
        subprocess.run(f"""ssh onolab 'cd ~/workspace/sspo-exercise/matlab && exec-matlab {file_name}' && poetry run inv download-output > /dev/null""", shell=True)
