import os
import subprocess

os_name = os.name

def git_pull(filename):
    try:
        result = subprocess.run(['git','pull'], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Une erreur s'est produite : {e.stderr}"

def git_add(filename=None):
    if filename == None:
        filename = '.'
    try:
        result = subprocess.run(['git', 'add', filename], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Une erreur s'est produite : {e.stderr}"

def git_commit(name):
    try:
        result = subprocess.run(['git', 'commit', '-m',name], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Une erreur s'est produite : {e.stderr}"

def git_push():
    try:
        result = subprocess.run(['git', 'push'], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Une erreur s'est produite : {e.stderr}"