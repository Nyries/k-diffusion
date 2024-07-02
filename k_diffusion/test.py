import os


config = {
    "path" : "/v6" 
}

if not os.path.exists(f'Checkpoint{config["path"]}'):
    os.makedirs(f'Checkpoint{config["path"]}')
    print(f"New directory created: Checkpoint{config["path"]}")