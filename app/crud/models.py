import os

import requests
from fastapi import UploadFile

from app.con import con
from app.schemas import ModelOptionEnum


def download(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    if os.path.exists(file_path):
        raise FileExistsError("File already exists.")

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        return os.path.abspath(file_path)
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))


async def upload(file: UploadFile, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = file.filename.split('/')[-1].replace(" ", "_")
    file_path = os.path.join(dest_folder, filename)

    if os.path.exists(file_path):
        raise FileExistsError("File already exists.")

    contents = await file.read()

    with open(file_path, 'wb') as f:
        f.write(contents)

    return os.path.abspath(file_path)


def retrieve_models():
    models = []

    sql = 'SELECT * FROM MODEL'

    with con:
        data = con.execute(sql)
        for row in data:
            models.append({
                "model": row[1],
                "file": row[2]
            })

    return models


def retrieve_model(model: ModelOptionEnum):
    sql = f"SELECT * FROM MODEL WHERE model = '{model}'"

    with con:
        data = con.execute(sql)
        model = data.fetchone()
        if model:
            return {
                "model": model[1],
                "file": model[2]
            }
        else:
            raise ModuleNotFoundError("Model not found.")
