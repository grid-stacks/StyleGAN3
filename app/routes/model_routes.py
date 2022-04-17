from typing import Optional

from fastapi import APIRouter, UploadFile, File, Body

from app.con import con
from app.crud.models import download, retrieve_models, retrieve_model, upload
from app.crud.processes import w_stds, G, generate_image, generate_video, seeding, timestring_run
from app.schemas import ModelSelectionSchema, ModelOptionEnum, SeedSchema

router = APIRouter()


@router.get("/create_table")
def create_table():
    try:
        with con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS MODEL (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    model TEXT,
                    file TEXT
                );
            """)
        return {"msg": "Table created successfully."}
    except Exception as e:
        print(e)
        return {"error": "Failed to create table"}


@router.post("/download_model")
def download_model(data: ModelSelectionSchema):
    get_sql = f"SELECT COUNT(id) FROM MODEL WHERE model = '{data.model}'"

    try:
        file_path = download(data.download_url, data.dest_folder)
    except FileExistsError as e:
        return {"error": str(e)}

    with con:
        get_data = con.execute(get_sql)
        get_count, = get_data.fetchone()

        if get_count:
            sql = f"UPDATE MODEL SET model = '{data.model}', file = '{file_path}' WHERE model = '{data.model}'"
            try:
                con.execute(sql)
                return {"msg": "Model updated successfully."}
            except Exception as e:
                print(e)
                return {"error": "Failed to add model"}
        else:
            sql = 'INSERT INTO MODEL (model, file) values(?, ?)'
            data = [
                (data.model, file_path)
            ]

            try:
                con.executemany(sql, data)
                return {"msg": "Model added successfully."}
            except Exception as e:
                print(e)
                return {"error": "Failed to add model"}


@router.post("/upload_model")
async def upload_model(file: UploadFile = File(...), model: ModelOptionEnum = Body(ModelOptionEnum.ELDARON),
                       dest_folder: Optional[str] = Body("files/models")):
    get_sql = f"SELECT COUNT(id) FROM MODEL WHERE model = '{model}'"

    try:
        file_path = await upload(file, dest_folder)
    except FileExistsError as e:
        return {"error": str(e)}

    with con:
        get_data = con.execute(get_sql)
        get_count, = get_data.fetchone()

        if get_count:
            sql = f"UPDATE MODEL SET model = '{model}', file = '{file_path}' WHERE model = '{model}'"
            try:
                con.execute(sql)
                return {"msg": "Model updated successfully."}
            except Exception as e:
                print(e)
                return {"error": "Failed to add model"}
        else:
            sql = 'INSERT INTO MODEL (model, file) values(?, ?)'
            data = [
                (model, file_path)
            ]

            try:
                con.executemany(sql, data)
                return {"msg": "Model added successfully."}
            except Exception as e:
                print(e)
                return {"error": "Failed to add model"}


@router.get("/get_models")
def get_models():
    try:
        models = retrieve_models()
        return {"msg": "Models retrieved successfully.", "data": models}
    except Exception as e:
        print(e)
        return {"error": "Failed to retrieve model"}


@router.get("/get_models/{model}")
def get_models(model: ModelOptionEnum):
    try:
        model = retrieve_model(model)
        return {"msg": "Model retrieved successfully.", "data": model}
    except ModuleNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        print(e)
        return {"error": "Failed to retrieve model"}


@router.get("/G")
def get_G(model: ModelOptionEnum):
    G(model)
    return True


@router.get("/w_stds")
def get_w_stds(model: ModelOptionEnum):
    w_stds(model)
    return True


@router.post("/generate_seeds")
def generate_seeds(data: SeedSchema):
    seeding(data)
    return True


@router.post("/run_timestring")
def run_timestring(data: SeedSchema, model: ModelOptionEnum):
    timestring_run(data, model)
    return True


@router.post("/process_image")
def process_image(data: SeedSchema, model: ModelOptionEnum, archive_name: str = Body(...)):
    generate_image(data, model, archive_name)


@router.post("/process_video")
def process_video(data: SeedSchema, model: ModelOptionEnum, video_name: str = Body(...)):
    generate_video(data, model, video_name)
