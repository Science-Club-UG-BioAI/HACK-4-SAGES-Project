#for testing working of py - js 
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="Frontend"), name="static")

@app.get("/")
def read_index():
    return FileResponse("Frontend/index.html")

@app.get("/test")
def get_test():
    return {"message": "API jest g"}


@app.post("/upload")
async def upload_spectrum(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8").strip()

        if not text:
            raise HTTPException(status_code=400, detail="Plik jest pusty.")

        values = [float(x.strip()) for x in text.split(",") if x.strip() != ""]

        if not values:
            raise HTTPException(status_code=400, detail="Nie znaleziono danych liczbowych.")

        # tutaj w przyszłości:
        # prediction_df = model.predict(values)
        # a potem prediction_df zamieniasz na JSON

        # na razie zwracamy testowe dane
        result = [
            {"feature": "CO2", "prediction": 0.78, "error_margin": 0.05},
            {"feature": "Temperature", "prediction": 288.4, "error_margin": 3.2},
            {"feature": "Gravity", "prediction": 9.7, "error_margin": 0.4},
        ]

        return {
            "filename": file.filename,
            "vector_length": len(values),
            "first_values": values[:5],
            "results": result,
        }

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Plik ma niepoprawny format. Oczekiwano liczb oddzielonych przecinkami."
        )





if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=7000,
        reload=True
    )


