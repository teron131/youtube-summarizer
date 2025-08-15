Add env variables.

Build:
```
uv pip install -U -r requirements.txt
```

Deploy:
```
python -m uvicorn app:app --host 0.0.0.0 --port $PORT
```
