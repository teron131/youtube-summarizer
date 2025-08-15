Add env variables.

# Backend
Build:
```
uv pip install -U -r requirements.txt
```

Deploy:
```
python -m uvicorn app:app --host 0.0.0.0 --port $PORT
```

# Frontend

Build:
```
npm ci && npm run build
```

Run:
```
npm run start -- -p $PORT
```