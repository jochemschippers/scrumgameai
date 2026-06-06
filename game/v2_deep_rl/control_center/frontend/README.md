# Frontend

This directory contains the browser client for the Control Center API.

## Entry Point

- `index.html`
- `main.js`

`main.js` owns application startup and event wiring. Feature code is split by
responsibility:

- `api/` handles authentication tokens and HTTP requests.
- `components/` contains page and feature behavior.
- `constants/` contains shared immutable defaults.
- `state/` owns the mutable browser state object.
- `utils/` contains formatting, DOM, and chart helpers.
- `styles/` contains shared and page-specific CSS.

The monolithic `app.js` is retained as an inactive migration reference. The
browser does not load it; `index.html` loads `main.js`.

## Pages

- Design: edit and validate game and training configs.
- Train: queue jobs, follow progress, and manage campaigns or autopilot.
- Inspect: review runs and checkpoints.
- Evaluate: run direct or comparative evaluations.
- Play: create and advance shared-board matches.

## Run

Start the backend from `game/v2_deep_rl/control_center/backend`:

```powershell
py run_api.py
```

Then open `http://127.0.0.1:8000/`. The backend serves these files, so opening
`index.html` directly is not required.
