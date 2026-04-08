# Frontend

This frontend now contains the first custom Control Center shell.

## Current scaffold

- `index.html`
- `app.js`
- `styles/app.css`

Pages currently wired in the shell:

- `Rules`
- `Models`
- `Training`
- `Testing`
- `Play`

## What works now

Once the backend API is running, the shell can:

- load game configs
- load training configs
- load runs
- load checkpoints
- run checkpoint compatibility checks
- queue training jobs
- queue robustness jobs
- list and stop jobs

## Run

Open [index.html](./index.html) in the browser after starting the backend API.

Set the API base URL to the running backend, usually:

`http://127.0.0.1:8000`

## Next frontend work

- move the full rule editor into the `Rules` page
- add run and checkpoint detail drawers
- add testing result viewers
- connect the `Play` page to play-session API routes
