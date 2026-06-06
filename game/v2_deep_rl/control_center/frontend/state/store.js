/**
 * Frontend State Store.
 * 
 * This module exports a single shared state object representing the global client-side cache and session states
 * (active tab page, selected config assets, loaded run directories, job list, autopilot states, evaluation parameters,
 * and live game sessions).
 * 
 * Connections:
 *   - Imported by: `main.js` and almost all component and utility files to read/write shared interface state.
 */

export const state = {
  apiBaseUrl: window.location.protocol.startsWith("http")
    ? window.location.origin
    : "http://188.166.52.37:8000",
  userRole: null,   // "admin" | "guest" — populated after login
  health: null,
  gameConfigs: [],
  trainingConfigs: [],
  runs: [],
  checkpoints: [],
  jobs: [],
  activePage: "rules",
  activeGameConfigId: "",
  activeTrainingConfigId: "",
  activeCheckpointId: "",
  compatibility: null,
  activeGameConfigPayload: null,
  activeTrainingConfigPayload: null,
  visualGameConfig: null,
  playSession: null,
  directEvaluation: null,
  comparisonEvaluation: null,
  playSeatDrafts: [
    { id: "draft_1", type: "human", display_name: "Player" },
    { id: "draft_2", type: "model-expert", display_name: "AI Expert" },
  ],
  activeProgressJobId: null,
  activeProgressRunId: null,
  trainingProgress: null,
  trainingPreflight: null,
  gameConfigValidation: null,
  trainingConfigValidation: null,
  activeRunId: null,
  runDetail: null,
  activeJobDetailId: null,
  jobDetail: null,
  jobLog: null,
  progressPollHandle: null,
  includeCheckpointSelections: false,
  autopilotSettings: null,
  autopilotHistory: [],
  autopilotStopRequested: false,
  campaigns: [],
  activeCampaignId: null,
  jobsPage: 0,
  runsPage: 0,
  playDiceBox: null,
  playDiceBoxReady: false,
  playDiceBoxInitPromise: null,
  runRating: null,
  messageTimer: null,
};
