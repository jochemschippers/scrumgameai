# Frontend Restructuring Guide Prompt

Plak de tekst hieronder als nieuwe Cowork-sessie om de frontend te herstructureren.

---

## PROMPT (kopieer alles hieronder)

---

Ik wil de frontend van mijn project herstructureren van monolithische bestanden naar een nette modulaire mappenstructuur. **Alle bestaande functionaliteit en stijlen moeten exact behouden blijven.** Dit is puur een refactor — geen nieuwe features, geen stijlwijzigingen.

### Wat je gaat herstructureren

Het gaat om twee frontend-apps in dit project:

**App 1 — Control Center** (de grote, prioriteit 1):
`game/v2_deep_rl/control_center/frontend/`
- `app.js` — 4057 regels, volledig monolithisch, ~100+ globale functies
- `index.html` — 692 regels, groot HTML-template met 5 tabbladen
- `styles/app.css` — 1625 regels, alle stijlen in één bestand

**App 2 — Config Editor** (kleiner, prioriteit 2):
`game/v2_deep_rl/config_editor/`
- `app.js` — 614 regels
- `index.html` — 181 regels
- `styles.css` — 346 regels

---

### Doelstructuur — Control Center

Herstructureer `game/v2_deep_rl/control_center/frontend/` naar het volgende:

```
frontend/
├── index.html                  (lean shell, importeert main.js als module)
├── main.js                     (entry point: importeert alles, roept init aan)
│
├── constants/
│   └── defaults.js             (DEFAULT_GAME_CONFIG, AUTH_TOKEN_KEY, DICE_BOX_MODULE_URL)
│
├── state/
│   └── store.js                (het globale `state` object + getters/setters indien handig)
│
├── api/
│   └── client.js               (apiRequest, getToken, setToken, clearToken)
│
├── utils/
│   ├── formatting.js           (formatJson, formatNumber, formatCurrency, formatRunSourceLabel,
│   │                            checkpointUiLabel, checkpointCompatibilityTone, checkpointCategory,
│   │                            sidebarCheckpointOptions, escapeHtml)
│   ├── helpers.js              (clone, parseJsonEditor, parseNumberList, normalizeProductKey,
│   │                            $() shorthand, showMessage, clearMessage)
│   └── charts.js               (renderLineChart, renderBarChart, buildPolyline, renderTable)
│
├── components/
│   ├── auth.js                 (showLoginScreen, hideLoginScreen, logout)
│   ├── connection.js           (autoConnect, _tryConnect, _showConnectedUi, _showManualConnectUi,
│   │                            updateStatusCard, startProgressPolling, _runPollCycle)
│   ├── navigation.js           (setPage, pages object, updateSummaryPills, renderContextCard)
│   │
│   ├── configs/
│   │   ├── gameConfig.js       (loadActiveGameConfigIntoEditor, saveGameConfig,
│   │   │                        validateGameConfigDraft, renderGameConfigs,
│   │   │                        renderGameConfigValidation, refreshAll)
│   │   ├── trainingConfig.js   (loadActiveTrainingConfigIntoEditor, saveTrainingConfig,
│   │   │                        validateTrainingConfigDraft, renderTrainingConfigs,
│   │   │                        renderTrainingConfigValidation)
│   │   └── visualEditor.js     (ensureVisualGameConfig, rebuildVisualBoard,
│   │                            rebuildVisualProductNames, rebuildVisualRefinementRules,
│   │                            ensureVisualShapeConsistency, syncVisualShapeFromInputs,
│   │                            readVisualEditorIntoState, syncGameJsonEditorFromVisual,
│   │                            renderVisualMetadata, renderVisualProductNames,
│   │                            renderVisualBoardMatrix, renderVisualDiceRules,
│   │                            renderVisualRefinementRules, renderVisualIncidentCards,
│   │                            renderVisualEditor)
│   │
│   ├── training/
│   │   ├── jobs.js             (refreshJobs, renderJobs, renderJobDetail, renderJobLog,
│   │   │                        queueTrainingJob, queueRobustnessJob,
│   │   │                        renderTrainingSelectionSummary, renderTrainingPreflight,
│   │   │                        refreshTrainingPreflight)
│   │   ├── progress.js         (fetchTrainingProgress, fetchRunProgress,
│   │   │                        renderRuns, renderRunDetail)  (let op: kan ook in jobs.js als klein)
│   │   ├── autopilot.js        (fetchAutopilotData, renderAutopilotPanel,
│   │   │                        renderAutopilotTrainingPanel, refreshAutopilotSettings)
│   │   └── campaigns.js        (refreshCampaigns, renderCampaignPanel)
│   │
│   ├── evaluation/
│   │   ├── checkpoints.js      (refreshCheckpoints, runCompatibility,
│   │   │                        renderCompatibility, renderCheckpointDetail)
│   │   ├── directEval.js       (runDirectEvaluation, renderDirectEvaluation,
│   │   │                        exportDirectEvaluationJson, exportDirectEvaluationCsv)
│   │   └── comparison.js       (runCheckpointComparison, renderCheckpointComparison,
│   │                            exportComparisonJson, exportComparisonCsv)
│   │
│   └── play/
│       ├── session.js          (createPlaySession, advancePlayRound, refreshPlaySession,
│       │                        playSeatPayload, productNameById, latestPlayTurn, defaultSeatName)
│       ├── board.js            (renderPlayBoard, renderPlayTopbar, renderPlayStandings,
│       │                        renderPlayTurnLog, renderPlayActionButtons, renderPlaySeatEditor)
│       └── dice.js             (ensurePlayDiceBox, rollPlayDice, diceNotationFromTurnDice,
│                                renderFallbackDice, renderPlayDiceZone, renderPlayDicePreview,
│                                showPlayDiceOverlay, hidePlayDiceOverlay)
│
└── styles/
    ├── base.css                (CSS-variabelen, :root, kleuren, reset)
    ├── layout.css              (.app-shell, .app-header, .main-panel, .page, .grid varianten)
    ├── components.css          (.panel, .button, .field, .tag, .status-card, .message,
    │                            .form-stack, .list-stack, .list-card, .empty-state, utilities)
    └── pages/
        ├── rules.css           (visuele editor, .rules-editor-grid, .matrix-table,
        │                        .summary-grid, .advanced-json-grid)
        ├── training.css        (.progress-track, .progress-fill, .chart-card, .chart-stack)
        ├── evaluate.css        (.evaluation-form-stack en evaluate-pagina stijlen)
        └── play.css            (.play-dashboard, .play-grid, .play-topbar, .play-card,
                                 .play-dice-overlay, .play-dice-box, .play-dice-modal)
```

---

### Doelstructuur — Config Editor

Herstructureer `game/v2_deep_rl/config_editor/` naar:

```
config_editor/
├── index.html                  (lean shell, importeert main.js als module)
├── main.js                     (entry point)
├── constants/
│   └── defaults.js             (DEFAULT_GAME_CONFIG, standard config values)
├── components/
│   ├── editor.js               (form-to-state sync, product grid, board matrix,
│   │                            refinement rules, incident rules)
│   └── output.js               (JSON stringify/render, summary stats, filename suggestion)
├── utils/
│   └── helpers.js              (import/export, validation, DOM utils)
└── styles/
    ├── base.css                (variabelen, dark mode kleuren)
    ├── layout.css              (.shell, sticky header, twee-kolom layout)
    └── components.css          (cards, inputs, buttons)
```

---

### Technische vereisten

**Gebruik ES6 modules** (`type="module"` in de `<script>` tag in `index.html`). Exporteer elke functie expliciet en importeer ze waar nodig. Geen bundler nodig — native browser modules zijn voldoende.

**Richtlijnen per stap:**

1. **Begin met `constants/defaults.js`** — verplaats `DEFAULT_GAME_CONFIG`, `AUTH_TOKEN_KEY`, `DICE_BOX_MODULE_URL` daarheen. Let op: `DEFAULT_GAME_CONFIG` staat zowel in `control_center/app.js` als in `config_editor/app.js` — zorg dat ze identiek zijn voordat je dedupliceert.

2. **Dan `state/store.js`** — verplaats het globale `state` object. Exporteer het als een enkel object. Alle modules importeren dit object en muteren het direct (geen getters/setters verplicht — behoud het huidige patroon).

3. **Dan `api/client.js`** — verplaats `apiRequest`, `getToken`, `setToken`, `clearToken`. Dit zijn de enige functies die `localStorage` en `fetch` direct aanraken.

4. **Dan `utils/`** — puur functies zonder side effects eerst. Controleer altijd of er circulaire imports kunnen optreden.

5. **Dan de `components/`** — in de volgorde: `auth.js` → `connection.js` → `navigation.js` → `configs/*` → `training/*` → `evaluation/*` → `play/*`.

6. **`main.js`** — importeert alle componenten, roept `attachEvents()` aan en start de app op. De `attachEvents()` functie mag worden gesplitst in kleinere `bindXxxEvents()` functies per component, maar mag ook als geheel in `main.js` staan als dat overzichtelijker is.

7. **`index.html`** — verwijder alle inline `<script>` tags. Voeg één `<script type="module" src="main.js">` toe onderaan `<body>`. Alle HTML-structuur blijft **exact hetzelfde**.

8. **CSS splitsen** — splits `styles/app.css` op basis van de stijlen die al logisch gegroepeerd zijn. Kopieer secties, verwijder ze niet direct — valideer eerst dat niets mist.

**Wat absoluut hetzelfde moet blijven:**
- Alle DOM element IDs en classes (de JS is ervan afhankelijk)
- Alle HTML-structuur en tabbladen
- Alle CSS-variabelen (met name de `:root` kleuren — die gaan we later aanpassen)
- Alle API-aanroep patronen en URL-paden
- De 3D dice integratie (DICE_BOX_MODULE_URL, dynamic import)
- Polling gedrag en timers
- Authenticatie flow (Bearer token, localStorage key)

---

### Verificatiestap (doe dit aan het einde)

Nadat je klaar bent:

1. Open de Control Center in de browser en controleer alle 5 tabbladen: Design, Train, Inspect, Evaluate, Play
2. Verifieer dat de login flow werkt (401 → login overlay → token opslaan)
3. Verifieer dat de backend-connectie status correct werkt
4. Controleer de browser-console op module import errors
5. Doe een `grep -r "function " app.js` op de **originele** `app.js` en controleer of elke functienaam ergens in de nieuwe structuur terug te vinden is
6. Verwijder het originele `app.js` pas als alle bovenstaande checks groen zijn

---

### Wat je NIET doet

- Geen framework toevoegen (geen React, Vue, etc.)
- Geen build tool instellen (geen Webpack, Vite, etc.)
- Geen CSS wijzigen of klassen hernoemen
- Geen functionaliteit toevoegen of verwijderen
- Geen TypeScript
- Geen HTML-structuur wijzigen

---
