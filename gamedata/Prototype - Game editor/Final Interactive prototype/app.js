/* Prototype V26 — no backend. State stored in-memory only. */

(function () {
  'use strict';

  const USER = { name: 'Micha', role: 'admin' };

  const DEFAULTS = {
    templateKey: '',
    templateName: '',

    // Basic
    playersCount: 4,
    boardName: 'classical setup V1.00',
    boardDescription: 'Standard classical mostly used in physical games',
    productsCount: 7,
    sprintsPerProduct: 4,
    tokensPerPlayer: 6,
    startingMoney: 25000,
    roundsPerPlayer: 6,

    ringValue: 5000,
    costContinue: 0,
    costSwitchMid: 5000,
    costSwitchAfter: 0,
    mandatoryLoan: 50000,
    loanInterest: 5000,
    penaltyNeg: 1000,
    penaltyPos: 5000,

    dailyScrumsPerSprint: 5,
    dailyScrumTarget: 12,

    // Refinements
    refinementCards: 8,
    refinementsActive: 'Yes',
    refinementModel: 'Standard (ID 301)',
    refRollIncreaseMin: 1,
    refRollIncreaseMax: 2,
    refRollDecreaseMin: 19,
    refRollDecreaseMax: 20,

    // Incident (active in digital, optional)
    incidentCards: 8,
    incidentsActive: 'Yes',
    incidentFrequency: 'Normal',
    allowPlayerSpecificIncidents: 'No',
  };

  const LIMITS = {
    playersCount: { min: 1, max: 12 },
    boardName: { minLen: 4, maxLen: 64 },
    boardDescription: { minLen: 0, maxLen: 255 },
    productsCount: { min: 4, max: 12 },
    sprintsPerProduct: { min: 4, max: 12 },
    tokensPerPlayer: { min: 4, max: 32 },
    startingMoney: { min: 0, max: 250000 },
    roundsPerPlayer: { min: 4, max: 32 },

    money: { min: 0, max: 250000 },

    dailyScrumsPerSprint: { min: 4, max: 12 },
    dailyScrumTarget: { min: 4, max: 12 },

    playerName: { minLen: 4, maxLen: 32 },
    productName: { minLen: 4, maxLen: 32 },
    productDesc: { minLen: 0, maxLen: 128 },

    difficulty: { min: 0.0, max: 1.0 },

    refinementCards: { min: 1, max: 32 },
    rollInc: { min: 1, max: 10 },
    rollDec: { min: 11, max: 20 },

    layoutValue: { min: 0, max: 32 },
    layoutFeatures: { min: 0, max: 32 },
  };

  /** @type {{ activeTab: string, isValid: boolean, staged: any[], boardId: string, data: any, players: any[], products: any[], layout: any, selectedCell: {p:number,s:number} | null, layoutChanges: any[] }} */
  const state = {
    activeTab: 'start',
    isValid: false,
    makeFinal: false,
    staged: [],
    boardId: generateBoardId(),
    data: {},
    players: [],
    products: [],
    layout: { cells: [] }, // [p][s] => {value, features, original:{...}}
    selectedCell: null,
    layoutChanges: [],
  };

  function generateBoardId() {
    // deterministic-ish for UX: timestamp + random chunk
    const rand = Math.floor(Math.random() * 1e6).toString().padStart(6, '0');
    return `board_${Date.now()}_${rand}`;
  }

  function $(id) {
    return document.getElementById(id);
  }

  function nowStamp() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const yy = String(d.getFullYear()).slice(-2);
    return `${pad(d.getDate())}/${pad(d.getMonth() + 1)}/${yy} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }

  function clampInt(v, min, max) {
    const n = Number.parseInt(String(v), 10);
    if (Number.isNaN(n)) return null;
    return Math.min(max, Math.max(min, n));
  }

  function clampFloat(v, min, max) {
    const n = Number.parseFloat(String(v));
    if (Number.isNaN(n)) return null;
    return Math.min(max, Math.max(min, n));
  }

  function isHexColor(v) {
    const s = String(v || '').trim();
    return /^#([0-9a-fA-F]{6})$/.test(s);
  }

  function stageChange(path, oldValue, newValue, meta = {}) {
    if (oldValue === newValue) return;
    state.staged.push({
      path,
      oldValue,
      newValue,
      user: USER.name,
      role: USER.role,
      dateTime: nowStamp(),
      ...meta,
    });
    updateStagedUI();
  }

  function clearStaged() {
    state.staged = [];
    state.layoutChanges = [];
    // Reset layout to original
    for (let p = 0; p < state.layout.cells.length; p++) {
      for (let s = 0; s < state.layout.cells[p].length; s++) {
        const c = state.layout.cells[p][s];
        c.value = c.original.value;
        c.features = c.original.features;
      }
    }
    state.selectedCell = null;
    updateStagedUI();
    renderBoardGrid();
    renderLayoutChanges();
    updateProgressUI();
  }

  function updateStagedUI() {
    const count = state.staged.length;
    const badge = $('uiStagedBadge');
    const summary = $('uiStagedSummary');

    // Note: "Staged changes" panel was removed in V13; the badge may still exist.
    if (!badge && !summary) return;

    if (count === 0) {
      if (badge) {
        badge.dataset.state = 'none';
        badge.textContent = 'No staged changes';
      }
      if (summary) summary.textContent = 'No staged changes.';
    } else {
      if (badge) {
        badge.dataset.state = 'warn';
        badge.textContent = `${count} staged change${count === 1 ? '' : 's'}`;
      }
      if (summary) summary.textContent = `${count} change${count === 1 ? '' : 's'} staged.`;
    }
  }

  function setValidationState(ok, errors) {
    state.isValid = ok;
    const badge = $('uiValidationBadge');
    const btnSaveTop = $('btnSaveTop');
    const btnSave = $('btnSave');

    if (badge) {
      if (ok) {
        badge.dataset.state = 'ok';
        badge.textContent = 'Valid';
      } else {
        badge.dataset.state = 'warn';
        badge.textContent = errors.length ? `Errors: ${errors.length}` : 'Not validated';
      }
    }
    if (btnSaveTop) btnSaveTop.disabled = !ok;
    if (btnSave) btnSave.disabled = !ok;

    updateProgressUI();
  }

  function applyDefaults(templateKey) {
    // For now: only differs in a couple of values, but the rest stays consistent.
    const d = { ...DEFAULTS };
    if (templateKey === 'classical') {
      d.templateKey = 'classical';
      d.templateName = 'Classical setup v1.00';
      d.boardName = 'classical setup V1.00';
      d.boardDescription = 'Standard classical mostly used in physical games';
    }
    if (templateKey === 'simsetup') {
      d.templateKey = 'simsetup';
      d.templateName = 'Sim setup v1.00';
      d.playersCount = 8;
      d.productsCount = 5;
      d.sprintsPerProduct = 7;
      d.boardName = 'sim setup V1.00';
      d.boardDescription = 'Entrepreneurs that dare';
    }
    state.data = d;

    // Rebuild dependent tables & matrix
    buildPlayersFromCount(d.playersCount);
    buildProductsFromCount(d.productsCount);
    buildLayoutMatrix(d.productsCount, d.sprintsPerProduct);

    // Reset staged + validation
    state.staged = [];
    state.layoutChanges = [];
    state.selectedCell = null;
    setValidationState(false, []);
    updateStagedUI();
  }

  function buildPlayersFromCount(count) {
    const list = [];
    const defaultNames = ['Player 1', 'Player 2', 'Player 3', 'Player 4'];
    const defaultColors = ['#FF0000', '#00FF00', '#FF00FF', '#FFFF00'];
    const defaultStrategies = ['scrum', 'waterfall', 'random', 'scrum'];
    const defaultDifficulties = [0.3, 0.3, 0.3, 0.8];

    for (let i = 0; i < count; i++) {
      list.push({
        number: i + 1,
        name: defaultNames[i] || `Player ${i + 1}`,
        color: defaultColors[i % defaultColors.length],
        strategy: defaultStrategies[i % defaultStrategies.length],
        difficulty: defaultDifficulties[i % defaultDifficulties.length],
      });
    }
    state.players = list;
  }

  function buildProductsFromCount(count) {
    const list = [];
    // NOTE: Product colors can be edited in TAB "Product".
    // Layout TAB uses a fixed palette to match the V2 screenshot.
    const colors = ['#F2C94C', '#F2994A', '#EB5757', '#27AE60', '#2D9CDB', '#9B51E0', '#828282', '#F2C94C'];
    for (let i = 0; i < count; i++) {
      list.push({
        number: i + 1,
        name: `Product ${i + 1}`,
        color: colors[i % colors.length],
        description: '',
      });
    }
    state.products = list;
  }

  function buildLayoutMatrix(productsCount, sprintsPerProduct) {
    // Default cell values.
    // For the classic 7x4 setup we seed the exact values visible in the V2 screenshot.
    const cells = [];
    const isClassicScreenshot = (productsCount === 7 && sprintsPerProduct === 4);
    const screenshotSeed = isClassicScreenshot ? {
      values: [
        [4, 2, 1, 1],
        [5, 3, 2, 1],
        [6, 4, 3, 2],
        [5, 3, 2, 1],
        [4, 3, 2, 1],
        [5, 3, 2, 1],
        [7, 4, 3, 2],
      ],
      features: [
        [3, 3, 2, 1],
        [2, 2, 1, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 1],
        [3, 2, 2, 1],
        [2, 2, 1, 1],
        [1, 1, 1, 1],
      ],
    } : null;
    for (let p = 0; p < productsCount; p++) {
      const row = [];
      for (let s = 0; s < sprintsPerProduct; s++) {
        const v = screenshotSeed ? screenshotSeed.values[p][s] : 1;
        const f = screenshotSeed ? screenshotSeed.features[p][s] : 2;
        const c = { value: v, features: f, original: { value: v, features: f } };
        row.push(c);
      }
      cells.push(row);
    }
    state.layout.cells = cells;
  }

  function init() {
    // Wire up tabs
    document.querySelectorAll('.tab').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const tab = e.currentTarget.getAttribute('data-tab');
        if (tab) setTab(tab);
      });
    });

    // Sidebar go-to buttons
    document.querySelectorAll('[data-goto]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const tab = e.currentTarget.getAttribute('data-goto');
        if (tab) setTab(tab);
      });
    });

    // Top validate/save (with micro progress feedback)
    $('btnValidateTop')?.addEventListener('click', () => runValidateWithProgress());
    $('btnSaveTop')?.addEventListener('click', () => saveWithProgress());

    // Progress overview toggle (V13)
    // V14: when hidden, the main panel expands to full width.
    $('btnToggleProgress')?.addEventListener('click', () => {
      const aside = $('progressAside');
      const btn = $('btnToggleProgress');
      const workspace = document.querySelector('.workspace');
      if (!aside || !btn || !workspace) return;
      const nowHidden = aside.classList.toggle('is-hidden');
      workspace.classList.toggle('progress-hidden', nowHidden);
      btn.setAttribute('aria-expanded', String(!nowHidden));
      btn.textContent = nowHidden ? 'Progress ▸' : 'Progress ▾';
    });

    // Template select
    $('templateSelect')?.addEventListener('change', (e) => {
      const key = e.target.value;
      const old = state.data.templateKey;

      if (!key) {
        state.data.templateKey = '';
        state.data.templateName = '';
        $('uiSelectedTemplate').textContent = '—';
        stageChange('template', old, '(none)');
        updateProgressUI();
        setTab('start');
        return;
      }

      applyDefaults(key);
      stageChange('template', old, key);
      $('uiSelectedTemplate').textContent = state.data.templateName;
      fillAllInputs();
      renderAll();
    });

    // Basic fields
    bindInputNumber('playersCount', 'playersCount', LIMITS.playersCount, (n) => {
      const old = state.players.length;
      buildPlayersFromCount(n);
      stageChange('basic.playersCount', old, n);
      renderPlayersTable();
    });

    bindInputText('boardName', 'boardName', LIMITS.boardName);
    bindInputText('boardDescription', 'boardDescription', LIMITS.boardDescription);
    // Incident (active)
    bindInputNumber('incidentCards', 'incidentCards', LIMITS.refinementCards);
    bindSelect('incidentsActive', 'incidentsActive');
    bindSelect('incidentFrequency', 'incidentFrequency');
    bindSelect('allowPlayerSpecificIncidents', 'allowPlayerSpecificIncidents');


    bindInputNumber('productsCount', 'productsCount', LIMITS.productsCount, (n) => {
      const old = state.products.length;
      buildProductsFromCount(n);
      buildLayoutMatrix(n, state.data.sprintsPerProduct);
      state.selectedCell = null;
      stageChange('basic.productsCount', old, n);
      renderProductsTable();
      renderBoardGrid();
      renderLayoutChanges();
    });

    bindInputNumber('sprintsPerProduct', 'sprintsPerProduct', LIMITS.sprintsPerProduct, (n) => {
      const old = state.data.sprintsPerProduct;
      buildLayoutMatrix(state.data.productsCount, n);
      state.selectedCell = null;
      stageChange('basic.sprintsPerProduct', old, n);
      renderBoardGrid();
      renderLayoutChanges();
    });

    bindInputNumber('tokensPerPlayer', 'tokensPerPlayer', LIMITS.tokensPerPlayer);
    bindInputNumber('startingMoney', 'startingMoney', LIMITS.startingMoney);
    bindInputNumber('roundsPerPlayer', 'roundsPerPlayer', LIMITS.roundsPerPlayer);

    // money fields
    bindInputNumber('ringValue', 'ringValue', LIMITS.money);
    bindInputNumber('costContinue', 'costContinue', LIMITS.money);
    bindInputNumber('costSwitchMid', 'costSwitchMid', LIMITS.money);
    bindInputNumber('costSwitchAfter', 'costSwitchAfter', LIMITS.money);
    bindInputNumber('mandatoryLoan', 'mandatoryLoan', LIMITS.money);
    bindInputNumber('loanInterest', 'loanInterest', LIMITS.money);
    bindInputNumber('penaltyNeg', 'penaltyNeg', LIMITS.money);
    bindInputNumber('penaltyPos', 'penaltyPos', LIMITS.money);

    bindInputNumber('dailyScrumsPerSprint', 'dailyScrumsPerSprint', LIMITS.dailyScrumsPerSprint);
    bindInputNumber('dailyScrumTarget', 'dailyScrumTarget', LIMITS.dailyScrumTarget);

    // Refinements
    bindInputNumber('refinementCards', 'refinementCards', LIMITS.refinementCards);
    bindSelect('refinementsActive', 'refinementsActive');
    bindSelect('refinementModel', 'refinementModel');
    bindInputNumber('refRollIncreaseMin', 'refRollIncreaseMin', LIMITS.rollInc);
    bindInputNumber('refRollIncreaseMax', 'refRollIncreaseMax', LIMITS.rollInc);
    bindInputNumber('refRollDecreaseMin', 'refRollDecreaseMin', LIMITS.rollDec);
    bindInputNumber('refRollDecreaseMax', 'refRollDecreaseMax', LIMITS.rollDec);

    // Layout editor
    $('btnApplyCell')?.addEventListener('click', applySelectedCellEdit);
    $('btnResetCell')?.addEventListener('click', resetSelectedCell);

    // Validate tab buttons
    $('btnValidate')?.addEventListener('click', () => runValidateWithProgress());
    $('btnSave')?.addEventListener('click', () => saveWithProgress());
    $('btnDownloadJson')?.addEventListener('click', downloadJson);

    // Staged actions
    $('btnDiscardStaged')?.addEventListener('click', () => {
      clearStaged();
      fillAllInputs();
      renderAll();
    });

    $('btnCopyPatch')?.addEventListener('click', copyPatchToClipboard);

    // Board ID and defaults
    $('boardId').value = state.boardId;

    applyDefaults(DEFAULTS.templateKey);
    $('uiSelectedTemplate').textContent = state.data.templateName;

    fillAllInputs();
    renderAll();
  }

  function setTab(tab) {
    state.activeTab = tab;

    document.querySelectorAll('.tab').forEach((btn) => {
      const t = btn.getAttribute('data-tab');
      const isActive = t === tab;
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });

    document.querySelectorAll('.panel').forEach((panel) => {
      panel.classList.toggle('active', panel.getAttribute('data-panel') === tab);
    });

    // Soft progress update
    updateProgressUI();
  }

  function bindSelect(id, key) {
    const el = $(id);
    if (!el) return;
    el.addEventListener('change', (e) => {
      const old = state.data[key];
      state.data[key] = e.target.value;
      stageChange(`data.${key}`, old, state.data[key]);
      updateProgressUI();
    });
  }

  function bindInputText(id, key, limits) {
    const el = $(id);
    if (!el) return;
    el.addEventListener('input', (e) => {
      const old = state.data[key];
      const val = String(e.target.value || '');
      state.data[key] = val;
      stageChange(`data.${key}`, old, val);
      validateInlineText(id, val, limits);
      updateProgressUI();
    });
  }

  function bindInputNumber(id, key, limits, onAfterApply) {
    const el = $(id);
    if (!el) return;
    el.addEventListener('input', (e) => {
      const old = state.data[key];
      const raw = e.target.value;
      const n = clampInt(raw, limits.min, limits.max);
      if (n === null) {
        // keep raw in field, show inline error
        validateInlineNumber(id, raw, limits);
        return;
      }
      state.data[key] = n;
      stageChange(`data.${key}`, old, n);
      validateInlineNumber(id, n, limits);

      if (typeof onAfterApply === 'function') onAfterApply(n);

      renderAllLight();
    });
  }

  function validateInlineNumber(id, val, limits) {
    const err = $('err_' + id);
    if (!err) return;
    const n = Number(val);
    if (!Number.isFinite(n)) {
      err.textContent = 'Please enter a number.';
      return;
    }
    if (n < limits.min || n > limits.max) {
      err.textContent = `Must be between ${limits.min} and ${limits.max}.`;
      return;
    }
    err.textContent = '';
  }

  function validateInlineText(id, val, limits) {
    const err = $('err_' + id);
    if (!err) return;
    const len = val.trim().length;
    if (len < limits.minLen) {
      err.textContent = `Must be at least ${limits.minLen} characters.`;
      return;
    }
    if (len > limits.maxLen) {
      err.textContent = `Must be at most ${limits.maxLen} characters.`;
      return;
    }
    err.textContent = '';
  }

  function fillAllInputs() {
    // Start
    const sel = $('templateSelect');
    if (sel) sel.value = state.data.templateKey;

    // Basic
    setVal('playersCount', state.data.playersCount);
    setVal('boardId', state.boardId);
    setVal('boardName', state.data.boardName);
    setVal('boardDescription', state.data.boardDescription);
    setVal('productsCount', state.data.productsCount);
    setVal('sprintsPerProduct', state.data.sprintsPerProduct);
    setVal('tokensPerPlayer', state.data.tokensPerPlayer);
    setVal('startingMoney', state.data.startingMoney);
    setVal('roundsPerPlayer', state.data.roundsPerPlayer);

    setVal('ringValue', state.data.ringValue);
    setVal('costContinue', state.data.costContinue);
    setVal('costSwitchMid', state.data.costSwitchMid);
    setVal('costSwitchAfter', state.data.costSwitchAfter);
    setVal('mandatoryLoan', state.data.mandatoryLoan);
    setVal('loanInterest', state.data.loanInterest);
    setVal('penaltyNeg', state.data.penaltyNeg);
    setVal('penaltyPos', state.data.penaltyPos);

    setVal('dailyScrumsPerSprint', state.data.dailyScrumsPerSprint);
    setVal('dailyScrumTarget', state.data.dailyScrumTarget);

    // Refinements
    setVal('refinementCards', state.data.refinementCards);
    setVal('refinementsActive', state.data.refinementsActive);
    setVal('refinementModel', state.data.refinementModel);
    setVal('refRollIncreaseMin', state.data.refRollIncreaseMin);
    setVal('refRollIncreaseMax', state.data.refRollIncreaseMax);
    setVal('refRollDecreaseMin', state.data.refRollDecreaseMin);
    setVal('refRollDecreaseMax', state.data.refRollDecreaseMax);

    // Incident (disabled)
    setVal('incidentCards', state.data.incidentCards);
    setVal('incidentsActive', state.data.incidentsActive);
    setVal('incidentFrequency', state.data.incidentFrequency);
    setVal('allowPlayerSpecificIncidents', state.data.allowPlayerSpecificIncidents);
  }

  function setVal(id, v) {
    const el = $(id);
    if (!el) return;
    el.value = v;
  }

  function renderAll() {
    updateStagedUI();
    renderPlayersTable();
    renderProductsTable();
    renderBoardGrid();
    renderLayoutChanges();
    updateProgressUI();
  }

  function renderAllLight() {
    updateStagedUI();
    updateProgressUI();
  }

  
function updateProgressUI() {
  const templateChosen = !!(state.data.templateKey && String(state.data.templateKey).trim());

  // Boardname must be different from the selected template name (copied default).
  const templateNameNorm = String(state.data.templateName || '').trim().toLowerCase();
  const boardNameNorm = String(state.data.boardName || '').trim().toLowerCase();
  const boardNameDiffersFromTemplate = !!boardNameNorm && !!templateNameNorm && boardNameNorm !== templateNameNorm;

  // Statusbar: Template + Boardname
  const uiT = $('uiSelectedTemplate');
  if (uiT) uiT.textContent = templateChosen ? state.data.templateName : '—';
  const uiB = $('uiBoardName');
  if (uiB) uiB.textContent = templateChosen ? (state.data.boardName || '—') : '—';

  // Inline boardname error + red input until it differs
  const boardNameEl = $('boardName');
  const errBoardName = $('err_boardName');
  if (boardNameEl && templateChosen) {
    const len = String(state.data.boardName || '').trim().length;
    const lenOk = len >= LIMITS.boardName.minLen && len <= LIMITS.boardName.maxLen;
    const diffOk = boardNameDiffersFromTemplate;
    boardNameEl.classList.toggle('is-invalid', !(lenOk && diffOk));
    if (errBoardName) {
      if (!lenOk) {
        errBoardName.textContent = `Must be between ${LIMITS.boardName.minLen} and ${LIMITS.boardName.maxLen} characters.`;
      } else if (!diffOk) {
        errBoardName.textContent = 'Must be different from the template name.';
      } else {
        errBoardName.textContent = '';
      }
    }
  } else if (boardNameEl) {
    boardNameEl.classList.remove('is-invalid');
    if (errBoardName) errBoardName.textContent = '';
  }

  const basicValid =
    templateChosen &&
    inRange(state.data.playersCount, LIMITS.playersCount) &&
    inRange(state.data.productsCount, LIMITS.productsCount) &&
    inRange(state.data.sprintsPerProduct, LIMITS.sprintsPerProduct) &&
    inRange(state.data.tokensPerPlayer, LIMITS.tokensPerPlayer) &&
    inRange(state.data.startingMoney, LIMITS.startingMoney) &&
    inRange(state.data.roundsPerPlayer, LIMITS.roundsPerPlayer) &&
    String(state.data.boardName || '').trim().length >= LIMITS.boardName.minLen &&
    String(state.data.boardName || '').trim().length <= LIMITS.boardName.maxLen &&
    boardNameDiffersFromTemplate &&
    String(state.data.boardDescription || '').trim().length >= LIMITS.boardDescription.minLen &&
    String(state.data.boardDescription || '').trim().length <= LIMITS.boardDescription.maxLen;

  const playerValid =
    templateChosen &&
    state.players.every((p) => {
      const nameOk = String(p.name || '').trim().length >= LIMITS.playerName.minLen &&
        String(p.name || '').trim().length <= LIMITS.playerName.maxLen;
      const colorOk = /^#[0-9A-Fa-f]{6}$/.test(String(p.color || ''));
      const diffOk = inRange(p.difficulty, LIMITS.difficulty);
      const stratOk = ['scrum', 'waterfall', 'random'].includes(String(p.strategy || ''));
      return nameOk && colorOk && diffOk && stratOk;
    });

  const productValid =
    templateChosen &&
    state.products.every((p) => {
      const nameOk = String(p.name || '').trim().length >= LIMITS.productName.minLen &&
        String(p.name || '').trim().length <= LIMITS.productName.maxLen;
      const colorOk = /^#[0-9A-Fa-f]{6}$/.test(String(p.color || ''));
      const desc = String(p.description || '');
      const descOk = desc.length >= LIMITS.productDesc.minLen && desc.length <= LIMITS.productDesc.maxLen;
      return nameOk && colorOk && descOk;
    });

  const layoutValid =
    templateChosen &&
    Array.isArray(state.layout?.cells) &&
    state.layout.cells.length === state.data.productsCount &&
    state.layout.cells.every((row) => Array.isArray(row) && row.length === state.data.sprintsPerProduct) &&
    state.layout.cells.every((row) =>
      row.every((cell) => inRange(cell.value, LIMITS.layoutValue) && inRange(cell.features, LIMITS.layoutFeatures))
    );

  const refinementsValid =
    templateChosen &&
    inRange(state.data.refinementCards, LIMITS.refinementCards) &&
    ['Yes', 'No'].includes(String(state.data.refinementsActive || 'Yes')) &&
    ['Standard (ID 301)', 'Chaos (Custom)'].includes(String(state.data.refinementModel || 'Standard (ID 301)')) &&
    inRange(state.data.refRollIncreaseMin, LIMITS.rollInc) &&
    inRange(state.data.refRollIncreaseMax, LIMITS.rollInc) &&
    inRange(state.data.refRollDecreaseMin, LIMITS.rollDec) &&
    inRange(state.data.refRollDecreaseMax, LIMITS.rollDec) &&
    Number(state.data.refRollIncreaseMin) <= Number(state.data.refRollIncreaseMax) &&
    Number(state.data.refRollDecreaseMin) <= Number(state.data.refRollDecreaseMax);

  const incidentValid =
    templateChosen &&
    inRange(state.data.incidentCards, LIMITS.refinementCards) &&
    ['Low', 'Normal', 'High'].includes(String(state.data.incidentFrequency || 'Normal')) &&
    ['Yes', 'No'].includes(String(state.data.incidentsActive || 'Yes')) &&
    ['Yes', 'No'].includes(String(state.data.allowPlayerSpecificIncidents || 'No')) &&
    (String(state.data.allowPlayerSpecificIncidents || 'No') === 'No' || Number(state.data.playersCount) >= 2);

  // Overall: all required tabs valid (option B)
  const allTabsValid = templateChosen && basicValid && playerValid && productValid && layoutValid && refinementsValid && incidentValid;

  // Progress overview steps
  setChip('chipTemplate', templateChosen ? 'Complete' : 'To do', templateChosen ? 'chip-green' : 'chip-red');
  setStepBg('stepTemplate', templateChosen ? 'complete' : 'todo');

  setChip('chipBasic', basicValid ? 'Complete' : 'To do', basicValid ? 'chip-green' : 'chip-red');
  setDot('dotBasic', basicValid ? 'done' : 'todo');
  setStepBg('stepBasic', basicValid ? 'complete' : 'todo');

  setChip('chipLayout', layoutValid ? 'Complete' : 'To do', layoutValid ? 'chip-green' : 'chip-red');
  setDot('dotLayout', layoutValid ? 'done' : 'todo');
  setStepBg('stepLayout', layoutValid ? 'complete' : 'todo');

  const validateComplete = state.makeFinal && state.isValid && allTabsValid;
  setChip('chipValidate', validateComplete ? 'Complete' : (state.isValid ? 'Validated' : 'To do'),
    validateComplete ? 'chip-green' : (state.isValid ? 'chip-amber' : 'chip-red'));
  setDot('dotValidate', validateComplete ? 'done' : (state.isValid ? 'doing' : 'todo'));
  setStepBg('stepValidate', validateComplete ? 'complete' : (state.isValid ? '' : 'todo'));

  // Tabs: green only if tab data is valid (option B)
  setTabComplete('start', templateChosen);
  setTabComplete('basic', basicValid);
  setTabComplete('player', playerValid);
  setTabComplete('product', productValid);
  setTabComplete('layout', layoutValid);
  setTabComplete('refinements', refinementsValid);
  setTabComplete('incident', incidentValid);
  setTabComplete('validate', validateComplete);
}


  function setTabComplete(tabKey, isComplete) {
    const el = document.querySelector(`.tab[data-tab="${tabKey}"]`);
    if (!el) return;
    el.classList.toggle('is-complete', !!isComplete);
  }

  function setStepBg(stepId, stateName) {
    const el = $(stepId);
    if (!el) return;
    if (!stateName) {
      delete el.dataset.state;
      return;
    }
    el.dataset.state = stateName;
  }

  function setChip(id, text, cls) {
    const el = $(id);
    if (!el) return;
    el.textContent = text;
    el.classList.remove('chip-green', 'chip-amber', 'chip-red');
    el.classList.add(cls);
  }

  function setDot(id, stateName) {
    const el = $(id);
    if (!el) return;
    el.classList.remove('done', 'doing', 'todo');
    el.classList.add(stateName);
  }

  function inRange(v, limits) {
    if (!Number.isFinite(Number(v))) return false;
    return Number(v) >= limits.min && Number(v) <= limits.max;
  }

  function renderPlayersTable() {
    const tbody = $('playersTbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    state.players.forEach((p, idx) => {
      const tr = document.createElement('tr');

      const tdNum = tdNumEl(p.number);
      const tdName = document.createElement('td');
      const nameId = `player_name_${p.number}`;
      tdName.appendChild(inputText(nameId, p.name, (val) => {
        const old = p.name;
        p.name = val;
        stageChange(`players[${idx}].name`, old, val);
      }, LIMITS.playerName.minLen, LIMITS.playerName.maxLen));

      const tdColor = document.createElement('td');
      const colorId = `player_color_${p.number}`;
      tdColor.appendChild(inputColor(colorId, p.color, (val) => {
        const old = p.color;
        p.color = val;
        stageChange(`players[${idx}].color`, old, val);
      }));

      const tdStrategy = document.createElement('td');
      const sel = document.createElement('select');
      sel.className = 'input';
      ['scrum', 'waterfall', 'random'].forEach((opt) => {
        const o = document.createElement('option');
        o.value = opt;
        o.textContent = opt;
        if (opt === p.strategy) o.selected = true;
        sel.appendChild(o);
      });
      sel.addEventListener('change', () => {
        const old = p.strategy;
        p.strategy = sel.value;
        stageChange(`players[${idx}].strategy`, old, sel.value);
      });
      tdStrategy.appendChild(sel);

      const tdDiff = document.createElement('td');
      tdDiff.className = '';
      const diffId = `player_diff_${p.number}`;
      const inp = document.createElement('input');
      inp.id = diffId;
      inp.className = 'input';
      inp.type = 'number';
      inp.step = '0.1';
      inp.min = String(LIMITS.difficulty.min);
      inp.max = String(LIMITS.difficulty.max);
      inp.value = String(p.difficulty);
      inp.addEventListener('input', () => {
        const old = p.difficulty;
        const n = clampFloat(inp.value, LIMITS.difficulty.min, LIMITS.difficulty.max);
        if (n === null) return;
        p.difficulty = Number(n.toFixed(2));
        stageChange(`players[${idx}].difficulty`, old, p.difficulty);
      });
      tdDiff.appendChild(inp);

      tr.appendChild(tdNum);
      tr.appendChild(tdName);
      tr.appendChild(tdColor);
      tr.appendChild(tdStrategy);
      tr.appendChild(tdDiff);
      tbody.appendChild(tr);
    });
  }

  function renderProductsTable() {
    const tbody = $('productsTbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    state.products.forEach((p, idx) => {
      const tr = document.createElement('tr');

      tr.appendChild(tdNumEl(p.number));

      const tdName = document.createElement('td');
      tdName.appendChild(inputText(`product_name_${p.number}`, p.name, (val) => {
        const old = p.name;
        p.name = val;
        stageChange(`products[${idx}].name`, old, val);
        renderBoardGrid(); // update labels
        renderLayoutChanges();
      }, LIMITS.productName.minLen, LIMITS.productName.maxLen));
      tr.appendChild(tdName);

      const tdColor = document.createElement('td');
      tdColor.appendChild(inputText(`product_color_${p.number}`, p.color, (val) => {
        const old = p.color;
        p.color = val;
        stageChange(`products[${idx}].color`, old, val);
        renderBoardGrid();
      }));
      tr.appendChild(tdColor);

      const tdDesc = document.createElement('td');
      tdDesc.appendChild(inputText(`product_desc_${p.number}`, p.description, (val) => {
        const old = p.description;
        p.description = val;
        stageChange(`products[${idx}].description`, old, val);
      }, LIMITS.productDesc.minLen, LIMITS.productDesc.maxLen));
      tr.appendChild(tdDesc);

      tbody.appendChild(tr);
    });
  }

  function inputText(id, value, onChange, minLen, maxLen) {
    const inp = document.createElement('input');
    inp.id = id;
    inp.className = 'input';
    inp.type = 'text';
    inp.value = value ?? '';
    if (typeof minLen === 'number') inp.minLength = minLen;
    if (typeof maxLen === 'number') inp.maxLength = maxLen;

    inp.addEventListener('input', () => onChange(inp.value));
    inp.addEventListener('blur', () => {
      // Soft validate hex on blur for color fields
      if (id.includes('_color_')) {
        const ok = isHexColor(inp.value);
        inp.setAttribute('aria-invalid', ok ? 'false' : 'true');
      }
    });

    return inp;
  }

function inputColor(id, value, onChange) {
  const inp = document.createElement('input');
  inp.id = id;
  inp.className = 'color-input';
  inp.type = 'color';

  const initial = String(value || '#000000').trim();
  const normalized = /^#[0-9A-Fa-f]{6}$/.test(initial) ? initial : '#000000';
  inp.value = normalized;
  inp.style.backgroundColor = normalized;

  inp.addEventListener('input', () => {
    const val = String(inp.value || '#000000').toUpperCase();
    inp.style.backgroundColor = val;
    onChange(val);
  });

  return inp;
}


  function tdNumEl(n) {
    const td = document.createElement('td');
    td.className = 'num';
    td.textContent = String(n);
    return td;
  }

  function renderBoardGrid() {
    const host = $('boardGrid');
    if (!host) return;

    const productsCount = state.layout.cells.length;
    const sprintsPer = productsCount ? state.layout.cells[0].length : 0;

    // Fixed palette to match V2 screenshot.
    const palette = ['#F2C94C', '#F2994A', '#EB5757', '#27AE60', '#2D9CDB', '#9B51E0', '#828282'];

    host.innerHTML = '';

    const table = document.createElement('div');
    table.className = 'boardgrid__table';
    table.style.setProperty('--cols', String(productsCount));

    // Corner
    const corner = document.createElement('div');
    corner.className = 'boardgrid__corner';
    corner.setAttribute('aria-hidden', 'true');
    table.appendChild(corner);

    // Product headers
    for (let p = 0; p < productsCount; p++) {
      const head = document.createElement('div');
      head.className = 'phead';

      const dot = document.createElement('span');
      dot.className = 'phead__dot';
      dot.style.background = palette[p % palette.length];
      dot.setAttribute('aria-hidden', 'true');

      const txt = document.createElement('span');
      // Screenshot uses the generic naming "Product 1..N".
      txt.textContent = `Product ${p + 1}`;

      head.appendChild(dot);
      head.appendChild(txt);
      table.appendChild(head);
    }

    // Rows (sprints)
    for (let s = 0; s < sprintsPer; s++) {
      const sl = document.createElement('div');
      sl.className = 'slabel';
      sl.setAttribute('aria-hidden', 'true');
      sl.innerHTML = `<span class="slabel__top">Sprint</span><span class="slabel__num">${s + 1}</span>`;
      table.appendChild(sl);

      for (let p = 0; p < productsCount; p++) {
        const cell = state.layout.cells[p][s];
        const c = document.createElement('div');
        c.className = 'bgcell';
        c.dataset.p = String(p);
        c.dataset.s = String(s);
        c.setAttribute('role', 'button');
        c.tabIndex = 0;
        c.setAttribute('aria-selected', isSelected(p, s) ? 'true' : 'false');
        if (isSelected(p, s)) c.classList.add('bgcell--selected');

        const title = document.createElement('div');
        title.className = 'bgcell__title';
        title.textContent = `Sprint ${s + 1}`;
        c.appendChild(title);

        const line1 = document.createElement('div');
        line1.className = 'bgcell__line';
        line1.innerHTML = `<span class="bgcell__label">value:</span>`;
        const b1 = document.createElement('span');
        b1.className = 'badge';
        b1.style.background = palette[p % palette.length];
        b1.textContent = String(cell.value);
        line1.appendChild(b1);
        c.appendChild(line1);

        const line2 = document.createElement('div');
        line2.className = 'bgcell__line';
        line2.innerHTML = `<span class="bgcell__label"># features:</span>`;
        const b2 = document.createElement('span');
        b2.className = 'badge badge--sq';
        b2.style.background = palette[p % palette.length];
        b2.textContent = String(cell.features);
        line2.appendChild(b2);
        c.appendChild(line2);

        c.setAttribute('aria-label', `Product ${p + 1}, Sprint ${s + 1}. Value ${cell.value}. Features ${cell.features}.`);
        c.addEventListener('click', () => selectCell(p, s));
        c.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            selectCell(p, s);
          }
        });
        table.appendChild(c);
      }
    }

    host.appendChild(table);
  }

  function isSelected(p, s) {
    return state.selectedCell && state.selectedCell.p === p && state.selectedCell.s === s;
  }

  function selectCell(p, s) {
    state.selectedCell = { p, s };

    const prodName = state.products[p]?.name || `Product ${p + 1}`;
    if ($('cellProduct')) $('cellProduct').value = prodName;
    if ($('cellSprint')) $('cellSprint').value = String(s + 1);

    const cell = state.layout.cells[p][s];
    if ($('cellValue')) $('cellValue').value = String(cell.value);
    if ($('cellFeatures')) $('cellFeatures').value = String(cell.features);

    // Re-render to show selection ring/border on the grid.
    renderBoardGrid();

    updateProgressUI();
  }

  function applySelectedCellEdit() {
    if (!state.selectedCell) return;
    const { p, s } = state.selectedCell;
    const cell = state.layout.cells[p][s];

    const newValue = clampInt($('cellValue').value, LIMITS.layoutValue.min, LIMITS.layoutValue.max);
    const newFeatures = clampInt($('cellFeatures').value, LIMITS.layoutFeatures.min, LIMITS.layoutFeatures.max);

    if (newValue === null || newFeatures === null) return;

    const oldValue = cell.value;
    const oldFeatures = cell.features;

    if (oldValue === newValue && oldFeatures === newFeatures) return;

    cell.value = newValue;
    cell.features = newFeatures;

    const prodName = state.products[p]?.name || `Product ${p + 1}`;

    // Track layout change row
    const change = {
      id: state.layoutChanges.length + 1,
      p,
      s,
      productName: prodName,
      sprint: s + 1,
      value: newValue,
      features: newFeatures,
      oldValue,
      oldFeatures,
      user: USER.name,
      role: USER.role,
      dateTime: nowStamp(),
    };
    state.layoutChanges.push(change);

    // Stage change (for badge/patch)
    stageChange(`layout.cells[${p}][${s}]`, { value: oldValue, features: oldFeatures }, { value: newValue, features: newFeatures }, { kind: 'layout' });

    renderBoardGrid();
    renderLayoutChanges();
    updateProgressUI();
  }

  function resetSelectedCell() {
    if (!state.selectedCell) return;
    const { p, s } = state.selectedCell;
    const cell = state.layout.cells[p][s];

    $('cellValue').value = String(cell.original.value);
    $('cellFeatures').value = String(cell.original.features);

    applySelectedCellEdit();
  }

  function renderLayoutChanges() {
    const tbody = $('layoutChangesTbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    state.layoutChanges.forEach((c, idx) => {
      const tr = document.createElement('tr');

      tr.appendChild(tdNumEl(c.id));
      const tdProd = document.createElement('td');
      tdProd.textContent = c.productName;
      tr.appendChild(tdProd);

      tr.appendChild(tdNumEl(c.sprint));
      tr.appendChild(tdNumEl(c.value));
      tr.appendChild(tdNumEl(c.features));

      const tdUser = document.createElement('td');
      tdUser.textContent = c.user;
      tr.appendChild(tdUser);

      const tdRole = document.createElement('td');
      tdRole.textContent = c.role;
      tr.appendChild(tdRole);

      const tdDt = document.createElement('td');
      tdDt.textContent = c.dateTime;
      tr.appendChild(tdDt);

      const tdAct = document.createElement('td');
      tdAct.className = 'num';
      const btn = document.createElement('button');
      btn.className = 'btn';
      btn.type = 'button';
      btn.textContent = 'Delete';
      btn.addEventListener('click', () => deleteLayoutChange(idx));
      tdAct.appendChild(btn);
      tr.appendChild(tdAct);

      tbody.appendChild(tr);
    });
  }

  function deleteLayoutChange(idx) {
    const c = state.layoutChanges[idx];
    if (!c) return;

    // restore previous values for that change
    const cell = state.layout.cells[c.p][c.s];
    cell.value = c.oldValue;
    cell.features = c.oldFeatures;

    // remove change row (and keep ids stable by re-numbering)
    state.layoutChanges.splice(idx, 1);
    state.layoutChanges.forEach((x, i) => { x.id = i + 1; });

    stageChange(`layout.undo[${c.p}][${c.s}]`, { value: c.value, features: c.features }, { value: c.oldValue, features: c.oldFeatures }, { kind: 'layout-undo' });

    renderBoardGrid();
    renderLayoutChanges();
    updateProgressUI();
  }

  function runValidate(scrollToResult) {
    const errors = [];

    // Basic numeric
    mustInRange(errors, 'How many players are playing this game', state.data.playersCount, LIMITS.playersCount);
    mustLen(errors, 'Boardname from the setup', state.data.boardName, LIMITS.boardName.minLen, LIMITS.boardName.maxLen);
    mustInRange(errors, 'Number of products', state.data.productsCount, LIMITS.productsCount);
    mustInRange(errors, 'Sprints per product', state.data.sprintsPerProduct, LIMITS.sprintsPerProduct);
    mustInRange(errors, 'Game Length (Sprint Tokens per Player)', state.data.tokensPerPlayer, LIMITS.tokensPerPlayer);
    mustInRange(errors, 'Starting Money', state.data.startingMoney, LIMITS.startingMoney);
    mustInRange(errors, 'Game Length (game rounds per player)', state.data.roundsPerPlayer, LIMITS.roundsPerPlayer);

    // Money fields
    [
      ['Value of 1 Ring', state.data.ringValue],
      ['Cost to Continue', state.data.costContinue],
      ['Cost to Switch (mid)', state.data.costSwitchMid],
      ['Cost to Switch (after last)', state.data.costSwitchAfter],
      ['Mandatory Loan Amount', state.data.mandatoryLoan],
      ['Loan Interest (per turn)', state.data.loanInterest],
      ['Penalty Multiplier (negative net)', state.data.penaltyNeg],
      ['Penalty Multiplier (positive net)', state.data.penaltyPos],
    ].forEach(([label, val]) => mustInRange(errors, label, val, LIMITS.money));

    mustInRange(errors, 'Daily Scrums per Sprint', state.data.dailyScrumsPerSprint, LIMITS.dailyScrumsPerSprint);
    mustInRange(errors, 'Daily Scrum Target Number', state.data.dailyScrumTarget, LIMITS.dailyScrumTarget);

    // Players
    state.players.forEach((p, i) => {
      mustLen(errors, `Player ${i + 1} name`, p.name, LIMITS.playerName.minLen, LIMITS.playerName.maxLen);
      if (!isHexColor(p.color)) errors.push(`Player ${i + 1} color must be a hex value like #FF0000.`);
      if (!['scrum', 'waterfall', 'random'].includes(p.strategy)) errors.push(`Player ${i + 1} strategy must be scrum, waterfall, or random.`);
      mustInRange(errors, `Player ${i + 1} difficulty`, p.difficulty, LIMITS.difficulty);
    });

    // Products
    state.products.forEach((p, i) => {
      mustLen(errors, `Product ${i + 1} name`, p.name, LIMITS.productName.minLen, LIMITS.productName.maxLen);
      if (!isHexColor(p.color)) errors.push(`Product ${i + 1} color must be a hex value like #00FF00.`);
      mustLen(errors, `Product ${i + 1} description`, p.description, LIMITS.productDesc.minLen, LIMITS.productDesc.maxLen);
    });

    // Layout (prototype constraints 0–32)
    for (let p = 0; p < state.layout.cells.length; p++) {
      for (let s = 0; s < state.layout.cells[p].length; s++) {
        const c = state.layout.cells[p][s];
        if (!inRange(c.value, LIMITS.layoutValue)) errors.push(`Layout cell (Product ${p + 1}, Sprint ${s + 1}) value must be between ${LIMITS.layoutValue.min} and ${LIMITS.layoutValue.max}.`);
        if (!inRange(c.features, LIMITS.layoutFeatures)) errors.push(`Layout cell (Product ${p + 1}, Sprint ${s + 1}) features must be between ${LIMITS.layoutFeatures.min} and ${LIMITS.layoutFeatures.max}.`);
      }
    }

    // Refinements
    mustInRange(errors, 'Refinement cards', state.data.refinementCards, LIMITS.refinementCards);
    mustInRange(errors, 'roll 1 to 2 (increase) min', state.data.refRollIncreaseMin, LIMITS.rollInc);
    mustInRange(errors, 'roll 1 to 2 (increase) max', state.data.refRollIncreaseMax, LIMITS.rollInc);
    if (state.data.refRollIncreaseMin > state.data.refRollIncreaseMax) errors.push('Refinement increase range: min must be <= max.');

    mustInRange(errors, 'roll 19 to 20 (decrease) min', state.data.refRollDecreaseMin, LIMITS.rollDec);
    mustInRange(errors, 'roll 19 to 20 (decrease) max', state.data.refRollDecreaseMax, LIMITS.rollDec);
    if (state.data.refRollDecreaseMin > state.data.refRollDecreaseMax) errors.push('Refinement decrease range: min must be <= max.');

    // Incident (Spec v1.1)
    mustInRange(errors, 'Incident cards', state.data.incidentCards, LIMITS.refinementCards);
    if (!['Yes', 'No'].includes(String(state.data.incidentsActive))) errors.push('Incidents Active: must be Yes or No.');
    if (!['Low', 'Normal', 'High'].includes(String(state.data.incidentFrequency))) errors.push('Incident Frequency (Severity): must be Low, Normal, or High.');
    if (!['Yes', 'No'].includes(String(state.data.allowPlayerSpecificIncidents))) errors.push('Allow Player-Specific Incidents: must be Yes or No.');
    // Player-specific incidents only make sense with 2+ players
    if (String(state.data.allowPlayerSpecificIncidents) === 'Yes' && Number(state.data.playersCount) < 2) {
      errors.push('Allow Player-Specific Incidents: requires at least 2 players.');
    }

    // Render result
    const ok = errors.length === 0;
    renderValidation(ok, errors);
    setValidationState(ok, errors);

    if (scrollToResult) setTab('validate');

    return ok;
  }

  function runValidateWithProgress() {
    return runOperation('Validating…', 700, () => runValidate(true));
  }

  function saveWithProgress() {
    if (!state.isValid) return;
    return runOperation('Finalizing…', 900, () => saveIfValid());
  }

  function runOperation(label, durationMs, fn) {
    const box = $('opProgress');
    const bar = $('opProgressBar');
    const lab = $('opProgressLabel');
    if (!box || !bar || !lab) {
      return fn();
    }

    lab.textContent = label;
    box.hidden = false;
    bar.style.width = '0%';

    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / durationMs);
      const pct = Math.round(t * 100);
      bar.style.width = pct + '%';
      if (t < 1) {
        requestAnimationFrame(tick);
      } else {
        // small micro-pause for perceived completion
        setTimeout(() => {
          box.hidden = true;
          fn();
        }, 60);
      }
    };
    requestAnimationFrame(tick);
  }

  function renderValidation(ok, errors) {
    const result = $('validationResult');
    const list = $('validationList');
    if (!result || !list) return;

    list.innerHTML = '';

    if (ok) {
      result.textContent = 'Validation passed. You can save.';
      return;
    }

    result.textContent = 'Validation failed. Fix the following:';
    errors.forEach((e) => {
      const li = document.createElement('li');
      li.textContent = e;
      list.appendChild(li);
    });
  }

  function mustInRange(errors, label, value, limits) {
    if (!Number.isFinite(Number(value))) {
      errors.push(`${label}: must be a number.`);
      return;
    }
    if (Number(value) < limits.min || Number(value) > limits.max) {
      errors.push(`${label}: must be between ${limits.min} and ${limits.max}.`);
    }
  }

  function mustLen(errors, label, value, minLen, maxLen) {
    const len = String(value ?? '').trim().length;
    if (len < minLen || len > maxLen) {
      errors.push(`${label}: must be ${minLen}–${maxLen} characters.`);
    }
  }

  function saveIfValid() {
    if (!state.isValid) return;
    state.makeFinal = true;
    updateProgressUI();
    alert('Make final (prototype). In the real app this would persist to the backend.');
  }

  function downloadJson() {
    const blob = new Blob([JSON.stringify(exportConfig(), null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gameconfig_prototype_v9.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function exportConfig() {
    return {
      template: { key: state.data.templateKey, name: state.data.templateName },
      boardId: state.boardId,
      basic: {
        playersCount: state.data.playersCount,
        boardName: state.data.boardName,
        productsCount: state.data.productsCount,
        sprintsPerProduct: state.data.sprintsPerProduct,
        tokensPerPlayer: state.data.tokensPerPlayer,
        startingMoney: state.data.startingMoney,
        roundsPerPlayer: state.data.roundsPerPlayer,
        costs: {
          ringValue: state.data.ringValue,
          costContinue: state.data.costContinue,
          costSwitchMid: state.data.costSwitchMid,
          costSwitchAfter: state.data.costSwitchAfter,
          mandatoryLoan: state.data.mandatoryLoan,
          loanInterest: state.data.loanInterest,
          penaltyNeg: state.data.penaltyNeg,
          penaltyPos: state.data.penaltyPos,
        },
        scrum: {
          dailyScrumsPerSprint: state.data.dailyScrumsPerSprint,
          dailyScrumTarget: state.data.dailyScrumTarget,
        },
      },
      players: state.players,
      products: state.products,
      layout: {
        products: state.data.productsCount,
        sprintsPerProduct: state.data.sprintsPerProduct,
        cells: state.layout.cells.map((row) => row.map((c) => ({ value: c.value, features: c.features }))),
      },
      refinements: {
        refinementCards: state.data.refinementCards,
        refinementsActive: state.data.refinementsActive,
        refinementModel: state.data.refinementModel,
        increaseRange: [state.data.refRollIncreaseMin, state.data.refRollIncreaseMax],
        decreaseRange: [state.data.refRollDecreaseMin, state.data.refRollDecreaseMax],
      },
      incident: {
        // disabled in prototype, still included for completeness
        incidentCards: state.data.incidentCards,
        incidentsActive: state.data.incidentsActive,
        incidentFrequency: state.data.incidentFrequency,
        allowPlayerSpecificIncidents: state.data.allowPlayerSpecificIncidents,
        enabledInUI: false,
      },
      staged: state.staged,
      prototype: { version: 9, timestamp: nowStamp() },
    };
  }

  async function copyPatchToClipboard() {
    const payload = {
      stagedCount: state.staged.length,
      staged: state.staged,
    };
    const txt = JSON.stringify(payload, null, 2);
    try {
      await navigator.clipboard.writeText(txt);
      alert('Staged patch JSON copied to clipboard.');
    } catch {
      // Fallback
      const ta = document.createElement('textarea');
      ta.value = txt;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
      alert('Staged patch JSON copied (fallback).');
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;',
    }[c]));
  }

  // boot
  document.addEventListener('DOMContentLoaded', init);
})();
