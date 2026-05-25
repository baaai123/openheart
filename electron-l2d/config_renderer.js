// config_renderer.js — OpenHeart backend config panel renderer
// v4.5.0 §13 — Runs in config window (Electron renderer, browser context)
// IPC via window.electronAPI (exposed by preload.js with contextIsolation:true)

(function () {
  'use strict';

  const api = window.electronAPI;
  if (!api) {
    document.body.innerHTML = '<div style="padding:20px;color:var(--danger)">Error: electronAPI not available — preload.js missing?</div>';
    return;
  }

  // ---- Helpers ----

  function byId(id) { return document.getElementById(id); }

  function toast(msg) {
    const el = byId('toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(el._hide);
    el._hide = setTimeout(() => el.classList.remove('show'), 2000);
  }

  // ---- Toggle helpers ----

  function setToggle(id, active) {
    const el = byId(id);
    if (!el) return;
    el.classList.toggle('active', active);
  }

  function isToggleActive(id) {
    const el = byId(id);
    return el ? el.classList.contains('active') : false;
  }

  // ---- Save a single config key ----

  function saveField(key, value) {
    api.saveConfig({ key: key, value: value }).catch(function (err) {
      console.warn('[config] saveConfig failed:', err);
    });
  }

  // ---- Debounced save for text/number inputs ----

  const debounceTimers = {};

  function debouncedSave(key, value, delayMs) {
    delayMs = delayMs || 400;
    if (debounceTimers[key]) clearTimeout(debounceTimers[key]);
    debounceTimers[key] = setTimeout(function () {
      saveField(key, value);
      toast(key + ' saved');
    }, delayMs);
  }

  // ---- Populate fields from config ----

  async function loadSettings() {
    let cfg;
    try {
      cfg = await api.loadConfig();
    } catch (err) {
      console.warn('[config] loadConfig failed:', err);
      toast('Failed to load settings');
      return;
    }
    if (!cfg) return;

    // API Config
    byId('api-baseurl').value = cfg.baseUrl || '';
    byId('api-model').value = cfg.model || '';
    byId('api-key').value = cfg.apiKey || '';

    // Persona
    byId('persona-prompt').value = cfg.systemPrompt || '';

    // Module switches (default on)
    setToggle('tog-voice', cfg.voiceEnabled !== false);
    setToggle('tog-visual', cfg.visualEnabled !== false);
    setToggle('tog-l2d', cfg.l2dEnabled !== false);
  }

  // ---- Progress bar ----

  function updateProgress(text, percent) {
    const fill = byId('progress-fill');
    const label = byId('progress-text');
    if (fill) fill.style.width = Math.min(100, Math.max(0, percent)) + '%';
    if (label) label.textContent = text;
  }

  // ---- Status dots ----

  function setDotColor(id, color) {
    const el = byId(id);
    if (!el) return;
    el.className = 'status-dot ' + color;
  }

  function updateBackendStatus(state) {
    const textEl = byId('status-backend-text');
    if (!textEl) return;
    switch (state) {
      case 'running':
        setDotColor('status-backend-dot', 'green');
        textEl.textContent = 'Running';
        break;
      case 'starting':
        setDotColor('status-backend-dot', 'yellow');
        textEl.textContent = 'Starting...';
        break;
      default:
        setDotColor('status-backend-dot', 'red');
        textEl.textContent = 'Offline';
    }
  }

  function updateL2DStatus(state) {
    const textEl = byId('status-l2d-text');
    if (!textEl) return;
    switch (state) {
      case 'running':
        setDotColor('status-l2d-dot', 'green');
        textEl.textContent = 'Running';
        break;
      case 'starting':
        setDotColor('status-l2d-dot', 'yellow');
        textEl.textContent = 'Starting...';
        break;
      default:
        setDotColor('status-l2d-dot', 'red');
        textEl.textContent = 'Offline';
    }
  }

  // ---- Wire up inputs ----

  function initInputs() {
    // API Config
    byId('api-baseurl').addEventListener('input', function () {
      debouncedSave('baseUrl', this.value);
    });
    byId('api-model').addEventListener('input', function () {
      debouncedSave('model', this.value);
    });
    byId('api-key').addEventListener('input', function () {
      debouncedSave('apiKey', this.value);
    });

    // Persona
    byId('persona-prompt').addEventListener('input', function () {
      debouncedSave('systemPrompt', this.value);
    });

    // Toggle switches
    byId('tog-voice').addEventListener('click', function () {
      this.classList.toggle('active');
      saveField('voiceEnabled', this.classList.contains('active'));
    });
    byId('tog-visual').addEventListener('click', function () {
      this.classList.toggle('active');
      saveField('visualEnabled', this.classList.contains('active'));
    });
    byId('tog-l2d').addEventListener('click', function () {
      this.classList.toggle('active');
      saveField('l2dEnabled', this.classList.contains('active'));
    });
  }

  // ---- Start Backend button ----

  function initStartButton() {
    byId('btn-start').addEventListener('click', function () {
      const btn = this;
      btn.disabled = true;
      btn.textContent = '⟳ Starting...';
      updateProgress('Initializing...', 10);

      api.startBackend().then(function (result) {
        toast('Backend started');
        updateProgress('Running', 100);
        updateBackendStatus('running');
        updateL2DStatus('running');
        btn.textContent = '✓ BACKEND RUNNING';
      }).catch(function (err) {
        console.warn('[config] startBackend failed:', err);
        toast('Failed to start backend');
        updateProgress('Failed', 0);
        updateBackendStatus('stopped');
        updateL2DStatus('stopped');
        btn.disabled = false;
        btn.textContent = '▶ START BACKEND';
      });
    });
  }

  // ---- IPC status / progress listeners ----

  function initStatusListeners() {
    if (api.onBackendStatus) {
      api.onBackendStatus(function (state) {
        updateBackendStatus(state);
      });
    }
    if (api.onL2DStatus) {
      api.onL2DStatus(function (state) {
        updateL2DStatus(state);
      });
    }
    if (api.onBackendProgress) {
      api.onBackendProgress(function (data) {
        updateProgress(data.text || '', data.percent || 0);
      });
    }
  }

  // ---- Init ----

  function init() {
    loadSettings();
    initInputs();
    initStartButton();
    initStatusListeners();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
