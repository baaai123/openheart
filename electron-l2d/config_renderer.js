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

  // ---- Constants ----
  const BACKEND_API_BASE = 'http://localhost:8081';
  const STATUS_POLL_INTERVAL = 3000; // ms
  const LOCAL_STORAGE_KEY = 'openheart_config';

  // ---- Helpers ----

  function byId(id) { return document.getElementById(id); }

  function toast(msg) {
    const el = byId('toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(el._hide);
    el._hide = setTimeout(function () { el.classList.remove('show'); }, 2000);
  }

  // ---- Toggle helpers ----

  function setToggle(id, active) {
    var el = byId(id);
    if (!el) return;
    el.classList.toggle('active', active);
  }

  function isToggleActive(id) {
    var el = byId(id);
    return el ? el.classList.contains('active') : false;
  }

  // ---- HTTP helpers ----

  function apiUrl(path) {
    return BACKEND_API_BASE + path;
  }

  async function apiFetch(path, options) {
    var url = apiUrl(path);
    var opts = options || {};
    opts.headers = opts.headers || {};
    opts.headers['Content-Type'] = 'application/json';
    var res = await fetch(url, opts);
    if (!res.ok) {
      var body;
      try { body = await res.text(); } catch (_) { body = ''; }
      throw new Error('API ' + res.status + ' ' + res.statusText + (body ? ': ' + body : ''));
    }
    // 204 No Content → no body
    if (res.status === 204) return null;
    return res.json();
  }

  // ---- Build full config object from UI ----

  function collectConfig() {
    var voiceModeEl = byId('voice-mode');
    return {
      baseUrl: byId('api-baseurl').value || '',
      model: byId('api-model').value || '',
      apiKey: byId('api-key').value || '',
      systemPrompt: byId('persona-prompt').value || '',
      voiceEnabled: isToggleActive('tog-voice'),
      visualEnabled: isToggleActive('tog-visual'),
      l2dEnabled: isToggleActive('tog-l2d'),
      voiceMode: voiceModeEl ? voiceModeEl.value : 'asr'
    };
  }

  // ---- Populate fields from localStorage cache + GET /api/config ----

  function applyConfig(cfg) {
    if (!cfg) return;
    byId('api-baseurl').value = cfg.baseUrl || '';
    byId('api-model').value = cfg.model || '';
    byId('api-key').value = cfg.apiKey || '';
    byId('persona-prompt').value = cfg.systemPrompt || '';
    setToggle('tog-voice', cfg.voiceEnabled !== false);
    setToggle('tog-visual', cfg.visualEnabled !== false);
    setToggle('tog-l2d', cfg.l2dEnabled !== false);
    var voiceModeEl = byId('voice-mode');
    if (voiceModeEl) {
      voiceModeEl.value = cfg.voiceMode === 'text' ? 'text' : 'asr';
    }
  }

  async function loadSettings() {
    // (1) Load cached config from localStorage immediately (fast, works offline)
    var localCfg = loadFromLocalStorage();
    if (localCfg) {
      applyConfig(localCfg);
    }

    // (2) Try API for fresh config — overwrites localStorage values if successful
    try {
      var cfg = await apiFetch('/api/config');
      if (cfg) {
        applyConfig(cfg);
        saveToLocalStorage(cfg);
        return;
      }
    } catch (err) {
      // Backend unreachable — keep localStorage values if available
      if (!localCfg) {
        toast('Backend unreachable — using defaults');
      }
    }
  }

  // ---- Save full config with fallback chain: API → Electron IPC → localStorage ----

  var _saveInFlight = false;

  async function saveConfigToBackend() {
    if (_saveInFlight) return; // prevent concurrent saves
    _saveInFlight = true;
    var config = collectConfig();
    try {
      // (1) Try POST to API first
      await apiFetch('/api/config', { method: 'POST', body: JSON.stringify(config) });
      // Also save to localStorage so page load works when backend is down
      saveToLocalStorage(config);
      toast('Saved to backend');
    } catch (_apiErr) {
      // API failed (backend not running) — fall back to Electron IPC
      console.warn('[config] POST /api/config failed, trying IPC fallback:', _apiErr);
      try {
        if (api && api.saveConfig) {
          await api.saveConfig(config);
          saveToLocalStorage(config);
          toast('Saved locally (backend not running)');
        } else {
          throw new Error('electronAPI.saveConfig not available');
        }
      } catch (_ipcErr) {
        // IPC also failed — last resort: localStorage only
        console.warn('[config] IPC saveConfig failed, using localStorage:', _ipcErr);
        saveToLocalStorage(config);
        toast('Saved locally (backend not running)');
      }
    } finally {
      _saveInFlight = false;
    }
  }

  // ---- localStorage helpers ----

  function saveToLocalStorage(config) {
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(config));
    } catch (_e) {
      // localStorage full or disabled — non-critical, swallow silently
      console.warn('[config] localStorage write failed:', _e);
    }
  }

  function loadFromLocalStorage() {
    try {
      var raw = localStorage.getItem(LOCAL_STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (_e) {
      // Corrupted data or localStorage disabled — ignore
      console.warn('[config] localStorage read failed:', _e);
      return null;
    }
  }

  // ---- Debounced full config save ----

  var _debounceTimer = null;

  function debouncedSaveToBackend() {
    if (_debounceTimer) clearTimeout(_debounceTimer);
    _debounceTimer = setTimeout(function () {
      saveConfigToBackend();
    }, 400);
  }

  // ---- Progress bar ----

  function updateProgress(text, percent) {
    var fill = byId('progress-fill');
    var label = byId('progress-text');
    if (fill) fill.style.width = Math.min(100, Math.max(0, percent)) + '%';
    if (label) label.textContent = text;
  }

  // ---- Status dots ----

  function setDotColor(id, color) {
    var el = byId(id);
    if (!el) return;
    el.className = 'status-dot ' + color;
  }

  function updateBackendStatus(state) {
    var textEl = byId('status-backend-text');
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
    var textEl = byId('status-l2d-text');
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

  // ---- Poll GET /api/status every 3s ----

  var _statusPollTimer = null;

  async function pollBackendStatus() {
    try {
      var status = await apiFetch('/api/status');

      // Backend status — accept 'backend' or 'status' field
      var backendState = status.backend || status.status || 'offline';
      updateBackendStatus(backendState);

      // L2D status — accept 'l2d' or 'l2d_status' field
      var l2dState = status.l2d || status.l2d_status;
      if (l2dState !== undefined) {
        updateL2DStatus(l2dState);
      }
    } catch (err) {
      // Backend unreachable → both offline
      updateBackendStatus('offline');
      updateL2DStatus('offline');
    }
  }

  function startStatusPolling() {
    pollBackendStatus(); // immediate first poll
    _statusPollTimer = setInterval(pollBackendStatus, STATUS_POLL_INTERVAL);
  }

  function stopStatusPolling() {
    if (_statusPollTimer) {
      clearInterval(_statusPollTimer);
      _statusPollTimer = null;
    }
  }

  // ---- Wire up inputs ----

  function initInputs() {
    // API Config — debounced save on input
    byId('api-baseurl').addEventListener('input', debouncedSaveToBackend);
    byId('api-model').addEventListener('input', debouncedSaveToBackend);
    byId('api-key').addEventListener('input', debouncedSaveToBackend);

    // Persona
    byId('persona-prompt').addEventListener('input', debouncedSaveToBackend);

    // Toggle switches — immediate save on click
    byId('tog-voice').addEventListener('click', function () {
      this.classList.toggle('active');
      saveConfigToBackend();
    });
    byId('tog-visual').addEventListener('click', function () {
      this.classList.toggle('active');
      saveConfigToBackend();
    });
    byId('tog-l2d').addEventListener('click', function () {
      this.classList.toggle('active');
      saveConfigToBackend();
    });

    // Voice mode switch — save on change
    var voiceModeEl = byId('voice-mode');
    if (voiceModeEl) {
      voiceModeEl.addEventListener('change', function () {
        saveConfigToBackend();
      });
    }
  }

  // ---- Start Backend button ----
  // v4.5.0 §13: POST config first, then start backend via IPC

  function initStartButton() {
    byId('btn-start').addEventListener('click', async function () {
      var btn = this;
      btn.disabled = true;
      btn.textContent = '⏳ Saving config...';

      // 1. POST current config to backend
      await saveConfigToBackend();

      // 2. Start backend via IPC
      btn.textContent = '⏳ Starting...';
      try {
        await api.startBackend();
        toast('Backend starting...');
      } catch (err) {
        console.warn('[config] startBackend failed:', err);
        toast('Failed to start backend');
      } finally {
        btn.disabled = false;
        btn.textContent = '▶ START L2D';
      }
    });
  }

  // ---- Init ----

  function init() {
    loadSettings();
    initInputs();
    initStartButton();
    startStatusPolling();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
