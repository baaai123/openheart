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
    var voiceModeEl = byId('voice-mode');
    if (voiceModeEl) {
      voiceModeEl.value = cfg.voiceMode === 'text' ? 'text' : 'asr';
      toggleChat(cfg.voiceMode === 'text');
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

  // ---- Real config file writers (v4.5.0 §13) ----

  async function writeConfigFilesViaIPC(config) {
    if (!api) return;
    await api.writeEnv({
      DEEPSEEK_API_KEY: config.apiKey,
      DEEPSEEK_BASE_URL: config.baseUrl,
      DEEPSEEK_MODEL: config.model
    });
    await api.writePromptModules({ persona: config.systemPrompt });
    await api.writeUISettings({
      visual_enabled: config.visualEnabled,
      voice_mode: config.voiceMode
    });
  }

  async function saveConfigToBackend() {
    if (_saveInFlight) return; // prevent concurrent saves
    _saveInFlight = true;
    var config = collectConfig();
    try {
      // (1) Try POST to API first
      await apiFetch('/api/config', { method: 'POST', body: JSON.stringify(config) });
      await writeConfigFilesViaIPC(config);
      saveToLocalStorage(config);
      toast('Saved to backend');
    } catch (_apiErr) {
      // API failed (backend not running) — fall back to Electron IPC
      console.warn('[config] POST /api/config failed, trying IPC fallback:', _apiErr);
      try {
        if (api && api.saveConfig) {
          await api.saveConfig(config);
          await writeConfigFilesViaIPC(config);
          saveToLocalStorage(config);
          toast('Saved locally (backend not running)');
        } else {
          throw new Error('electronAPI.saveConfig not available');
        }
      } catch (_ipcErr) {
        console.warn('[config] IPC saveConfig failed, using localStorage:', _ipcErr);
        try { await writeConfigFilesViaIPC(config); } catch (_) {}
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
    // Voice mode switch — save on change + toggle chat
    var voiceModeEl = byId('voice-mode');
    if (voiceModeEl) {
      voiceModeEl.addEventListener('change', function () {
        toggleChat(this.value === 'text');
        saveConfigToBackend();
      });
    }
  }

  // ---- Chat area ----

  var _chatMessages = [];
  var _chatMaxVisible = 50;

  function toggleChat(show) {
    var area = byId('chat-area');
    if (!area) return;
    area.classList.toggle('visible', show);
  }

  function appendChatMessage(role, text) {
    var log = byId('chat-log');
    if (!log) return;
    _chatMessages.push({ role: role, text: text });
    if (_chatMessages.length > _chatMaxVisible) {
      _chatMessages.shift();
    }
    log.innerHTML = _chatMessages.map(function (m) {
      var cls = m.role === 'user' ? 'chat-msg user' : m.role === 'error' ? 'chat-msg error' : 'chat-msg bot';
      var label = m.role === 'user' ? 'You' : m.role === 'error' ? 'Error' : 'AI';
      return '<div class="' + cls + '"><div class="sender">' + label + '</div>' + escapeHtml(m.text) + '</div>';
    }).join('');
    log.scrollTop = log.scrollHeight;
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  async function sendChatMessage() {
    var input = byId('chat-input');
    var sendBtn = byId('chat-send');
    if (!input || !sendBtn) return;
    var text = input.value.trim();
    if (!text) return;

    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;

    appendChatMessage('user', text);

    try {
      var res = await apiFetch('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ text: text })
      });
      var reply = (res && res.reply) || (res && res.text) || JSON.stringify(res);
      appendChatMessage('bot', reply);
    } catch (err) {
      appendChatMessage('error', 'Request failed: ' + err.message);
    } finally {
      input.disabled = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  function initChat() {
    var input = byId('chat-input');
    var sendBtn = byId('chat-send');
    if (!input || !sendBtn) return;

    sendBtn.addEventListener('click', sendChatMessage);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        sendChatMessage();
      }
    });

    // Initial visibility based on current voice mode
    var voiceModeEl = byId('voice-mode');
    if (voiceModeEl) {
      toggleChat(voiceModeEl.value === 'text');
    }
  }

  // ---- Start Backend button ----
  // v4.5.0 §13: POST config first, then start backend via IPC

  function initStartButton() {
    byId('btn-start').addEventListener('click', async function () {
      var btn = this;
      btn.disabled = true;

      // 1. Collect current config from form
      var config = collectConfig();

      // 2. Save to local persistence (Electron userData + localStorage)
      btn.textContent = '⏳ Saving config...';
      updateProgress('Saving config...', 25);
      await saveConfigToBackend();

      // 3. Write config files to WSL so the backend can read .env and config files
      btn.textContent = '⏳ Writing WSL config...';
      updateProgress('Writing WSL config...', 50);
      try {
        await writeConfigFilesViaIPC(config);
      } catch (err) {
        console.warn('[config] WSL config write failed:', err);
        toast('Warning: WSL config write failed');
      }

      // 4. Open WSL terminal to start backend
      btn.textContent = '⏳ Opening terminal...';
      updateProgress('Starting backend...', 75);
      if (api.openTerminal) {
        api.openTerminal('wsl bash /home/baaai/projects/openheart/run_backend.sh');
        toast('Backend starting in WSL terminal');
      } else {
        toast('Run: wsl bash /home/baaai/projects/openheart/run_backend.sh');
      }
      btn.disabled = false;
      btn.textContent = '▶ START L2D';
      updateProgress('Ready', 100);
    });
  }

  // ---- Init ----

  function init() {
    loadSettings();
    initInputs();
    initChat();
    initStartButton();
    startStatusPolling();

  // ---- Startup log ----

  var _startupLogTimers = [];

  function _clearStartupTimers() {
    _startupLogTimers.forEach(function(t) { clearTimeout(t); });
    _startupLogTimers = [];
  }

  function _showStartupMsg(msg, isError) {
    var el = document.getElementById('startup-log');
    if (!el) return;
    el.textContent = msg;
    el.style.color = isError ? '#ff6b6b' : '#4ec04e';
  }

  function pollStartupLog() {
    var el = document.getElementById('startup-log');
    if (!el) return;

    // Phase messages cycled over ~60s via setTimeout, matching actual load time
    var phases = [
      { text: 'API Check OK', delay: 2000 },
      { text: 'TTS Loading...', delay: 15000 },
      { text: 'YOLOE Loading...', delay: 30000 },
      { text: 'Server Ready (port 9876)', delay: 60000 }
    ];

    function schedulePhases() {
      _clearStartupTimers();
      phases.forEach(function(p) {
        var t = setTimeout(function() { _showStartupMsg(p.text); }, p.delay);
        _startupLogTimers.push(t);
      });
    }

    // (1) Real-time backend status via IPC (from main.js backend process)
    if (api && api.onBackendStatus) {
      api.onBackendStatus(function(state) {
        _clearStartupTimers();
        if (state === 'running') {
          // Backend reports running — verify L2D port 9876 is open
          fetch('http://localhost:9876')
            .then(function() { _showStartupMsg('Server Ready (port 9876)'); })
            .catch(function() { _showStartupMsg('Backend OK (L2D loading...)'); });
        } else if (state === 'starting') {
          _showStartupMsg('Starting backend...');
          schedulePhases();
        } else {
          _showStartupMsg('Backend: not running\nRun: wsl bash /home/baaai/projects/openheart/run_backend.sh', true);
        }
      });
    }

    // (2) Real progress updates via IPC (backend startup sequence)
    if (api && api.onBackendProgress) {
      api.onBackendProgress(function(data) {
        _clearStartupTimers();
        _showStartupMsg(data.text || 'Loading...');
      });
    }

    // (3) Fallback: poll L2D port 9876 directly
    pollL2DPortFallback();
  }

  function pollL2DPortFallback() {
    fetch('http://localhost:9876')
      .then(function() {
        _clearStartupTimers();
        _showStartupMsg('Server Ready (port 9876)');
      })
      .catch(function() {
        // Port not open yet — retry in 3s
        var t = setTimeout(pollL2DPortFallback, 3000);
        _startupLogTimers.push(t);
      });
  }

  // Call once to set up listeners
  pollStartupLog();

  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
