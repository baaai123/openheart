// config_renderer.js — Control panel renderer logic
// v4.5.0 §7.3.4 — Runs in config window (Electron renderer, browser context)
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

    byId('ws-host').value = cfg.wsHost || 'localhost';
    byId('ws-port').value = cfg.wsPort || 9876;
    byId('disp-width').value = cfg.width || 500;
    byId('disp-height').value = cfg.height || 900;
    byId('model-path').value = cfg.modelPath || './models/xiaoyue/xiaoyue.model3.json';

    setToggle('tog-atop', !!cfg.alwaysOnTop);
    setToggle('tog-clickthru', !!cfg.clickThrough);
    setToggle('tog-devtools', !!cfg.devTools);
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

  // ---- Wire up inputs ----

  function initInputs() {
    // Connection inputs
    byId('ws-host').addEventListener('input', function () {
      debouncedSave('wsHost', this.value);
    });
    byId('ws-port').addEventListener('input', function () {
      debouncedSave('wsPort', parseInt(this.value, 10) || 9876);
    });

    // Display inputs
    byId('disp-width').addEventListener('input', function () {
      debouncedSave('width', parseInt(this.value, 10) || 500);
    });
    byId('disp-height').addEventListener('input', function () {
      debouncedSave('height', parseInt(this.value, 10) || 900);
    });

    // Toggle switches
    byId('tog-atop').addEventListener('click', function () {
      this.classList.toggle('active');
      saveField('alwaysOnTop', this.classList.contains('active'));
    });
    byId('tog-clickthru').addEventListener('click', function () {
      this.classList.toggle('active');
      saveField('clickThrough', this.classList.contains('active'));
    });
    byId('tog-devtools').addEventListener('click', function () {
      this.classList.toggle('active');
      saveField('devTools', this.classList.contains('active'));
    });
  }

  // ---- Expression buttons ----

  function initExpressionButtons() {
    document.querySelectorAll('.expr-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var name = this.getAttribute('data-expr');
        if (!name) return;
        api.sendExpression(name).catch(function (err) {
          console.warn('[config] sendExpression failed:', err);
        });
        toast('Expression: ' + name);
      });
    });
  }

  // ---- Reconnect button ----

  function initReconnectButton() {
    byId('btn-reconnect').addEventListener('click', function () {
      var host = byId('ws-host').value.trim() || 'localhost';
      var port = parseInt(byId('ws-port').value, 10) || 9876;
      saveField('wsHost', host);
      saveField('wsPort', port);
      api.reconnectWs().catch(function (err) {
        console.warn('[config] reconnectWs failed:', err);
      });
      toast('Reconnecting to ' + host + ':' + port);
    });
  }

  // ---- Init ----

  function init() {
    loadSettings();
    initInputs();
    initExpressionButtons();
    initReconnectButton();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
