// renderer.js — Live2D xiaoyue model rendering with pixi-live2d-display
// v4.5.0 §7.3.4 — Runs in Electron renderer (browser context, NOT Node.js)
// Dependencies loaded via <script> tags: live2dcubismcore, pixi.js, pixi-live2d-display

(function () {
  'use strict';

  // ---- Guards ----
  var api = window.electronAPI;
  if (!api) {
    console.error('[l2d] electronAPI not available — preload.js missing?');
    showPlaceholder('IPC 桥接未加载');
    return;
  }

  if (typeof PIXI === 'undefined') {
    console.error('[l2d] PIXI not loaded');
    showPlaceholder('PIXI 未加载');
    return;
  }

  if (!PIXI.live2d || !PIXI.live2d.Live2DModel) {
    console.error('[l2d] pixi-live2d-display not loaded');
    showPlaceholder('Live2D 插件未加载');
    return;
  }

  var Live2DModel = PIXI.live2d.Live2DModel;

  // ---- Placeholder management ----
  function showPlaceholder(msg) {
    var el = document.getElementById('placeholder');
    if (el) {
      el.style.display = 'block';
      el.textContent = msg;
    }
  }

  function hidePlaceholder() {
    var el = document.getElementById('placeholder');
    if (el) el.style.display = 'none';
  }

  // ---- Model path (relative to index.html; requires webSecurity: false for file://) ----
  var MODEL_PATH = './models/xiaoyue/xiaoyue.model3.json';

  // ---- Expression name mapping (Chinese → parameter IDs from xiaoyue.cdi3.json) ----
  var PARAM_EXPRESSIONS = {
    '丸子头': 'AJWZT',
    '前倾':   'AJQQ',
    '加载中': 'AJJZZ',
    '右手势1': 'AJYSS1',
    '左手势1': 'AJZSS1',
    '星星眼': 'AJXX',
    '晕晕眼': 'AJYYY',
    '爱心眼': 'AJAX',
    '眼镜':   'AJYJ1',
    '精灵耳': 'AJJLE',
    '脸红':   'AJLH',
    '黑脸':   'AJHL',
    'smile':   'AJAX',
    'happy':   'AJXX',
    'sad':     'AJYYY',
    'neutral': null,
    'surprised': 'AJYYY',
    'blush':   'AJLH',
  };

  var activeParams = {};

  // ---- Global cursor tracking ----
  var USE_GLOBAL_TRACKING = true;
  var _globalCursorData = null;
  var _globalCursorTs = 0;
  var _GLOBAL_CURSOR_STALE_MS = 300;
  var _FAKE_MOUSEMOVE_INTERVAL_MS = 33;

  // ---- Init PIXI Application ----
  var canvas = document.getElementById('l2d');
  if (!canvas) {
    console.error('[l2d] Canvas #l2d not found');
    return;
  }

  var app = new PIXI.Application({
    view: canvas,
    width: 500,
    height: 900,
    backgroundAlpha: 0,
    antialias: true,
    resolution: window.devicePixelRatio || 1,
    autoDensity: true,
  });

  // ---- Load model ----
  var model = null;

  console.log('[l2d] Loading model from:', MODEL_PATH);

  Live2DModel.from(MODEL_PATH).then(function (loadedModel) {
    model = loadedModel;

    // v4.5.0 §7.3.4 — Scale and position model
    model.scale.set(0.08);
    model.x = 200;
    model.y = 480;
    if (model.anchor) model.anchor.set(0.5, 0.5);

    app.stage.addChild(model);
    hidePlaceholder();
    console.log('[l2d] Model loaded successfully');

    // ---- Auto-blink: close then open eyes every 3-7 seconds ----
    // v4.5.0 §7.3.4 — Standard Cubism ParamEyeLOpen/ParamEyeROpen
    var blinkTimer = null;
    (function startBlinking() {
      if (!model || !model.internalModel) {
        setTimeout(startBlinking, 1000);
        return;
      }
      var cm = model.internalModel.coreModel;
      var delay = 3000 + Math.random() * 4000;
      blinkTimer = setTimeout(function () {
        try { cm.setParameterValueById('ParamEyeLOpen', 0); cm.update(); } catch (e) { /* blink close L — best-effort */ }
        try { cm.setParameterValueById('ParamEyeROpen', 0); cm.update(); } catch (e) { /* blink close R — best-effort */ }
        console.debug('[blink] eyes closed');
        setTimeout(function () {
          try { cm.setParameterValueById('ParamEyeLOpen', 1); cm.update(); } catch (e) { /* blink open L — best-effort */ }
          try { cm.setParameterValueById('ParamEyeROpen', 1); cm.update(); } catch (e) { /* blink open R — best-effort */ }
          console.debug('[blink] eyes opened');
          startBlinking();
        }, 100);
      }, delay);
    })();

    // Cleanup: clear blinkTimer on window close
    window.addEventListener('beforeunload', function () {
      if (blinkTimer) clearTimeout(blinkTimer);
    });

    // ---- Auto-breathing: subtle body rise/fall using ParamBreath ----
    // v4.5.0 §7.3.4 — Sinusoidal ~0.8 Hz, ~15 FPS, self-retry if model not ready
    (function startBreathing() {
      if (!model || !model.internalModel) {
        setTimeout(startBreathing, 1000);
        return;
      }
      var cm = model.internalModel.coreModel;
      var t = Date.now() / 1000;
      var value = 0.5 + 0.5 * Math.sin(t * 0.8);
      try {
        cm.setParameterValueById('ParamBreath', value);
        cm.update();
      } catch (e) { /* breathing — best-effort */ }
      setTimeout(startBreathing, 66);
    })();

    // ---- Enable global cursor tracking for eye parameters ----
    if (USE_GLOBAL_TRACKING && api.onGlobalCursor) {
      api.onGlobalCursor(function (pos) {
        _globalCursorData = pos;
        _globalCursorTs = performance.now();
      });

      app.ticker.add(function () {
        if (!model || !model.internalModel) return;
        var coreModel = model.internalModel.coreModel;
        if (!coreModel) return;

        if (_globalCursorData && (performance.now() - _globalCursorTs) < _GLOBAL_CURSOR_STALE_MS) {
          coreModel.setParameterValueById('ParamEyeBallX', _globalCursorData.nx);
          coreModel.setParameterValueById('ParamEyeBallY', (_globalCursorData.ny - 0.5) * 3);
          coreModel.setParameterValueById('ParamAngleX', (_globalCursorData.nx - 0.5) * 60);
          coreModel.setParameterValueById('ParamAngleY', (_globalCursorData.ny - 0.5) * -60);
          coreModel.setParameterValueById('ParamAngleZ', (_globalCursorData.nx - 0.5) * 20);
          coreModel.setParameterValueById('ParamBodyAngleX', (_globalCursorData.nx - 0.5) * 15);
        }
      });

      // Fallback: dispatch fake mousemove events on canvas using local coordinates from IPC
      setInterval(function () {
        if (!_globalCursorData || (performance.now() - _globalCursorTs) > _GLOBAL_CURSOR_STALE_MS) {
          return;
        }
        if (_globalCursorData.lx >= 0 && _globalCursorData.ly >= 0) {
          try {
            var evt = new MouseEvent('mousemove', {
              clientX: _globalCursorData.lx,
              clientY: _globalCursorData.ly,
              bubbles: true,
              cancelable: true,
              view: window
            });
            canvas.dispatchEvent(evt);
          } catch (e) { /* ignore dispatching errors on teardown */ }
        }
      }, _FAKE_MOUSEMOVE_INTERVAL_MS);

      console.log('[l2d] Global cursor tracking enabled');
    }

    // ---- Register IPC handlers ----

    window.applyExpression = applyExpression;
    function applyExpression(expName) {
      if (!model || !model.internalModel) {
        console.warn('[l2d] Model not ready for expression:', expName);
        return;
      }

      var coreModel = model.internalModel.coreModel;

      for (var pid in activeParams) {
        if (activeParams.hasOwnProperty(pid)) {
          coreModel.setParameterValueById(pid, 0);
          clearTimeout(activeParams[pid]);
        }
      }
      activeParams = {};

      if (expName === 'neutral') {
        console.log('[l2d] Expression: neutral (reset)');
        coreModel.update();
        return;
      }

      var paramId = PARAM_EXPRESSIONS[expName];
      if (paramId) {
        coreModel.setParameterValueById(paramId, 1.0);
        coreModel.update();
        activeParams[paramId] = setTimeout(function () {
          try {
            // try: coreModel may have been destroyed during timer
            if (coreModel) {
              coreModel.setParameterValueById(paramId, 0);
              coreModel.update();
            }
          } catch (e) { /* best-effort cleanup — coreModel may be released */ }
          delete activeParams[paramId];
        }, 2500);
        console.log('[l2d] Expression:', expName, '\u2192', paramId, '= 1.0');
        return;
      }

      try {
        // try: .expression() may throw if the name is not found in exp3 definitions
        model.expression(expName);
        console.log('[l2d] Expression (native):', expName);
      } catch (e) {
        console.warn('[l2d] Unknown expression:', expName);
      }
    }

    api.onExpression(function (name) {
      console.log('[l2d] Expression received:', name);
      applyExpression(name);
    });

    api.onMotion(function (name) {
      if (!model) {
        console.warn('[l2d] Motion ignored: model not loaded');
        return;
      }
      console.log('[l2d] Motion requested:', name);
      try {
        model.motion(name);
        console.log('[l2d] Motion started:', name);
      } catch (e) {
        console.warn('[l2d] Motion not found, trying idle:', name, e.message);
        try {
          if (model.internalModel) {
            model.internalModel.coreModel.startRandomMotion('Idle', 3);
          }
        } catch (e2) { /* ignore */ }
      }
    });

    // Audio lip-sync — playback-timed mouth animation
    var mouthTicker = null;

    api.onAudioChunk(function (data) {
      console.log('[l2d] Audio chunk received, type:', typeof data, 'isArray:', Array.isArray(data), 'len:', data ? data.length : 'null');
      if (!data || !data.length) { console.warn('[l2d] Audio chunk empty or invalid'); return; }
      if (!model || !model.internalModel) return;
      try {
        var len = data.length;
        var rms = 0;
        for (var i = 0; i < len; i++) {
          rms += data[i] * data[i];
        }
        rms = Math.sqrt(rms / len);

        var mouthValue = Math.min(rms * 2.5, 1.0);
        model.internalModel.coreModel.setParameterValueById("ParamMouthOpenY", mouthValue);
        model.internalModel.coreModel.update();
      } catch (e) {
        console.debug('[l2d] Audio lip-sync error:', e.message);
      }
    });

    // WAV audio playback from Python TTS
    var _audioCtx = null;
    api.onWav(function(msg) {
      try {
        var b = atob(msg.audio);
        var bytes = new Uint8Array(b.length);
        for (var i = 0; i < b.length; i++) bytes[i] = b.charCodeAt(i);
        var f32 = new Float32Array(bytes.length / 2);
        var view = new DataView(bytes.buffer);
        for (var i = 0; i < f32.length; i++) f32[i] = view.getInt16(i*2, true) / 32768.0;
        var ctx = _audioCtx || (_audioCtx = new (window.AudioContext || window.webkitAudioContext)());
        if (ctx.state === 'suspended') ctx.resume();
        var buf = ctx.createBuffer(1, f32.length, msg.sampleRate);
        buf.getChannelData(0).set(f32);
        var src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        src.start();
      } catch(e) { console.warn('[audio]', e.message); }
    });

    // Playback-timed mouth animation — start/finish design
    // v5.x: On 'speak_start' → start ticker + show subtitle, on 'speak_finish' → stop ticker & reset mouth
    var mouthTicker = null;
    var _phase = 0;
    var _tickerActive = false;
    
    function _startTicker() {
      if (_tickerActive) return;
      _tickerActive = true;
      var cm = model && model.internalModel && model.internalModel.coreModel;
      if (!cm) { _tickerActive = false; return; }
      _phase = _phase || 0;
      clearInterval(mouthTicker);
      mouthTicker = setInterval(function() {
        if (!model || !model.internalModel) { clearInterval(mouthTicker); _tickerActive = false; return; }
        _phase += 0.50;
        var open = 0.3 + 0.5 * Math.abs(Math.sin(_phase));
        model.internalModel.coreModel.setParameterValueById('ParamMouthOpenY', open);
        model.internalModel.coreModel.update();
      }, 120);
    }
    
    function _stopTicker() {
      _tickerActive = false;
      if (mouthTicker) {
        clearInterval(mouthTicker);
        mouthTicker = null;
      }
      if (model && model.internalModel) {
        try {
          model.internalModel.coreModel.setParameterValueById('ParamMouthOpenY', 0);
          model.internalModel.coreModel.update();
        } catch (e) { /* best-effort reset — model may be destroyed */ }
      }
    }
    
    // 'speak_start' → start mouth animation + show sentence subtitle
    api.onStart && api.onStart(function(sentence) {
      _startTicker();
      if (sentence) {
        _showSubtitle({ role: 'assistant', text: sentence });
      }
    });
    
    // 'speak_finish' → stop mouth animation and close mouth
    api.onFinish && api.onFinish(function() {
      _stopTicker();
    });
    
    // v5.x fix: onSpeak handler with duration-based auto-stop
    // Falls back to ticker if onFinish is delayed or never arrives
    api.onSpeak && api.onSpeak(function(durationSeconds) {
      _startTicker();
      if (durationSeconds && durationSeconds > 0) {
        setTimeout(function() {
          _stopTicker();
        }, durationSeconds * 1000 + 500);
      }
    });

    // Subtitle overlay on PIXI canvas
    var _subtitleText = null;
    var _subtitleFade = null;
    var _subtitleDelay = null;
    function _showSubtitle(msg) {
      if (_subtitleFade) { clearTimeout(_subtitleFade); _subtitleFade = null; }
      if (_subtitleDelay) { clearTimeout(_subtitleDelay); _subtitleDelay = null; }
      if (_subtitleText) { app.stage.removeChild(_subtitleText); _subtitleText.destroy(); _subtitleText = null; }
      _subtitleText = new PIXI.Text((msg.role === 'user' ? '\u4f60: ' : '\u96ea\u5948: ') + msg.text, {
        fontFamily: 'Microsoft YaHei, sans-serif', fontSize: 28, fontWeight: 'bold',
        fill: msg.role === 'user' ? '#7ec8e3' : '#ffb7c5',
        stroke: '#000000', strokeThickness: 4,
        align: 'center', wordWrap: true, wordWrapWidth: app.screen.width * 0.8,
      });
      _subtitleText.anchor.set(0.5, 1);
      _subtitleText.x = app.screen.width / 2;
      _subtitleText.y = app.screen.height - 50;
      app.stage.addChild(_subtitleText);
      _subtitleDelay = setTimeout(function() {
        function _fade() {
          if (!_subtitleText) return;
          _subtitleText.alpha -= 0.03;
          if (_subtitleText.alpha <= 0) { app.stage.removeChild(_subtitleText); _subtitleText.destroy(); _subtitleText = null; }
          else { _subtitleFade = setTimeout(_fade, 100); }
        }
        _subtitleFade = setTimeout(_fade, 8000);
      }, 2000);
    }
    api.onSubtitle && api.onSubtitle(function(msg) { _showSubtitle(msg); });

    // Notify main process: model is ready
    try {
      api.sendReady();
      console.log('[l2d] Ready signal sent to main process');
    } catch (e) {
      console.error('[l2d] Failed to send ready signal:', e.message);
    }

  }).catch(function (err) {
    console.error('[l2d] Model load failed:', err.message || err);
    showPlaceholder('模型加载失败\n' + (err.message || err));
    try { api.sendReady(); } catch (e) { /* ignore */ }
  });

})();
