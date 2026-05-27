// v4.5.0 §13 — Electron main process with WebSocket bridge for Python<->Live2D IPC
const { app, BrowserWindow, ipcMain, screen, dialog } = require('electron');

// GPU fallback for systems without hardware acceleration
app.commandLine.appendSwitch('enable-unsafe-swiftshader');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('disable-gpu-sandbox');

const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { WebSocket } = require('ws');

const WS_PORT = 9876;
const WS_HOST = 'localhost';
const ENABLE_GLOBAL_TRACKING = true;
const CURSOR_POLL_MS = 33;

// Config file persistence (v4.5.0 §13)
const CONFIG_FILE = path.join(app.getPath('userData'), 'config.json');
const CONFIG_DEFAULTS = {
  wsHost: WS_HOST,
  wsPort: WS_PORT,
  width: 500,
  height: 900,
  alwaysOnTop: true,
  clickThrough: true,
  devTools: false,
  modelPath: './models/xiaoyue/xiaoyue.model3.json',
  // OpenHeart backend config
  baseUrl: '',
  model: '',
  apiKey: '',
  systemPrompt: '',
  voiceEnabled: true,
  visualEnabled: true
};

function loadConfigFile() {
  try {
    const raw = fs.readFileSync(CONFIG_FILE, 'utf-8');
    const parsed = JSON.parse(raw);
    // Merge with defaults so missing keys get defaults
    return { ...CONFIG_DEFAULTS, ...parsed };
  } catch (err) {
    // File doesn't exist or is corrupted — use defaults
    if (err.code !== 'ENOENT') {
      console.warn('[Config] Failed to read config file:', err.message);
    }
    return { ...CONFIG_DEFAULTS };
  }
}

function saveConfigFile(cfg) {
  try {
    const dir = path.dirname(CONFIG_FILE);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(CONFIG_FILE, JSON.stringify(cfg, null, 2), 'utf-8');
    console.log('[Config] Saved to', CONFIG_FILE);
  } catch (err) {
    console.warn('[Config] Failed to write config file:', err.message);
  }
}

let l2dWindow = null;
let configWindow = null;
let ws = null;
let reconnectTimer = null;
let cursorInterval = null;
let backendProcess = null;

function connectWebSocket() {
  if (ws) { try { ws.close(); } catch(e) {} }
  ws = new WebSocket(`ws://${WS_HOST}:${WS_PORT}`);
  
  ws.on('open', () => {
    console.log('[WS] Connected to Python voice engine');
    sendConfigEvent('backend-status', 'running');
    sendConfigEvent('l2d-status', 'running');
    if (l2dWindow && !l2dWindow.isDestroyed()) {
      l2dWindow.webContents.send('model-ready');
    }
  });

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString());
      if (l2dWindow && !l2dWindow.isDestroyed()) {
        l2dWindow.webContents.send(msg.type, msg);
      }
    } catch (err) {
      console.warn('[WS] Failed to parse:', err.message);
    }
  });

  ws.on('close', () => {
    console.log('[WS] Disconnected. Reconnecting in 3s...');
    sendConfigEvent('backend-status', 'stopped');
    reconnectTimer = setTimeout(connectWebSocket, 3000);
  });

  ws.on('error', (err) => {
    console.warn('[WS] Error:', err.message);
    sendConfigEvent('backend-status', 'stopped');
  });
}

function createWindow() {
  l2dWindow = new BrowserWindow({
    width: 500,
    height: 900,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: true,
    hasShadow: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false,
      devTools: false
    }
  });

  l2dWindow.setAlwaysOnTop(true, 'screen-saver');
  l2dWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  // Click-through: mouse passes through transparent areas to windows below.
  // Toggle with Ctrl+Shift+P or right-click context menu.
  let clickThrough = true;
  l2dWindow.setIgnoreMouseEvents(clickThrough, { forward: true });

  const { globalShortcut } = require('electron');
  globalShortcut.register('CommandOrControl+Shift+P', () => {
    clickThrough = !clickThrough;
    l2dWindow.setIgnoreMouseEvents(clickThrough, { forward: true });
    console.log('[ClickThrough]', clickThrough ? 'ON' : 'OFF');
  });
  
  l2dWindow.on('blur', () => {
    if (l2dWindow && !l2dWindow.isDestroyed()) {
      l2dWindow.setAlwaysOnTop(true, 'screen-saver');
    }
  });

  l2dWindow.loadFile('index.html');
  l2dWindow.webContents.openDevTools({ mode: 'detach' });

  // Open DevTools if --dev flag passed
  if (process.argv.includes('--dev')) {
    l2dWindow.webContents.openDevTools({ mode: 'detach' });
  }

  l2dWindow.on('closed', () => {
    stopGlobalCursorTracking();
    if (configWindow && !configWindow.isDestroyed()) {
      configWindow.close();
    }
    l2dWindow = null;
  });
}

// ---- Config/control panel window ----
function createConfigWindow() {
  configWindow = new BrowserWindow({
    width: 400,
    height: 600,
    title: 'OpenHeart Control Panel',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false,
      devTools: false
    }
  });

  configWindow.loadFile(path.join(__dirname, 'config.html'));

  configWindow.on('closed', () => {
    configWindow = null;
  });
}

// ---- Global cursor tracking via screen.getCursorScreenPoint ----
function startGlobalCursorTracking() {
  if (!ENABLE_GLOBAL_TRACKING) return;
  if (cursorInterval) return;

  console.log('[Cursor] Global cursor tracking started');

  cursorInterval = setInterval(() => {
    if (!l2dWindow || l2dWindow.isDestroyed()) return;

    // try: screen API may fail during display reconfiguration
    try {
      const point = screen.getCursorScreenPoint();
      const [winX, winY] = l2dWindow.getPosition();
      const [winW, winH] = l2dWindow.getSize();
      const display = screen.getPrimaryDisplay();
      const { width: scrW, height: scrH } = display.workAreaSize;

      // nx/ny: normalized [0,1] relative to L2D window (preload.js API contract)
      const nx = (point.x - winX) / winW;
      const ny = (point.y - winY) / winH;

      // lx/ly: window-local coordinates (for fake mousemove fallback in renderer)
      const lx = point.x - winX;
      const ly = point.y - winY;

      l2dWindow.webContents.send('global-cursor', {
        x: point.x,
        y: point.y,
        nx: Math.max(0, Math.min(1, nx)),
        ny: Math.max(0, Math.min(1, ny)),
        lx,
        ly,
        screenWidth: winW,
        screenHeight: winH
      });
    } catch (err) {
      // Display reconfiguration or transient error — non-fatal
      console.warn('[Cursor] Failed to sample cursor:', err.message);
    }
  }, CURSOR_POLL_MS);
}

function stopGlobalCursorTracking() {
  if (cursorInterval) {
    clearInterval(cursorInterval);
    cursorInterval = null;
    console.log('[Cursor] Global cursor tracking stopped');
  }
}

// IPC: renderer signals it's ready for model commands
ipcMain.on('ready', () => {
  console.log('[IPC] Renderer signaled ready');
  startGlobalCursorTracking();
  ws.send(JSON.stringify({ type: "ready" }), function(err) {
    if (err) console.error("[MAIN] ws.send error:", err);
  });
});

// ---- IPC: Config window handlers ----
// Helper: send a message to config window if it exists
function sendConfigEvent(channel, data) {
  if (configWindow && !configWindow.isDestroyed()) {
    configWindow.webContents.send(channel, data);
  }
}

// Backend health check polling (v4.5.0 §13)
let healthCheckInterval = null;

function startHealthCheck() {
  stopHealthCheck();
  healthCheckInterval = setInterval(() => {
    const isWsOpen = ws && ws.readyState === WebSocket.OPEN;
    sendConfigEvent('backend-status', isWsOpen ? 'running' : 'stopped');
  }, 5000);
}

function stopHealthCheck() {
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval);
    healthCheckInterval = null;
  }
}

ipcMain.handle('load-config', () => loadConfigFile());

ipcMain.handle('save-config', (_event, param) => {
  console.log('[Config] Save:', param);
  const cfg = loadConfigFile();

  if (param && param.key !== undefined) {
    // Backward compat: old { key, value } call from other places
    cfg[param.key] = param.value;
    saveConfigFile(cfg);
    // Apply real-time effects for L2D window configs (only old-format keys trigger these)
    switch (param.key) {
      case 'width':
      case 'height':
        if (l2dWindow && !l2dWindow.isDestroyed()) {
          const currentSize = l2dWindow.getSize();
          const w = param.key === 'width' ? param.value : currentSize[0];
          const h = param.key === 'height' ? param.value : currentSize[1];
          l2dWindow.setSize(w, h);
        }
        break;
      case 'alwaysOnTop':
        if (l2dWindow && !l2dWindow.isDestroyed()) {
          l2dWindow.setAlwaysOnTop(param.value, 'screen-saver');
        }
        break;
      case 'clickThrough':
        if (l2dWindow && !l2dWindow.isDestroyed()) {
          l2dWindow.setIgnoreMouseEvents(param.value, { forward: true });
        }
        break;
      case 'devTools':
        break;
      case 'wsHost':
      case 'wsPort':
        break;
      default:
        // Backend config keys (baseUrl, model, apiKey, systemPrompt,
        // voiceEnabled, visualEnabled) are persisted to file
        // but have no real-time Electron window effect — handled when backend starts
        break;
    }
  } else {
    // New: full config object from config_renderer.js saveConfigToBackend IPC fallback
    const backendKeys = ['baseUrl', 'model', 'apiKey', 'systemPrompt', 'voiceEnabled', 'visualEnabled', 'voiceMode'];
    for (const k of backendKeys) {
      if (k in param) cfg[k] = param[k];
    }
    saveConfigFile(cfg);
  }
  return { success: true };
});

ipcMain.on('send-expression', (_event, name) => {
  console.log('[Config] Expression:', name);
  if (l2dWindow && !l2dWindow.isDestroyed()) {
    l2dWindow.webContents.send('expression', { name });
  }
});

ipcMain.on('reconnect-ws', () => {
  console.log('[Config] WS reconnect requested');
  if (reconnectTimer) clearTimeout(reconnectTimer);
  connectWebSocket();
});

// ---- Informational dialogs (v4.5.0 §13) ----

ipcMain.handle('show-message', async (_event, msg) => {
  const result = await dialog.showMessageBox(configWindow, {
    type: 'info',
    title: 'OpenHeart Backend',
    message: msg,
    buttons: ['OK']
  });
  return result.response;
});

ipcMain.handle('open-terminal', async (_event, cmd) => {
  const { exec } = require('child_process');
  // Platform-aware terminal launcher
  const platform = process.platform;
  if (platform === 'linux') {
    exec(`x-terminal-emulator -e bash -c "${cmd}; exec bash"`, (err) => {
      if (err) console.warn('[Terminal] Failed to open x-terminal-emulator:', err.message);
    });
  } else if (platform === 'win32') {
    exec(`start cmd /k "${cmd}"`, (err) => {
      if (err) console.warn('[Terminal] Failed to open cmd:', err.message);
    });
  } else if (platform === 'darwin') {
    exec(`open -a Terminal.app "${cmd}"`, (err) => {
      if (err) console.warn('[Terminal] Failed to open Terminal.app:', err.message);
    });
  }
});

// ---- Real config file writers (v4.5.0 §13) ----

ipcMain.handle('write-env', async (_event, params) => {
  // params: { DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL }
  // Read-modify-write .env file (KEY=VALUE format, one per line)
  // Write to WSL path so the backend (running in WSL) can read it (v4.5.0 §13)
  const envPath = '\\\\wsl.localhost\\Ubuntu\\home\\baaai\\projects\\openheart\\.env';
  let content = '';
  try {
    content = fs.readFileSync(envPath, 'utf-8');
  } catch (err) {
    // File doesn't exist yet — start with empty content; safe because we'll write a fresh one
    if (err.code !== 'ENOENT') console.warn('[write-env] Failed to read .env:', err.message);
  }
  const lines = content ? content.split('\n') : [];
  const updateLine = (prefix, value) => {
    if (!value || (typeof value === 'string' && value.trim() === '')) return; // skip empty — keep existing
    const idx = lines.findIndex(l => l.startsWith(prefix + '='));
    const newLine = prefix + '=' + value;
    if (idx >= 0) lines[idx] = newLine;
    else lines.push(newLine);
  };
  updateLine('DEEPSEEK_API_KEY', params.DEEPSEEK_API_KEY);
  updateLine('DEEPSEEK_BASE_URL', params.DEEPSEEK_BASE_URL);
  updateLine('DEEPSEEK_MODEL', params.DEEPSEEK_MODEL);
  const filtered = lines.filter(l => l.trim() !== '');
  fs.writeFileSync(envPath, filtered.join('\n') + '\n', 'utf-8');
  console.log('[write-env] Written to', envPath);
  return { success: true };
});

ipcMain.handle('write-prompt-modules', async (_event, params) => {
  // Write to WSL path so the backend (running in WSL) can read it (v4.5.0 §13)
  const pmPath = '\\\\wsl.localhost\\Ubuntu\\home\\baaai\\projects\\openheart\\config\\prompt_modules.json';
  let data = {};
  try {
    data = JSON.parse(fs.readFileSync(pmPath, 'utf-8'));
  } catch (err) {
    // File doesn't exist or is invalid JSON — start fresh; safe because we write valid JSON back
    if (err.code !== 'ENOENT') console.warn('[write-prompt-modules] Failed to read:', err.message);
  }
  if (params.persona !== undefined) {
    data.persona = params.persona;
  }
  fs.writeFileSync(pmPath, JSON.stringify(data, null, 2) + '\n', 'utf-8');
  console.log('[write-prompt-modules] Written to', pmPath);
  return { success: true };
});

ipcMain.handle('write-ui-settings', async (_event, params) => {
  // Write to WSL path so the backend (running in WSL) can read it (v4.5.0 §13)
  const uiPath = '\\\\wsl.localhost\\Ubuntu\\home\\baaai\\projects\\openheart\\config\\ui_settings.json';
  let data = {};
  try {
    data = JSON.parse(fs.readFileSync(uiPath, 'utf-8'));
  } catch (err) {
    if (err.code !== 'ENOENT') console.warn('[write-ui-settings] Failed to read:', err.message);
  }
  if (params.visual_enabled !== undefined) data.visual_enabled = params.visual_enabled;
  if (params.voice_mode !== undefined) data.voice_mode = params.voice_mode;
  fs.writeFileSync(uiPath, JSON.stringify(data, null, 2) + '\n', 'utf-8');
  console.log('[write-ui-settings] Written to', uiPath);
  return { success: true };
});

// ---- Backend control (v4.5.0 §13) ----

ipcMain.handle('start-backend', async () => {
  console.log('[Config] Start backend requested');
  sendConfigEvent('backend-status', 'starting');
  sendConfigEvent('backend-progress', { text: 'Launching AI backend...', percent: 60 });

  // If a backend process is already running, kill it first
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }

  return new Promise((resolve, reject) => {
    // Spawn the Python backend server silently — no visible terminal window
    const serverScript = path.join(__dirname, 'run_server.py');
    backendProcess = spawn('python3', [serverScript], {
      cwd: __dirname,
      detached: true,
      stdio: 'ignore'
    });

    // Unref so the child can outlive the parent if parent exits
    backendProcess.unref();

    backendProcess.on('error', (err) => {
      console.error('[Backend] Failed to start:', err.message);
      stopHealthCheck();
      sendConfigEvent('backend-status', 'stopped');
      sendConfigEvent('backend-progress', { text: 'Failed: ' + err.message, percent: 0 });
      backendProcess = null;
      reject(err);
    });

    backendProcess.on('exit', (code) => {
      console.log('[Backend] Process exited with code', code);
      if (code !== 0 && code !== null) {
        stopHealthCheck();
        sendConfigEvent('backend-status', 'stopped');
        sendConfigEvent('backend-progress', { text: 'Backend exited (code ' + code + ')', percent: 0 });
      }
      backendProcess = null;
    });

    // Also launch run_backend.sh in a visible terminal (platform-aware)
    // v4.5.0 §13: This starts the full AI backend (ASR, TTS, LLM, memory)
    const { exec } = require('child_process');
    const projectRoot = path.resolve(__dirname, '..');
    const backEndScript = path.join(projectRoot, 'run_backend.sh');
    const platform = process.platform;

    if (platform === 'win32') {
      // Windows: launch run_backend.sh directly in WSL via Windows Terminal
      const wslCmd = 'wsl -d Ubuntu -- bash -i /home/baaai/projects/openheart/run_backend.sh --no-api-check';
      exec(`start "OpenHeart Backend" ${wslCmd}`, (err) => {
        if (err) console.warn('[Backend] Failed to open WSL terminal:', err.message);
      });
    } else if (platform === 'linux') {
      // Linux: launch in x-terminal-emulator
      const termCmd = `bash -c "cd ${projectRoot} && bash ${backEndScript} --no-api-check; exec bash"`;
      exec(`x-terminal-emulator -e ${termCmd}`, (err) => {
        if (err) console.warn('[Backend] Failed to open x-terminal-emulator:', err.message);
      });
    } else if (platform === 'darwin') {
      // macOS: launch in Terminal.app
      const termCmd = `cd ${projectRoot} && bash ${backEndScript} --no-api-check`;
      exec(`open -a Terminal.app "${termCmd}"`, (err) => {
        if (err) console.warn('[Backend] Failed to open Terminal.app:', err.message);
      });
    }

    // Send progress update after launching both processes
    sendConfigEvent('backend-progress', { text: 'Waiting for API...', percent: 80 });

    // Give backend a moment to start, then start L2D connection
    setTimeout(() => {
      connectWebSocket();
      startHealthCheck();
      sendConfigEvent('l2d-status', 'running');
      sendConfigEvent('backend-progress', { text: 'All ready', percent: 100 });
      resolve({ success: true });
    }, 3000);
  });
});

// stop-backend: kills the backend process via WSL pkill + local process
ipcMain.on('stop-backend', () => {
  console.log('[Config] Stop backend requested');
  const { exec } = require('child_process');
  exec('wsl pkill -f demo_full.py', (err) => {
    if (err) console.warn('[Backend] pkill failed:', err.message);
  });
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
  sendConfigEvent('backend-status', 'stopped');
});

// stop-l2d: closes the Live2D render window
ipcMain.on('stop-l2d', () => {
  console.log('[IPC] stop-l2d requested');
  if (l2dWindow && !l2dWindow.isDestroyed()) {
    l2dWindow.close();
  }
});

app.whenReady().then(() => {
  createWindow();
  createConfigWindow();
  connectWebSocket();

  app.on('activate', () => {
    // MacOS re-activate: re-open config panel, not L2D window
    if (!configWindow || configWindow.isDestroyed()) {
      createConfigWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopGlobalCursorTracking();
  stopHealthCheck();
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
