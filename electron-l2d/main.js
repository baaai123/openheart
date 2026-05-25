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
  visualEnabled: true,
  l2dEnabled: true
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
      webSecurity: false
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
      webSecurity: false
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

ipcMain.handle('save-config', (_event, { key, value }) => {
  console.log('[Config] Save:', key, '=', value);
  // Load current config, update key, persist
  const cfg = loadConfigFile();
  cfg[key] = value;
  saveConfigFile(cfg);
  // Apply real-time effects for L2D window configs
  switch (key) {
    case 'width':
    case 'height':
      if (l2dWindow && !l2dWindow.isDestroyed()) {
        const currentSize = l2dWindow.getSize();
        const w = key === 'width' ? value : currentSize[0];
        const h = key === 'height' ? value : currentSize[1];
        l2dWindow.setSize(w, h);
      }
      break;
    case 'alwaysOnTop':
      if (l2dWindow && !l2dWindow.isDestroyed()) {
        l2dWindow.setAlwaysOnTop(value, 'screen-saver');
      }
      break;
    case 'clickThrough':
      if (l2dWindow && !l2dWindow.isDestroyed()) {
        l2dWindow.setIgnoreMouseEvents(value, { forward: true });
      }
      break;
    case 'devTools':
      break;
    case 'wsHost':
    case 'wsPort':
      break;
    default:
      // Backend config keys (baseUrl, model, apiKey, systemPrompt,
      // voiceEnabled, visualEnabled, l2dEnabled) are persisted to file
      // but have no real-time Electron window effect — handled when backend starts
      break;
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
    exec(`start cmd /c "${cmd}"`, (err) => {
      if (err) console.warn('[Terminal] Failed to open cmd:', err.message);
    });
  } else if (platform === 'darwin') {
    exec(`open -a Terminal.app "${cmd}"`, (err) => {
      if (err) console.warn('[Terminal] Failed to open Terminal.app:', err.message);
    });
  }
});

// ---- Backend control (v4.5.0 §13) ----

ipcMain.handle('start-backend', async () => {
  console.log('[Config] Start backend requested');
  sendConfigEvent('backend-status', 'starting');
  sendConfigEvent('backend-progress', { text: 'Starting backend server...', percent: 30 });

  // If a backend process is already running, kill it first
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }

  return new Promise((resolve, reject) => {
    // Spawn the Python backend server
    const serverScript = path.join(__dirname, 'run_server.py');
    backendProcess = spawn('python3', [serverScript], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    backendProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('[Backend]', output.trim());
      // Watch for ready signal from backend output
      if (output.includes('ready') || output.includes('listening') || output.includes('started')) {
        sendConfigEvent('backend-progress', { text: 'Starting L2D renderer...', percent: 60 });
        sendConfigEvent('backend-status', 'running');
      }
    });

    backendProcess.stderr.on('data', (data) => {
      console.warn('[Backend]', data.toString().trim());
    });

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

    // Give backend a moment to start, then start L2D connection
    setTimeout(() => {
      connectWebSocket();
      startHealthCheck();
      sendConfigEvent('l2d-status', 'running');
      sendConfigEvent('backend-progress', { text: 'Running', percent: 100 });
      resolve({ success: true });
    }, 3000);
  });
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
