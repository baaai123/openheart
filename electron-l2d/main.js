// v4.5.0 §13 — Electron main process with WebSocket bridge for Python<->Live2D IPC
const { app, BrowserWindow, ipcMain, screen } = require('electron');

// GPU fallback for systems without hardware acceleration
app.commandLine.appendSwitch('enable-unsafe-swiftshader');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('disable-gpu-sandbox');

const path = require('path');
const { WebSocket } = require('ws');

const WS_PORT = 9876;
const WS_HOST = 'localhost';
const ENABLE_GLOBAL_TRACKING = true;
const CURSOR_POLL_MS = 33;

let l2dWindow = null;
let ws = null;
let reconnectTimer = null;
let cursorInterval = null;

function connectWebSocket() {
  if (ws) { try { ws.close(); } catch(e) {} }
  ws = new WebSocket(`ws://${WS_HOST}:${WS_PORT}`);
  
  ws.on('open', () => {
    console.log('[WS] Connected to Python voice engine');
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
    reconnectTimer = setTimeout(connectWebSocket, 3000);
  });

  ws.on('error', (err) => {
    console.warn('[WS] Error:', err.message);
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
    l2dWindow = null;
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

app.whenReady().then(() => {
  createWindow();
  connectWebSocket();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopGlobalCursorTracking();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
