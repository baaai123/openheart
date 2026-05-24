/**
 * Electron main process — L2D WebSocket bridge to Python.
 *
 * Connects to Python Live2DServer on ws://localhost:9876.
 * Forwards renderer ipc requests and renders Live2D model.
 */

const { app, BrowserWindow, ipcMain } = require('electron');
const WebSocket = require('ws');

let ws = null;
let wsReady = false;
let mainWindow = null;

// ── WebSocket connection ─────────────────────────────────────────────

function connectWebSocket() {
  console.log('[L2D] Connecting WebSocket to ws://localhost:9876 ...');
  ws = new WebSocket('ws://localhost:9876');

  ws.on('open', () => {
    console.log('[L2D] WebSocket open — sending ready signal');
    wsReady = true;

    // Send ready message
    const readyMsg = JSON.stringify({ type: 'ready' });
    console.log('[L2D] ws.send data:', readyMsg);
    ws.send(readyMsg);
    console.log('[L2D] ws.send complete. ws.readyState:', ws.readyState);

    // If ws.send is async in this ws version, use callback form
    // ws v7+ supports both sync and callback
  });

  ws.on('message', (data) => {
    console.log('[L2D] Received from Python:', data.toString().substring(0, 200));
    try {
      const msg = JSON.parse(data.toString());
      // Forward to renderer if needed
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('l2d-message', msg);
      }
    } catch (e) {
      console.warn('[L2D] Failed to parse incoming WS message:', e.message);
    }
  });

  ws.on('close', (code, reason) => {
    console.log('[L2D] WebSocket closed — code:', code, 'reason:', reason ? reason.toString() : '');
    wsReady = false;
    // Reconnect after 1s
    setTimeout(connectWebSocket, 1000);
  });

  ws.on('error', (err) => {
    console.error('[L2D] WebSocket error:', err.message);
    wsReady = false;
  });
}

// ── IPC handlers ─────────────────────────────────────────────────────

ipcMain.handle('ws-send-ready', async () => {
  console.log('[IPC] ws-send-ready handler called');
  console.log('[IPC] ws state:', ws ? `readyState=${ws.readyState}` : 'ws is null');
  console.log('[IPC] wsReady flag:', wsReady);

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.error('[IPC] ws-send-ready FAILED: WebSocket not open');
    return { success: false, error: 'WebSocket not connected' };
  }

  const data = JSON.stringify({ type: 'ready' });
  console.log('[IPC] Sending data:', data, '(length:', data.length, ')');

  // v5.x fix: wrap ws.send callback in Promise — returns only after send confirms
  return new Promise((resolve) => {
    try {
      ws.send(data, (err) => {
        if (err) {
          console.error('[IPC] ws.send async error:', err.message);
          resolve({ success: false, error: err.message });
        } else {
          console.log('[IPC] ws.send async confirmed');
          resolve({ success: true });
        }
      });
    } catch (e) {
      console.error('[IPC] ws.send threw:', e.message);
      resolve({ success: false, error: e.message });
    }
  });
});

ipcMain.handle('ws-send', async (_event, msg) => {
  console.log('[IPC] ws-send called, type:', msg.type);
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return { success: false, error: 'WebSocket not connected' };
  }
  const data = JSON.stringify(msg);
  console.log('[IPC] Sending data (%d bytes):', data.length, data.substring(0, 200));
  try {
    ws.send(data);
    console.log('[IPC] ws-send complete');
    return { success: true };
  } catch (e) {
    console.error('[IPC] ws-send error:', e.message);
    return { success: false, error: e.message };
  }
});

// Disable GPU output for debugging
ipcMain.handle('disable-gpu', () => {
  app.disableHardwareAcceleration();
  return { success: true };
});

// ── App lifecycle ────────────────────────────────────────────────────

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: __dirname + '/preload.js',
    },
  });
  mainWindow.loadFile('index.html');
}

app.whenReady().then(() => {
  connectWebSocket();
  createWindow();
});

app.on('window-all-closed', () => {
  if (ws) ws.close();
  app.quit();
});
