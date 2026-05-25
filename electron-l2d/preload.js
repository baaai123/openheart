// preload.js — IPC bridge for Live2D Electron renderer
// v4.5.0 §7.3.4
// Runs with contextIsolation: true, nodeIntegration: false
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  /**
   * Notify main process that the model has finished loading and is ready to render.
   */
  sendReady: () => ipcRenderer.send('ready'),

  /**
   * Listen for expression commands from main process.
   * Callback receives the expression name (string).
   */
  onExpression: (callback) => {
    ipcRenderer.on('expression', (_event, msg) => callback(msg.name));
  },

  /**
   * Listen for motion commands from main process.
   * Callback receives the motion name (string).
   */
  onMotion: (callback) => {
    ipcRenderer.on('motion', (_event, msg) => callback(msg.name));
  },

  /**
   * Listen for audio PCM data from main process for lip-sync.
   * Callback receives a Float32Array of PCM samples.
   */
  onAudioChunk: (callback) => {
    ipcRenderer.on('audio', (_event, msg) => callback(msg.samples));
  },

  /**
   * Listen for global cursor position from main process.
   * Callback receives { x, y, nx, ny, lx, ly, screenWidth, screenHeight }.
   *   nx, ny: normalized [0,1] relative to L2D window
   *   lx, ly: window-local coordinates (for fake mousemove fallback)
   */
  onGlobalCursor: (callback) => {
    ipcRenderer.on('global-cursor', (_event, pos) => callback(pos));
  },

  /**
   * Listen for model-ready acknowledgement from main process.
   */
  onSpeak: (callback) => {
    ipcRenderer.on('speak', (_event, msg) => callback(msg.duration));
  },

  onSpeakStop: (callback) => {
    ipcRenderer.on('speak_stop', () => callback());
  },

  /**
   * Listen for speech start with sentence text.
   * Callback receives the sentence string.
   */
  onStart: (callback) => {
    ipcRenderer.on('start', (_event, msg) => callback(msg.sentence));
  },

  /**
   * Listen for speech finish signal.
   */
  onFinish: (callback) => {
    ipcRenderer.on('finish', () => callback());
  },

  onWav: (callback) => { ipcRenderer.on("wav", (_event, msg) => callback(msg)); },

  onSubtitle: (callback) => { ipcRenderer.on("subtitle", (_event, msg) => callback(msg)); },

  onModelReady: (callback) => {
    ipcRenderer.on('model-ready', () => callback());
  },

  // ---- Config window bridge methods (v4.5.0 §7.3.4) ----
  loadConfig: () => ipcRenderer.invoke('load-config'),

  saveConfig: (param) => ipcRenderer.invoke('save-config', param),

  sendExpression: (name) => ipcRenderer.send('send-expression', name),

  reconnectWs: () => ipcRenderer.send('reconnect-ws'),

  // ---- L2D window control (v4.5.0 §13) ----
  startL2D: () => ipcRenderer.invoke('start-l2d'),

  stopL2D: () => ipcRenderer.send('stop-l2d'),

  // ---- Backend control (v4.5.0 §13) ----
  startBackend: () => ipcRenderer.invoke('start-backend'),

  stopBackend: () => ipcRenderer.send('stop-backend'),

  onBackendStatus: (callback) => {
    ipcRenderer.on('backend-status', (_event, state) => callback(state));
  },

  onL2DStatus: (callback) => {
    ipcRenderer.on('l2d-status', (_event, state) => callback(state));
  },

  onBackendProgress: (callback) => {
    ipcRenderer.on('backend-progress', (_event, data) => callback(data));
  },

  // ---- Real config file writers (v4.5.0 §13) ----
  writeEnv: (params) => ipcRenderer.invoke('write-env', params),
  writePromptModules: (params) => ipcRenderer.invoke('write-prompt-modules', params),
  writeUISettings: (params) => ipcRenderer.invoke('write-ui-settings', params),

  /**
   * Show an informational message box in the config window.
   * Returns the index of the clicked button.
   */
  showMessage: (msg) => ipcRenderer.invoke('show-message', msg),

  /**
   * Open a terminal window running the given command.
   */
  openTerminal: (cmd) => ipcRenderer.invoke('open-terminal', cmd),
});
