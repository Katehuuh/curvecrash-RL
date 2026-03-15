// ==UserScript==
// @name         CurveCrash AI v6
// @namespace    curvecrash-ai
// @version      6.1
// @description  GS++ game-state obs + IMPALA-CNN + Voronoi territory (6-7ch, up to 2.3M params)
// @match        *://curvecrash.com/*
// @match        *://www.curvecrash.com/*
// @run-at       document-start
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_xmlhttpRequest
// @grant        unsafeWindow
// @require      https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.13.0/dist/tf.min.js
// ==/UserScript==

// Patch WebGL IMMEDIATELY before game creates its canvas
(function() {
    const _getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(type, attrs) {
        if (type === 'webgl' || type === 'webgl2') {
            attrs = attrs || {};
            attrs.preserveDrawingBuffer = true;
        }
        return _getContext.call(this, type, attrs);
    };
})();

(function() {
    'use strict';

    // Bridge to page's real window (Tampermonkey sandbox isolation)
    const pageWindow = (typeof unsafeWindow !== 'undefined') ? unsafeWindow : window;

    // ==================== CONFIGURATION ====================
    const CFG = {
        RES: 128,
        DEC_MS: 33,
        IDLE_LIMIT: 20,
        MIN_ROUND_SEC: 1.0,
        COOLDOWN_MS: 1000,
        WALL_BRIGHT: 140,
        BG_BRIGHT: 15,
        WEIGHTS_KEY: 'cc_v6_weights',
        POWERUP_RADIUS_FRAC: 0.0625 / 2,  // powerup radius as fraction of field height
    };

    // GS++ geometry: must match env exactly (curvecrash_env_ffa.py)
    // Real game: 922x763. Env maps to 512x512 sim grid preserving aspect ratio.
    // Playable area in sim: 512 wide x 424 tall, centered vertically (offset=44).
    // Observation: 128x128, downsampled 4x from sim.
    const GSPP_FW = 922;
    const GSPP_FH = 763;
    const ARENA_SIM = 512;
    const DS_FACTOR = ARENA_SIM / CFG.RES; // 4
    const SIM_SCALE = ARENA_SIM / GSPP_FW; // 512/922 ≈ 0.5553
    const OBS_SCALE = SIM_SCALE / DS_FACTOR; // 128/922 ≈ 0.1388 (UNIFORM for both X and Y)
    const ARENA_H_GSPP = Math.round(GSPP_FH * SIM_SCALE); // 424
    const ARENA_OFFSET_Y = Math.floor((ARENA_SIM - ARENA_H_GSPP) / 2); // 44
    const OBS_OFFSET_Y = ARENA_OFFSET_Y / DS_FACTOR; // 11

    // ==================== STATE ====================
    const S = {
        gc: null,
        ctx: null,
        model: null,
        gruState: null,      // legacy CPU state (for diagnostic testLogits)
        gruTensor: null,     // GPU-resident GRU hidden state (tf.keep'd)
        myColor: null,
        roundActive: false,
        roundStart: 0,
        cooldownUntil: 0,
        idleFrames: 0,
        prevSelf: null,
        prevEnemy: null,
        steps: 0,
        ep: 0,
        action: 1,
        pressed: null,
        // Game state (from JS API)
        gsAvailable: false,
        gsPosition: null,   // {x, y} in field coords
        gsAngle: null,       // radians
        gsFieldW: 0,
        gsFieldH: 0,
        gsPowerups: [],      // [{x, y, powerupId}, ...]
        gsAlive: true,
        gsSpeedMult: 1.0,
        lastAliveCheck: 0,
        // Trail tracking (self-built, since c.particles is always empty)
        selfTrail: null,     // Float32Array(128*128) — persistent self trail grid
        enemyTrail: null,    // Float32Array(128*128) — persistent enemy trail grid
        prevTrailPos: {},    // curveId -> {fieldX, fieldY, serverX, serverY, lastTs}
        prevPowerupIds: new Set(), // track powerup IDs to detect erase pickup
        // Perf metrics
        inferMs: 0,          // last inference time in ms
        inferAvg: 0,         // exponential moving average of inference time
    };

    // ==================== CANVAS DETECTION ====================
    function findGameCanvas() {
        const canvases = [...document.querySelectorAll('canvas')].filter(c =>
            c.id !== 'ai-viz' && c.width >= 200 && c.height >= 200
        );
        if (canvases.length === 0) return null;
        canvases.sort((a, b) => (b.width * b.height) - (a.width * a.height));
        return canvases[0];
    }

    // ==================== GAME STATE ACCESS ====================
    function readGameState() {
        /**
         * Read exact game state from window.gameGraphics.gameEngine.round.
         * Returns true if game state is available and we found our player.
         */
        try {
            const round = pageWindow.gameGraphics?.gameEngine?.round;
            if (!round) {
                S.gsAvailable = false;
                return false;
            }

            const curves = round.getCurves();
            if (!curves || curves.length === 0) {
                S.gsAvailable = false;
                return false;
            }

            // Find our player (isMyPlayer is on c.player, not c)
            let me = null;
            for (const c of curves) {
                if (c.player?.isMyPlayer) {
                    me = c;
                    break;
                }
            }
            if (!me) {
                S.gsAvailable = false;
                return false;
            }

            S.gsAvailable = true;
            S.gsPosition = { x: me.state.x, y: me.state.y };
            S.gsAngle = me.state.angle;
            S.gsAlive = me.state.isAlive;
            // speed is in pixels/frame (base=3 at 60fps = curveSpeed/fps = 180/60)
            const baseSpeed = round.gameSettings?.curveSpeed / round.gameSettings?.fps || 3;
            S.gsSpeedMult = baseSpeed > 0 ? me.state.speed / baseSpeed : 1.0;

            // Get player color from game state (colour is on c.player, as an object)
            if (!S.myColor && me.player?.colour) {
                const col = me.player.colour;
                // colour is {headColours: ["#ffa600"], ...}
                const hex = col.headColours?.[0] || col.textColours?.[0];
                if (hex && typeof hex === 'string' && hex.startsWith('#')) {
                    const r = parseInt(hex.slice(1, 3), 16);
                    const g = parseInt(hex.slice(3, 5), 16);
                    const b = parseInt(hex.slice(5, 7), 16);
                    S.myColor = [r, g, b];
                    console.log(`[AI] Color from game state: rgb(${r},${g},${b}) [${col.name}]`);
                }
            }

            // Get field dimensions from game settings (on round.gameSettings directly)
            const gs = round.gameSettings;
            if (gs) {
                S.gsFieldW = gs.fieldWidth || S.gc?.width || 800;
                S.gsFieldH = gs.fieldHeight || S.gc?.height || 600;
            }
            if (!S.gsFieldW) {
                S.gsFieldW = S.gc?.width || 800;
                S.gsFieldH = S.gc?.height || 600;
            }

            // Read powerups from round.state.fieldPowerups
            // Powerup coords are in p.state.x / p.state.y (same pattern as curves)
            S.gsPowerups = [];
            const currentPupIds = new Set();
            const fieldPowerups = round.state?.fieldPowerups;
            if (fieldPowerups && Array.isArray(fieldPowerups)) {
                for (const p of fieldPowerups) {
                    if (p?.state && p.state.x !== undefined && !p.state.isPicked) {
                        const pid = p.powerupId || 0;
                        S.gsPowerups.push({
                            x: p.state.x,
                            y: p.state.y,
                            powerupId: pid,
                            uid: p.id || `${p.state.x}_${p.state.y}`,
                        });
                        currentPupIds.add(p.id || `${p.state.x}_${p.state.y}`);
                    }
                }
            }

            // Detect erase pickup: if an erase powerup disappeared, clear trail grids
            // Erase powerupId != 1 (1 = speed). When eraser is picked up, trails get wiped.
            if (S.prevPowerupList && S.selfTrail) {
                for (const prev of S.prevPowerupList) {
                    if (prev.powerupId !== 1 && !currentPupIds.has(prev.uid)) {
                        // Erase powerup disappeared — clear trail grids
                        const NN = CFG.RES * CFG.RES;
                        S.selfTrail.fill(0);
                        S.enemyTrail.fill(0);
                        console.log(`[AI] Erase detected! Cleared trail grids.`);
                        break;
                    }
                }
            }
            S.prevPowerupList = S.gsPowerups;

            return true;
        } catch (e) {
            S.gsAvailable = false;
            return false;
        }
    }

    // ==================== PIXEL CAPTURE ====================
    function forceRender() {
        /** Force PixiJS to render before we read the canvas.
         *  Without this, our rAF callback runs before PIXI renders → canvas is cleared → black. */
        try {
            const app = pageWindow.gameGraphics?.pixiApp;
            if (app?.renderer && app.stage) {
                app.renderer.render(app.stage);
            }
        } catch(e) {}
    }

    function capture() {
        if (!S.gc) return null;
        try {
            forceRender();
            S.ctx.imageSmoothingEnabled = false;
            S.ctx.drawImage(S.gc, 0, 0, CFG.RES, CFG.RES);
            const imgData = S.ctx.getImageData(0, 0, CFG.RES, CFG.RES);

            if (!S.captureVerified) {
                let nonBlack = 0;
                const d = imgData.data, N = CFG.RES;
                for (const row of [0, 1, N >> 1, N - 2, N - 1]) {
                    for (let col = 0; col < N; col += 2) {
                        const i = (row * N + col) * 4;
                        if (d[i] + d[i+1] + d[i+2] > 10) nonBlack++;
                    }
                }
                if (nonBlack > 0) {
                    S.captureVerified = true;
                    S.captureMode = 'drawImage';
                } else {
                    const fallback = captureWebGL();
                    if (fallback) {
                        S.captureVerified = true;
                        S.captureMode = 'readPixels';
                        return fallback;
                    }
                }
            }

            if (S.captureMode === 'readPixels') return captureWebGL();
            return imgData;
        } catch (e) {
            return null;
        }
    }

    function captureWebGL() {
        if (!S.gc) return null;
        try {
            const gl = S.gc.getContext('webgl2') || S.gc.getContext('webgl');
            if (!gl) return null;
            const w = S.gc.width, h = S.gc.height;

            if (!S._pxBuf || S._pxBufSize !== w * h * 4) {
                S._pxBuf = new Uint8Array(w * h * 4);
                S._pxFlip = new Uint8Array(w * h * 4);
                S._pxBufSize = w * h * 4;
            }
            gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, S._pxBuf);

            for (let row = 0; row < h; row++) {
                const srcOff = row * w * 4;
                const dstOff = (h - 1 - row) * w * 4;
                S._pxFlip.set(S._pxBuf.subarray(srcOff, srcOff + w * 4), dstOff);
            }

            const N = CFG.RES;
            const result = S.ctx.createImageData(N, N);
            const rd = result.data;
            const src = S._pxFlip;
            const fX = w / N;
            const fY = h / N;

            for (let oy = 0; oy < N; oy++) {
                const yS = Math.floor(oy * fY);
                const yE = Math.min(h, Math.ceil((oy + 1) * fY));
                for (let ox = 0; ox < N; ox++) {
                    const xS = Math.floor(ox * fX);
                    const xE = Math.min(w, Math.ceil((ox + 1) * fX));

                    let bestR = 0, bestG = 0, bestB = 0, bestBright = 0;
                    for (let sy = yS; sy < yE; sy++) {
                        for (let sx = xS; sx < xE; sx++) {
                            const si = (sy * w + sx) * 4;
                            const r = src[si], g = src[si+1], b = src[si+2];
                            const bright = r + g + b;
                            if (bright > bestBright) {
                                bestBright = bright;
                                bestR = r; bestG = g; bestB = b;
                            }
                        }
                    }
                    const di = (oy * N + ox) * 4;
                    rd[di] = bestR;
                    rd[di+1] = bestG;
                    rd[di+2] = bestB;
                    rd[di+3] = 255;
                }
            }
            return result;
        } catch(e) {
            return null;
        }
    }

    // ==================== TRAIL LINE RASTERIZER ====================
    function rasterizeTrailLine(grid, x0, y0, x1, y1, hw, N) {
        /**
         * Draw a line from (x0,y0) to (x1,y1) on grid with half-width hw.
         * Matches env's _draw_trail(): interpolate along segment, stamp perpendicular.
         * grid: Float32Array(N*N), coords in obs space (0..N-1).
         */
        const dx = x1 - x0, dy = y1 - y0;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 0.01) return; // no movement

        const nStamps = Math.max(1, Math.ceil(dist));
        // Perpendicular direction for width stamping
        const perpX = -dy / dist;
        const perpY = dx / dist;

        for (let s = 0; s <= nStamps; s++) {
            const t = s / nStamps;
            const cx = x0 + dx * t;
            const cy = y0 + dy * t;

            for (let i = -hw; i <= hw; i++) {
                const ix = Math.round(cx + perpX * i);
                const iy = Math.round(cy + perpY * i);
                if (ix >= 0 && ix < N && iy >= 0 && iy < N) {
                    grid[iy * N + ix] = 1.0;
                }
            }
        }
    }

    // ==================== TRAIL TRACKING (self-built, with extrapolation) ====================
    function updateTrailTracking() {
        /**
         * Track player positions each rAF tick and draw trail lines on persistent grids.
         * c.particles is always empty (cosmetic effects only), so we build our own
         * trail data by recording positions — matching how the env's trail_owner works.
         *
         * EXTRAPOLATION: Server sends position updates at ~3-10Hz, but rAF runs at 60fps.
         * Between server updates, we extrapolate forward using heading + speed so the trail
         * density matches the training env (~60fps physics). Without this, trails are 10-20x
         * too sparse and the model sees almost no trail data.
         */
        const round = pageWindow.gameGraphics?.gameEngine?.round;
        if (!round) return;
        const curves = round.getCurves();
        const me = curves.find(c => c.player?.isMyPlayer);
        if (!me) return;

        if (!S.selfTrail || !S.enemyTrail) return;

        const N = CFG.RES;
        const hw = 1; // trail half-width at obs res (~1px, matching env's .any() downsample)
        const now = performance.now();
        const gs = round.gameSettings;
        const fps = gs?.fps || 60;

        for (const c of curves) {
            if (!c.state?.isAlive) continue;
            const cid = c.curveId;
            const isSelf = (cid === me.curveId);
            const grid = isSelf ? S.selfTrail : S.enemyTrail;

            const serverX = c.state.x;
            const serverY = c.state.y;
            const angle = c.state.angle;
            // c.state.speed is in field px per game frame (base = curveSpeed/fps ≈ 3)
            const speed = c.state.speed || ((gs?.curveSpeed || 180) / fps);

            const holeLeft = c.state.holeLeft;
            const inHole = (typeof holeLeft === 'number' && holeLeft > 0);

            // First time seeing this curve — initialize, don't draw yet
            if (!S.prevTrailPos[cid]) {
                S.prevTrailPos[cid] = {
                    fieldX: serverX,
                    fieldY: serverY,
                    serverX: serverX,
                    serverY: serverY,
                    lastTs: now,
                };
                continue;
            }

            const prev = S.prevTrailPos[cid];
            const serverMoved = (serverX !== prev.serverX || serverY !== prev.serverY);

            let newFieldX, newFieldY;

            if (serverMoved) {
                // Server sent a new position — snap to authoritative coords
                newFieldX = serverX;
                newFieldY = serverY;
                prev.serverX = serverX;
                prev.serverY = serverY;
            } else {
                // No server update — extrapolate forward using heading + speed
                const dtSec = Math.min((now - prev.lastTs) / 1000, 0.05); // cap at 50ms
                const fieldSpeedPerSec = speed * fps; // e.g. 3 * 60 = 180 field px/s
                newFieldX = prev.fieldX + Math.cos(angle) * fieldSpeedPerSec * dtSec;
                newFieldY = prev.fieldY + Math.sin(angle) * fieldSpeedPerSec * dtSec;
            }

            // Convert field coords to obs coords for trail drawing
            const prevOX = prev.fieldX * OBS_SCALE;
            const prevOY = prev.fieldY * OBS_SCALE + OBS_OFFSET_Y;
            const newOX = newFieldX * OBS_SCALE;
            const newOY = newFieldY * OBS_SCALE + OBS_OFFSET_Y;

            if (!inHole) {
                rasterizeTrailLine(grid, prevOX, prevOY, newOX, newOY, hw, N);
                // Head circle at current position (like env's _draw_trail head stamp)
                const ix = Math.round(newOX), iy = Math.round(newOY);
                for (let dy = -hw; dy <= hw; dy++) {
                    for (let dx = -hw; dx <= hw; dx++) {
                        if (dx * dx + dy * dy <= hw * hw) {
                            const cx = ix + dx, cy = iy + dy;
                            if (cx >= 0 && cx < N && cy >= 0 && cy < N) {
                                grid[cy * N + cx] = 1.0;
                            }
                        }
                    }
                }
            }

            prev.fieldX = newFieldX;
            prev.fieldY = newFieldY;
            prev.lastTs = now;
        }
    }

    // ==================== TRAIL CHANNELS FROM GAME STATE ====================
    function buildChannelsFromGameState() {
        /**
         * Build self/enemy trail channels from self-tracked trail data + walls + arrows.
         * Trail data comes from persistent grids (updated by updateTrailTracking()),
         * NOT from c.particles (which is always empty — cosmetic effects only).
         */
        const N = CFG.RES;
        const NN = N * N;

        // Start with persistent trail grids (accumulated over the round)
        const selfCh = new Float32Array(NN);
        const enemyCh = new Float32Array(NN);
        if (S.selfTrail) selfCh.set(S.selfTrail);
        if (S.enemyTrail) enemyCh.set(S.enemyTrail);

        const round = pageWindow.gameGraphics?.gameEngine?.round;
        if (!round) return { self: selfCh, enemy: enemyCh };

        const curves = round.getCurves();
        const me = curves.find(c => c.player?.isMyPlayer);
        if (!me) return { self: selfCh, enemy: enemyCh };

        // Wall mask matching env (curvecrash_env_ffa.py lines 223-237)
        // top_ds=11, bot_ds=117 → rows 0-11 and 116-127 are wall
        const top_ds = Math.floor(ARENA_OFFSET_Y / DS_FACTOR); // 11
        const bot_ds = Math.min(N, Math.floor((ARENA_OFFSET_Y + ARENA_H_GSPP + DS_FACTOR - 1) / DS_FACTOR)); // 117
        // Top wall band (rows 0 to top_ds-1)
        for (let row = 0; row < top_ds; row++) {
            for (let col = 0; col < N; col++) {
                selfCh[row * N + col] = 1.0;
            }
        }
        // Bottom wall band (rows bot_ds to N-1)
        for (let row = bot_ds; row < N; row++) {
            for (let col = 0; col < N; col++) {
                selfCh[row * N + col] = 1.0;
            }
        }
        // 1-pixel border at field edges
        if (top_ds >= 0 && top_ds < N) {
            for (let col = 0; col < N; col++) selfCh[top_ds * N + col] = 1.0; // row 11
        }
        if (bot_ds > 0 && bot_ds <= N) {
            for (let col = 0; col < N; col++) selfCh[(bot_ds - 1) * N + col] = 1.0; // row 116
        }
        // Left/right columns within playable area
        for (let row = top_ds; row < bot_ds; row++) {
            selfCh[row * N] = 1.0;           // left col
            selfCh[row * N + N - 1] = 1.0;   // right col
        }

        // Direction arrows for all alive players (5px in obs space, like env)
        for (const c of curves) {
            if (!c.state?.isAlive) continue;
            const isSelf = (c.curveId === me.curveId);
            const ch = isSelf ? selfCh : enemyCh;
            const px = c.state.x * OBS_SCALE;
            const py = c.state.y * OBS_SCALE + OBS_OFFSET_Y;
            const cos_a = Math.cos(c.state.angle);
            const sin_a = Math.sin(c.state.angle);
            for (let t = 0; t < 5; t++) {
                const ax = Math.round(px + cos_a * t);
                const ay = Math.round(py + sin_a * t);
                if (ax >= 0 && ax < N && ay >= 0 && ay < N) {
                    ch[ay * N + ax] = 1.0;
                }
            }
        }

        return { self: selfCh, enemy: enemyCh };
    }

    // ==================== EGO-CENTRIC ROTATION ====================
    function rotateChannel(ch, headX, headY, heading, fill) {
        const N = CFG.RES;
        const half = N / 2;
        const cos_a = Math.cos(heading);
        const sin_a = Math.sin(heading);
        const out = new Float32Array(N * N);

        for (let row = 0; row < N; row++) {
            const dy = row - half + 0.5;
            for (let col = 0; col < N; col++) {
                const dx = col - half + 0.5;
                const srcCol = Math.round(headX + cos_a * dx - sin_a * dy);
                const srcRow = Math.round(headY + sin_a * dx + cos_a * dy);

                if (srcCol >= 0 && srcCol < N && srcRow >= 0 && srcRow < N) {
                    out[row * N + col] = ch[srcRow * N + srcCol];
                } else {
                    out[row * N + col] = fill;
                }
            }
        }
        return out;
    }

    // ==================== POWERUP CHANNEL RENDERING ====================
    function renderPowerupChannels(headX, headY, heading) {
        /**
         * Render powerup locations into 2 channels (speed, erase) at observation resolution.
         * Uses game state positions, maps to observation coords, then applies ego-centric rotation.
         */
        const N = CFG.RES;
        const speedCh = new Float32Array(N * N);
        const eraseCh = new Float32Array(N * N);

        if (S.gsPowerups.length === 0) {
            // No powerups → both channels are zeros everywhere (rotation of zeros with fill=0 is zeros)
            return { speed: speedCh, erase: eraseCh };
        }

        // Powerup radius in observation pixels
        const radius = Math.max(1, Math.round(N * CFG.POWERUP_RADIUS_FRAC));

        // Map field coordinates to observation coordinates (uniform scale + Y offset)
        for (const pup of S.gsPowerups) {
            const px = Math.round(pup.x * OBS_SCALE);
            const py = Math.round(pup.y * OBS_SCALE + OBS_OFFSET_Y);
            // powerupId=1 is GREEN_SPEED, powerupId=9 is ERASER
            const ch = (pup.powerupId === 1) ? speedCh : eraseCh;

            for (let dy = -radius; dy <= radius; dy++) {
                for (let dx = -radius; dx <= radius; dx++) {
                    if (dx * dx + dy * dy <= radius * radius) {
                        const cx = px + dx;
                        const cy = py + dy;
                        if (cx >= 0 && cx < N && cy >= 0 && cy < N) {
                            ch[cy * N + cx] = 1.0;
                        }
                    }
                }
            }
        }

        return {
            speed: rotateChannel(speedCh, headX, headY, heading, 0.0),
            erase: rotateChannel(eraseCh, headX, headY, heading, 0.0)
        };
    }

    // ==================== VORONOI TERRITORY (BFS flood fill) ====================
    function computeVoronoi(selfCh, enemyCh, headX, headY, heading) {
        /**
         * Simultaneous BFS flood fill from ego and all enemies.
         * Returns ego-centric rotated voronoi channel:
         *   1.0 = cells ego reaches first, 0.0 = enemy territory or blocked.
         * Matches VoronoiWrapper._compute_voronoi_ds() from experiments.py.
         */
        const N = CFG.RES;
        const NN = N * N;

        // Blocked = any trail pixel or wall pixel
        const blocked = new Uint8Array(NN);
        for (let i = 0; i < NN; i++) {
            if (selfCh[i] > 0.5 || enemyCh[i] > 0.5) blocked[i] = 1;
        }

        // territory: 0=unclaimed, 1=ego, 2=enemy
        const territory = new Uint8Array(NN);

        // Seed BFS from ego position
        const egoIdx = Math.round(headY) * N + Math.round(headX);
        let egoQueue = [];
        if (egoIdx >= 0 && egoIdx < NN && !blocked[egoIdx]) {
            territory[egoIdx] = 1;
            blocked[egoIdx] = 1;
            egoQueue.push(egoIdx);
        }

        // Seed BFS from all enemy positions
        let enemyQueue = [];
        const round = pageWindow.gameGraphics?.gameEngine?.round;
        if (round) {
            const curves = round.getCurves();
            const me = curves.find(c => c.player?.isMyPlayer);
            for (const c of curves) {
                if (!c.state?.isAlive || c === me) continue;
                const ex = Math.round(c.state.x * OBS_SCALE);
                const ey = Math.round(c.state.y * OBS_SCALE + OBS_OFFSET_Y);
                const ei = ey * N + ex;
                if (ei >= 0 && ei < NN && !blocked[ei]) {
                    territory[ei] = 2;
                    blocked[ei] = 1;
                    enemyQueue.push(ei);
                }
            }
        }

        // Simultaneous BFS: expand both one step at a time
        const dx = [-1, 1, -N, N]; // left, right, up, down
        while (egoQueue.length > 0 || enemyQueue.length > 0) {
            const nextEgo = [];
            for (const idx of egoQueue) {
                const col = idx % N, row = (idx - col) / N;
                if (col > 0     && !blocked[idx - 1]) { blocked[idx - 1] = 1; territory[idx - 1] = 1; nextEgo.push(idx - 1); }
                if (col < N - 1 && !blocked[idx + 1]) { blocked[idx + 1] = 1; territory[idx + 1] = 1; nextEgo.push(idx + 1); }
                if (row > 0     && !blocked[idx - N]) { blocked[idx - N] = 1; territory[idx - N] = 1; nextEgo.push(idx - N); }
                if (row < N - 1 && !blocked[idx + N]) { blocked[idx + N] = 1; territory[idx + N] = 1; nextEgo.push(idx + N); }
            }
            const nextEnemy = [];
            for (const idx of enemyQueue) {
                const col = idx % N, row = (idx - col) / N;
                if (col > 0     && !blocked[idx - 1]) { blocked[idx - 1] = 1; territory[idx - 1] = 2; nextEnemy.push(idx - 1); }
                if (col < N - 1 && !blocked[idx + 1]) { blocked[idx + 1] = 1; territory[idx + 1] = 2; nextEnemy.push(idx + 1); }
                if (row > 0     && !blocked[idx - N]) { blocked[idx - N] = 1; territory[idx - N] = 2; nextEnemy.push(idx - N); }
                if (row < N - 1 && !blocked[idx + N]) { blocked[idx + N] = 1; territory[idx + N] = 2; nextEnemy.push(idx + N); }
            }
            egoQueue = nextEgo;
            enemyQueue = nextEnemy;
        }

        // Binary mask: ego territory = 1.0
        const voronoi = new Float32Array(NN);
        for (let i = 0; i < NN; i++) {
            if (territory[i] === 1) voronoi[i] = 1.0;
        }

        // Rotate to ego-centric frame
        return rotateChannel(voronoi, headX, headY, heading, 0.0);
    }

    // ==================== OBSERVATION BUILDING (6ch or 7ch) ====================
    function buildObservation(selfCh, enemyCh, headX, headY, heading) {
        const rSelf = rotateChannel(selfCh, headX, headY, heading, 1.0);
        const rEnemy = rotateChannel(enemyCh, headX, headY, heading, 0.0);

        let rPrevSelf, rPrevEnemy;
        if (S.prevSelf) {
            rPrevSelf = rotateChannel(S.prevSelf, headX, headY, heading, 1.0);
            rPrevEnemy = rotateChannel(S.prevEnemy, headX, headY, heading, 0.0);
        } else {
            const zeros = new Float32Array(CFG.RES * CFG.RES);
            rPrevSelf = rotateChannel(zeros, headX, headY, heading, 1.0);
            rPrevEnemy = rotateChannel(zeros, headX, headY, heading, 0.0);
        }

        // Powerup channels from game state
        const pups = renderPowerupChannels(headX, headY, heading);

        // Determine channel count from loaded model
        const nCh = S.model ? S.model.nInputChannels : 6;
        const NN = CFG.RES * CFG.RES;
        const obs = new Float32Array(NN * nCh);

        // Compute Voronoi if model needs 7 channels
        let rVoronoi = null;
        if (nCh >= 7) {
            rVoronoi = computeVoronoi(selfCh, enemyCh, headX, headY, heading);
        }

        for (let i = 0; i < NN; i++) {
            obs[i * nCh + 0] = rSelf[i];
            obs[i * nCh + 1] = rEnemy[i];
            obs[i * nCh + 2] = rPrevSelf[i];
            obs[i * nCh + 3] = rPrevEnemy[i];
            obs[i * nCh + 4] = pups.speed[i];
            obs[i * nCh + 5] = pups.erase[i];
            if (nCh >= 7) obs[i * nCh + 6] = rVoronoi[i];
        }

        S.prevSelf = selfCh;
        S.prevEnemy = enemyCh;
        return obs;
    }

    // ==================== BASE64 WEIGHT DECODING ====================
    function _b64toF32(b64, shape) {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        const dv = new DataView(bytes.buffer);
        const totalElems = shape.reduce((a, b) => a * b, 1);
        let f;
        if (bytes.length === totalElems * 2) {
            // Float16 → Float32
            f = new Float32Array(totalElems);
            for (let i = 0; i < totalElems; i++) {
                const h = dv.getUint16(i * 2, true);
                const s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
                if (e === 0) f[i] = (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
                else if (e === 31) f[i] = m ? NaN : (s ? -Infinity : Infinity);
                else f[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
            }
        } else {
            f = new Float32Array(bytes.buffer);
        }
        if (shape.length === 1) return tf.tensor1d(f);
        if (shape.length === 2) return tf.tensor2d(f, shape);
        if (shape.length === 4) return tf.tensor4d(f, shape);
        return tf.tensor(f, shape);
    }

    function _decW(layer, key) {
        const v = layer[key];
        if (typeof v === 'string' && layer[key + '_shape']) return _b64toF32(v, layer[key + '_shape']);
        if (Array.isArray(v)) {
            if (v.length > 0 && Array.isArray(v[0])) {
                if (Array.isArray(v[0][0])) { if (Array.isArray(v[0][0][0])) return tf.tensor4d(v); return tf.tensor3d(v); }
                return tf.tensor2d(v);
            }
            return tf.tensor1d(v);
        }
        return null;
    }

    function _decGate(g) {
        return {
            Wi: typeof g.kernel_input === 'string' ? _b64toF32(g.kernel_input, g.kernel_input_shape) : tf.tensor2d(g.kernel_input),
            Wh: typeof g.kernel_hidden === 'string' ? _b64toF32(g.kernel_hidden, g.kernel_hidden_shape) : tf.tensor2d(g.kernel_hidden),
            bi: typeof g.bias_input === 'string' ? _b64toF32(g.bias_input, g.bias_input_shape) : tf.tensor1d(g.bias_input),
            bh: typeof g.bias_hidden === 'string' ? _b64toF32(g.bias_hidden, g.bias_hidden_shape) : tf.tensor1d(g.bias_hidden),
        };
    }

    // ==================== MODEL INFERENCE (v6 + IMPALA) ====================
    function buildTFModel(weightsJson) {
        const model = { version: weightsJson.version || 'v6' };
        const nCh = weightsJson.n_input_channels || 6;
        model.nInputChannels = nCh;
        model.arch = weightsJson.arch || 'naturecnn';
        const layers = weightsJson.layers;
        let idx = 0;

        if (model.arch === 'impala') {
            const numStages = weightsJson.impala_channels ? weightsJson.impala_channels.length : 3;
            model.impalaStages = [];
            for (let si = 0; si < numStages; si++) {
                const l = layers[idx++];
                model.impalaStages.push({
                    convW: _decW(l,'conv_weight'), convB: _decW(l,'conv_bias'),
                    res1c1W: _decW(l,'res1_conv1_weight'), res1c1B: _decW(l,'res1_conv1_bias'),
                    res1c2W: _decW(l,'res1_conv2_weight'), res1c2B: _decW(l,'res1_conv2_bias'),
                    res2c1W: _decW(l,'res2_conv1_weight'), res2c1B: _decW(l,'res2_conv1_bias'),
                    res2c2W: _decW(l,'res2_conv2_weight'), res2c2B: _decW(l,'res2_conv2_bias'),
                });
            }
        } else {
            model.conv = [];
            for (let i = 0; i < 3; i++) {
                const l = layers[idx++];
                model.conv.push({ w: _decW(l,'weight'), b: _decW(l,'bias'), stride: l.stride });
            }
        }

        model.hasCBAM = weightsJson.has_cbam;
        if (model.hasCBAM) {
            const l = layers[idx++];
            model.cbam = { fc0w: _decW(l,'channel_fc0_weight'), fc2w: _decW(l,'channel_fc2_weight'), spConvW: _decW(l,'spatial_conv_weight') };
        }

        model.hasSpatialAttn = weightsJson.has_spatial_attn;
        if (model.hasSpatialAttn) {
            const l = layers[idx++];
            model.spatialAttn = { qkvW: _decW(l,'qkv_weight'), projW: _decW(l,'proj_weight'), projB: _decW(l,'proj_bias'), numHeads: l.num_heads, dim: l.dim, tokens: l.tokens };
        }

        const fcL = layers[idx++];
        model.fc = { w: _decW(fcL,'weight'), b: _decW(fcL,'bias') };

        model.hasGRU = weightsJson.gru_hidden > 0;
        if (model.hasGRU) {
            const gruL = layers[idx++];
            model.gruHidden = gruL.units;
            model.gru = {};
            for (const gate of ['reset', 'update', 'new']) model.gru[gate] = _decGate(gruL.gates[gate]);
        }

        const actL = layers[idx++];
        model.actor = { w: _decW(actL,'weight'), b: _decW(actL,'bias') };

        return model;
    }

    function infer(obsNhwc, step) {
        if (!S.model) return 1;
        const nCh = S.model.nInputChannels || 6;
        const t0 = performance.now();
        const m = S.model;

        // GRU state lives on GPU — avoids one dataSync per frame
        let newGruTensor = null;

        const action = tf.tidy(() => {
            let x = tf.tensor4d(obsNhwc, [1, CFG.RES, CFG.RES, nCh]);

            if (m.arch === 'impala') {
                // IMPALA ConvSequence stages
                for (const stage of m.impalaStages) {
                    // Main conv (same padding) → maxpool(3,s2,same)
                    x = tf.conv2d(x, stage.convW, 1, 'same').add(stage.convB);
                    x = tf.maxPool(x, 3, 2, 'same');
                    // ResBlock 1: relu → conv(same) → relu → conv(same) → +residual
                    let res = x;
                    x = tf.relu(x);
                    x = tf.conv2d(x, stage.res1c1W, 1, 'same').add(stage.res1c1B);
                    x = tf.relu(x);
                    x = tf.conv2d(x, stage.res1c2W, 1, 'same').add(stage.res1c2B);
                    x = x.add(res);
                    // ResBlock 2
                    res = x;
                    x = tf.relu(x);
                    x = tf.conv2d(x, stage.res2c1W, 1, 'same').add(stage.res2c1B);
                    x = tf.relu(x);
                    x = tf.conv2d(x, stage.res2c2W, 1, 'same').add(stage.res2c2B);
                    x = x.add(res);
                }
                // Final ReLU after all stages
                x = tf.relu(x);
            } else {
                // NatureCNN: 3 Conv layers
                for (let i = 0; i < 3; i++) {
                    x = tf.relu(tf.conv2d(x, m.conv[i].w, m.conv[i].stride, 'valid').add(m.conv[i].b));
                }
            }

            // CBAM
            if (m.hasCBAM) {
                const avgFlat = x.mean([1, 2], true).reshape([1, -1]);
                const maxFlat = x.max([1, 2], true).reshape([1, -1]);
                const avgFc = tf.relu(tf.matMul(avgFlat, m.cbam.fc0w)).matMul(m.cbam.fc2w);
                const maxFc = tf.relu(tf.matMul(maxFlat, m.cbam.fc0w)).matMul(m.cbam.fc2w);
                x = x.mul(tf.sigmoid(avgFc.add(maxFc)).reshape([1, 1, 1, -1]));

                const spCat = tf.concat([x.mean(3, true), x.max(3, true)], 3);
                x = x.mul(tf.sigmoid(tf.conv2d(spCat, m.cbam.spConvW, 1, 'same')));
            }

            // Spatial self-attention
            if (m.hasSpatialAttn) {
                const sa = m.spatialAttn;
                const [B, H, W, C] = x.shape;
                const numTokens = H * W;
                const headDim = sa.dim / sa.numHeads;
                const scale = 1.0 / Math.sqrt(headDim);

                const tokens = x.reshape([1, numTokens, C]);
                const qkv = tf.matMul(tokens, sa.qkvW);
                const qkvR = qkv.reshape([1, numTokens, 3, sa.numHeads, headDim]);
                const qkvT = qkvR.transpose([2, 0, 3, 1, 4]);
                const q = qkvT.gather(0).reshape([1, sa.numHeads, numTokens, headDim]);
                const k = qkvT.gather(1).reshape([1, sa.numHeads, numTokens, headDim]);
                const v = qkvT.gather(2).reshape([1, sa.numHeads, numTokens, headDim]);

                const attnWeights = tf.softmax(tf.matMul(q, k.transpose([0, 1, 3, 2])).mul(scale), -1);
                const attnOut = tf.matMul(attnWeights, v);
                const projected = tf.matMul(
                    attnOut.transpose([0, 2, 1, 3]).reshape([1, numTokens, C]),
                    sa.projW
                ).add(sa.projB);
                x = tokens.add(projected).reshape([1, H, W, C]);
            }

            // Flatten + FC hidden
            x = tf.relu(tf.matMul(x.reshape([1, -1]), m.fc.w).add(m.fc.b));

            // GRU step — hidden state stays on GPU (no upload/download per frame)
            if (m.hasGRU) {
                const hPrev = S.gruTensor || tf.zeros([1, m.gruHidden]);
                const rGate = m.gru.reset;
                const zGate = m.gru.update;
                const nGate = m.gru.new;

                const r = tf.sigmoid(
                    tf.matMul(x, rGate.Wi).add(rGate.bi)
                      .add(tf.matMul(hPrev, rGate.Wh).add(rGate.bh))
                );
                const z = tf.sigmoid(
                    tf.matMul(x, zGate.Wi).add(zGate.bi)
                      .add(tf.matMul(hPrev, zGate.Wh).add(zGate.bh))
                );
                const n = tf.tanh(
                    tf.matMul(x, nGate.Wi).add(nGate.bi)
                      .add(r.mul(tf.matMul(hPrev, nGate.Wh).add(nGate.bh)))
                );
                const hNew = tf.sub(tf.scalar(1), z).mul(n).add(z.mul(hPrev));
                newGruTensor = tf.keep(hNew);
                x = hNew;
            }

            // Single dataSync — blocking but necessary for real-time control
            const logits = tf.matMul(x, m.actor.w).add(m.actor.b);
            const l = logits.dataSync();
            const act = l[0] >= l[1] && l[0] >= l[2] ? 0 : (l[1] >= l[2] ? 1 : 2);

            const elapsed = performance.now() - t0;
            S.inferMs = elapsed;
            S.inferAvg = S.inferAvg > 0 ? S.inferAvg * 0.9 + elapsed * 0.1 : elapsed;

            if (step <= 5 || step % 30 === 0) {
                console.log(`[AI] logits step=${step}: L=${l[0].toFixed(3)} S=${l[1].toFixed(3)} R=${l[2].toFixed(3)} (${elapsed.toFixed(0)}ms, avg=${S.inferAvg.toFixed(0)}ms)`);
            }
            return act;
        });

        // Update GPU-resident GRU state (dispose old)
        if (newGruTensor) {
            if (S.gruTensor) S.gruTensor.dispose();
            S.gruTensor = newGruTensor;
        }

        return action;
    }

    // ==================== ACTION EXECUTION ====================
    function press(key) {
        if (S.pressed === key) return;
        release();
        const code = key === 'ArrowLeft' ? 37 : 39;
        document.dispatchEvent(new KeyboardEvent('keydown', {
            key, code: key, keyCode: code, which: code, bubbles: true
        }));
        S.pressed = key;
    }

    function release() {
        if (!S.pressed) return;
        const key = S.pressed;
        const code = key === 'ArrowLeft' ? 37 : 39;
        document.dispatchEvent(new KeyboardEvent('keyup', {
            key, code: key, keyCode: code, which: code, bubbles: true
        }));
        S.pressed = null;
    }

    function executeAction(action) {
        if (action === 0) press('ArrowLeft');
        else if (action === 2) press('ArrowRight');
        else release();
    }

    // ==================== ROUND DETECTION (game state + pixel fallback) ====================
    function detectRoundActive() {
        // Primary: use game state
        if (S.gsAvailable) {
            return S.gsAlive;
        }

        // Fallback: pixel-based colored pixel count
        return null; // caller uses pixel method
    }

    function detectRoundFromPixels(imgData) {
        const d = imgData.data;
        let colored = 0;
        const N = CFG.RES * CFG.RES;
        for (let i = 0; i < N; i++) {
            const r = d[i*4], g = d[i*4+1], b = d[i*4+2];
            if (!isBG(r, g, b)) colored++;
        }
        return colored;
    }

    // ==================== MAIN DECISION TICK ====================
    let lastDecTs = 0;
    function decisionTick() {
        const now = performance.now();
        if (now - lastDecTs < CFG.DEC_MS) return;
        lastDecTs = now;

        // Read game state (always try — it's cheap)
        readGameState();

        if (!S.gsAvailable) return; // No game state → can't do anything

        const canStart = now >= S.cooldownUntil;
        const roundSignal = S.gsAlive;

        if (!S._debugCount) S._debugCount = 0;
        S._debugCount++;
        if (S._debugCount <= 5 || S._debugCount % 200 === 0) {
            console.log(`[AI] tick ${S._debugCount}: gs=${S.gsAvailable} alive=${S.gsAlive} active=${S.roundActive} pups=${S.gsPowerups.length}`);
        }

        // Round start
        if (!S.roundActive && canStart && roundSignal) {
            S.roundActive = true;
            S.roundStart = now;
            S.idleFrames = 0;
            S.steps = 0;
            S.prevSelf = null;
            S.prevEnemy = null;
            // Reset trail tracking grids for new round
            const NN = CFG.RES * CFG.RES;
            S.selfTrail = new Float32Array(NN);
            S.enemyTrail = new Float32Array(NN);
            S.prevTrailPos = {};
            if (S.model && S.model.hasGRU) {
                S.gruState = new Float32Array(S.model.gruHidden);
                if (S.gruTensor) { S.gruTensor.dispose(); S.gruTensor = null; }
            }
            S.inferAvg = 0;
            S.ep++;
            console.log(`[AI] Round ${S.ep} started (gs=${S.gsAvailable})`);
        }

        if (!S.roundActive) return;

        // Round end detection
        if (!S.gsAlive) {
            const duration = (now - S.roundStart) / 1000;
            if (duration >= CFG.MIN_ROUND_SEC) {
                console.log(`[AI] Round ${S.ep} ended: ${duration.toFixed(1)}s, ${S.steps} steps (died)`);
                S.roundActive = false;
                S.cooldownUntil = now + CFG.COOLDOWN_MS;
                release();
                return;
            }
        }

        // Trail tracking runs in main loop at rAF rate (updateTrailTracking)
        // Build trail channels (persistent trails + walls + arrows)
        let selfCh, enemyCh;
        ({ self: selfCh, enemy: enemyCh } = buildChannelsFromGameState());

        // Position and heading from game state (uniform scale + Y offset, matching env)
        const headX = S.gsPosition.x * OBS_SCALE;
        const headY = S.gsPosition.y * OBS_SCALE + OBS_OFFSET_Y;
        const heading = S.gsAngle;

        S.idleFrames = 0;
        S.steps++;

        // Build 6-channel observation
        const obs = buildObservation(selfCh, enemyCh, headX, headY, heading);
        if (!obs) return;

        if (S.steps <= 3 || S.steps % 30 === 0) {
            const N = CFG.RES * CFG.RES;
            let ch0sum = 0, ch1sum = 0, ch2sum = 0, ch3sum = 0, ch4sum = 0, ch5sum = 0;
            for (let i = 0; i < N; i++) {
                ch0sum += obs[i * 6 + 0];
                ch1sum += obs[i * 6 + 1];
                ch2sum += obs[i * 6 + 2];
                ch3sum += obs[i * 6 + 3];
                ch4sum += obs[i * 6 + 4];
                ch5sum += obs[i * 6 + 5];
            }
            // Also count pre-rotation (world frame) channel sums
            let wSelf = 0, wEnemy = 0;
            for (let i = 0; i < N; i++) {
                wSelf += selfCh[i];
                wEnemy += enemyCh[i];
            }
            console.log(`[AI] Obs step=${S.steps}: self=${ch0sum.toFixed(0)} enemy=${ch1sum.toFixed(0)} prevS=${ch2sum.toFixed(0)} prevE=${ch3sum.toFixed(0)} spd=${ch4sum.toFixed(0)} ers=${ch5sum.toFixed(0)} | world: self=${wSelf.toFixed(0)} enemy=${wEnemy.toFixed(0)} | head=(${headX.toFixed(1)},${headY.toFixed(1)}) hdg=${(heading*180/Math.PI).toFixed(0)}deg`);
        }

        const action = infer(obs, S.steps);
        S.action = action;
        executeAction(action);

        if (S.steps <= 5 || S.steps % 30 === 0) {
            const aName = ['LEFT','STRAIGHT','RIGHT'][S.action];
            const elapsed = ((now - S.roundStart)/1000).toFixed(2);
            const spdStr = S.gsSpeedMult > 1.01 ? ` spd=${S.gsSpeedMult.toFixed(1)}x` : '';
            console.log(`[AI] step=${S.steps} t=${elapsed}s act=${aName} pos=(${headX.toFixed(1)},${headY.toFixed(1)}) hdg=${(heading*180/Math.PI).toFixed(0)}deg pups=${S.gsPowerups.length}${spdStr}`);
        }

        updateViz(selfCh, enemyCh);
    }

    // ==================== DEBUG VISUALIZATION ====================
    function updateViz(selfCh, enemyCh) {
        if (!S.vizCtx) return;
        const N = CFG.RES;
        const id = S.vizCtx.createImageData(N, N);
        const NN = N * N;
        for (let i = 0; i < NN; i++) {
            id.data[i*4+0] = enemyCh[i] * 255;
            id.data[i*4+1] = selfCh[i] * 200;
            id.data[i*4+2] = 0;
            id.data[i*4+3] = 255;
        }

        // Draw powerup locations (unrotated, world frame)
        if (S.gsPowerups.length > 0) {
            for (const pup of S.gsPowerups) {
                const px = Math.round(pup.x * OBS_SCALE);
                const py = Math.round(pup.y * OBS_SCALE + OBS_OFFSET_Y);
                const isSpeed = pup.powerupId === 1;
                for (let dy = -2; dy <= 2; dy++) {
                    for (let dx = -2; dx <= 2; dx++) {
                        if (dx*dx + dy*dy > 4) continue;
                        const cx = px + dx, cy = py + dy;
                        if (cx >= 0 && cx < N && cy >= 0 && cy < N) {
                            const idx = (cy * N + cx) * 4;
                            if (isSpeed) {
                                id.data[idx] = 0; id.data[idx+1] = 255; id.data[idx+2] = 0;
                            } else {
                                id.data[idx] = 100; id.data[idx+1] = 100; id.data[idx+2] = 255;
                            }
                        }
                    }
                }
            }
        }

        // Draw head position from game state
        if (S.gsAvailable && S.gsPosition) {
            const hx = Math.round(S.gsPosition.x * OBS_SCALE);
            const hy = Math.round(S.gsPosition.y * OBS_SCALE + OBS_OFFSET_Y);
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const px = hx + dx, py = hy + dy;
                    if (px >= 0 && px < N && py >= 0 && py < N) {
                        const idx = (py * N + px) * 4;
                        id.data[idx] = 255; id.data[idx+1] = 255; id.data[idx+2] = 0;
                    }
                }
            }
        }

        S.vizCtx.putImageData(id, 0, 0);
    }

    // ==================== UI ====================
    function createUI() {
        const div = document.createElement('div');
        div.style.cssText = 'position:fixed;top:10px;right:10px;z-index:999999;background:#000;border:2px solid #0f0;color:#0f0;font:12px monospace;padding:8px;width:310px';
        div.innerHTML = `
            <div style="font-weight:bold;color:#0ff;margin-bottom:4px">CurveCrash AI v6 (GS++ Hybrid 836K)</div>
            <div style="display:flex;gap:6px;margin-bottom:8px">
                <button id="ai-toggle" style="flex:1;padding:6px;background:#06a;color:#fff;font-weight:bold">ENABLE</button>
                <button id="ai-loadw" style="flex:1;padding:6px;background:#0a0;color:#fff;font-weight:bold">LOAD WEIGHTS</button>
            </div>
            <canvas id="ai-viz" width="${CFG.RES}" height="${CFG.RES}" style="width:160px;height:160px;border:1px solid #0ff;image-rendering:pixelated;background:#000;display:block"></canvas>
            <div id="ai-info" style="margin-top:6px;font-size:11px;color:#ccc;line-height:1.4"></div>
        `;
        document.body.appendChild(div);

        S.vizCtx = div.querySelector('#ai-viz').getContext('2d', { willReadFrequently: true });

        div.querySelector('#ai-toggle').onclick = function() {
            S.enabled = !S.enabled;
            this.textContent = S.enabled ? 'DISABLE' : 'ENABLE';
            this.style.background = S.enabled ? '#a00' : '#06a';
            if (!S.enabled) release();
        };

        div.querySelector('#ai-loadw').onclick = loadWeightsFromFile;

        S.infoDiv = div.querySelector('#ai-info');

        setInterval(() => {
            if (!S.infoDiv) return;
            const actionStr = ['LEFT', 'STRAIGHT', 'RIGHT'][S.action] || '?';
            const colorStr = S.myColor ? `rgb(${S.myColor.join(',')})` : 'detecting...';
            const archStr = S.model ? (S.model.arch || S.model.version) : '';
            const modelStr = S.model ? `loaded (${archStr}, ${S.model.nInputChannels}ch)` : 'NOT LOADED';
            const gruStr = S.model && S.model.hasGRU ? `GRU(${S.model.gruHidden})` : 'none';
            const gsStr = S.gsAvailable ? '<span style="color:#0f0">LIVE</span>' : '<span style="color:#f80">waiting...</span>';
            const posStr = S.gsPosition ?
                `(${S.gsPosition.x.toFixed(0)}, ${S.gsPosition.y.toFixed(0)})` : 'N/A';
            const hdgStr = S.gsAngle !== null ?
                `${(S.gsAngle * 180 / Math.PI).toFixed(0)}&deg;` : 'N/A';
            const spdStr = S.gsSpeedMult > 1.01 ?
                `<span style="color:#4CAF50">${S.gsSpeedMult.toFixed(1)}x</span>` : '1x';

            S.infoDiv.innerHTML = `
                <div>Model: <span style="color:${S.model ? '#0f0' : '#f00'}">${modelStr}</span></div>
                <div>Memory: ${gruStr} | CBAM: ${S.model?.hasCBAM ? 'ON' : 'off'} | Attn: ${S.model?.hasSpatialAttn ? 'ON' : 'off'}</div>
                <div>Game State: ${gsStr} | Powerups: ${S.gsPowerups.length}</div>
                <div>Color: <span style="color:${S.myColor ? `rgb(${S.myColor.join(',')})` : '#888'}">${colorStr}</span></div>
                <div>Position: ${posStr} | Heading: ${hdgStr}</div>
                <div>Speed: ${spdStr}</div>
                <div>Round: ${S.ep} | Steps: ${S.steps}</div>
                <div>Action: <span style="color:#ff0;font-weight:bold">${actionStr}</span></div>
                <div>Infer: <span style="color:${S.inferAvg > 80 ? '#f44' : S.inferAvg > 40 ? '#fa0' : '#0f0'}">${S.inferMs.toFixed(0)}ms</span> (avg ${S.inferAvg.toFixed(0)}ms) | ${S.inferAvg > 0 ? (1000/S.inferAvg).toFixed(0) : '?'} fps</div>
                <div>Active: ${S.roundActive ? 'YES' : 'no'} | Enabled: ${S.enabled ? 'YES' : 'no'}</div>
            `;
        }, 200);
    }

    // ==================== WEIGHT LOADING ====================
    function loadWeightsFromFile() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            try {
                const text = await file.text();
                const json = JSON.parse(text);
                if (!json.layers || json.layers.length < 5) {
                    console.error('[AI] Invalid weights file');
                    return;
                }
                S.model = buildTFModel(json);
                if (S.model.hasGRU) {
                    S.gruState = new Float32Array(S.model.gruHidden);
                    if (S.gruTensor) { S.gruTensor.dispose(); S.gruTensor = null; }
                }
                console.log(`[AI] Weights loaded from ${file.name} (${S.model.arch}, ${S.model.nInputChannels}ch, CBAM=${S.model.hasCBAM}, SpatialAttn=${S.model.hasSpatialAttn}, GRU=${S.model.gruHidden || 0})`);
                try { GM_setValue(CFG.WEIGHTS_KEY, text); } catch(e) {}
            } catch (err) {
                console.error('[AI] Failed to load weights:', err);
            }
        };
        input.click();
    }

    async function loadCachedWeights() {
        try {
            const cached = GM_getValue(CFG.WEIGHTS_KEY, null);
            if (cached) {
                const json = JSON.parse(cached);
                S.model = buildTFModel(json);
                if (S.model.hasGRU) {
                    S.gruState = new Float32Array(S.model.gruHidden);
                    if (S.gruTensor) { S.gruTensor.dispose(); S.gruTensor = null; }
                }
                console.log(`[AI] Loaded cached weights (${S.model.version}, ${S.model.nInputChannels}ch)`);
                return true;
            }
        } catch(e) {}
        return false;
    }

    // ==================== INIT ====================
    async function init() {
        let tf = window.tf;
        if (!tf) {
            await new Promise(r => {
                const check = setInterval(() => {
                    if (window.tf) { tf = window.tf; clearInterval(check); r(); }
                }, 100);
            });
        }
        await tf.ready();
        await tf.setBackend('webgl');

        const offscreen = document.createElement('canvas');
        offscreen.width = CFG.RES;
        offscreen.height = CFG.RES;
        S.ctx = offscreen.getContext('2d', { willReadFrequently: true });
        S.ctx.imageSmoothingEnabled = false;

        S.enabled = false;

        await loadCachedWeights();

        createUI();

        pageWindow._ccAI = {
            loadWeights: function(json) {
                S.model = buildTFModel(json);
                if (S.model.hasGRU) {
                    S.gruState = new Float32Array(S.model.gruHidden);
                    if (S.gruTensor) { S.gruTensor.dispose(); S.gruTensor = null; }
                }
                console.log('[AI] Weights loaded via automation API');
            },
            enable: function() {
                S.enabled = true;
                const btn = document.querySelector('#ai-toggle');
                if (btn) { btn.textContent = 'DISABLE'; btn.style.background = '#a00'; }
            },
            disable: function() {
                S.enabled = false; release();
                const btn = document.querySelector('#ai-toggle');
                if (btn) { btn.textContent = 'ENABLE'; btn.style.background = '#06a'; }
            },
            getState: function() {
                return {
                    enabled: S.enabled, round: S.ep, steps: S.steps,
                    gsAvailable: S.gsAvailable, position: S.gsPosition,
                    angle: S.gsAngle, color: S.myColor, active: S.roundActive,
                    powerups: S.gsPowerups.length, inferMs: S.inferMs, inferAvg: S.inferAvg,
                    speedMult: S.gsSpeedMult,
                };
            },
            // Diagnostic: run inference on current game state, return logits
            testLogits: function() {
                if (!S.model || !S.gsAvailable) return { error: 'no model or game state' };
                const channels = buildChannelsFromGameState();
                const headX = S.gsPosition.x * OBS_SCALE;
                const headY = S.gsPosition.y * OBS_SCALE + OBS_OFFSET_Y;
                const heading = S.gsAngle;
                const obs = buildObservation(channels.self, channels.enemy, headX, headY, heading);
                // Count non-zero pixels per channel
                const N = CFG.RES * CFG.RES;
                const nCh = S.model.nInputChannels || 6;
                const chSums = new Array(nCh).fill(0);
                for (let i = 0; i < N; i++) {
                    for (let c = 0; c < nCh; c++) chSums[c] += obs[i * nCh + c];
                }
                // Run inference using the main infer() path (reuses same forward pass)
                const action = infer(obs, -1);
                return {
                    action: ['LEFT', 'STRAIGHT', 'RIGHT'][action] || '?',
                    chSums: chSums.map(v => Math.round(v)),
                    head: { x: headX.toFixed(1), y: headY.toFixed(1) },
                    heading: (heading * 180 / Math.PI).toFixed(0) + 'deg',
                    arch: S.model.arch,
                    nCh: nCh,
                };
            }
        };

        console.log('[AI] CurveCrash AI v6.1 initialized (IMPALA + Voronoi support)');
        console.log('[AI] Supports: NatureCNN (6ch) and IMPALA-CNN + Voronoi (7ch)');
        console.log('[AI] Click LOAD WEIGHTS to load weights JSON, then ENABLE to play');

        function loop() {
            try {
                // Always read game state so HUD shows live data even when disabled
                if (!S.gc || S.gc.width < 200) S.gc = findGameCanvas();
                readGameState();
                // Track trail positions at rAF rate (higher than decision rate)
                // so we capture position updates between decision ticks
                if (S.roundActive && S.selfTrail) {
                    updateTrailTracking();
                }
                if (S.enabled) decisionTick();
            } catch(e) { console.error('[AI] Error:', e); }
            requestAnimationFrame(loop);
        }
        requestAnimationFrame(loop);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => setTimeout(init, 1000));
    } else {
        setTimeout(init, 1000);
    }
})();
