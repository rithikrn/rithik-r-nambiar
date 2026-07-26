/*!
 * fluid-cursor.js — interactive fluid background for rithikrn.github.io
 *
 * Solves incompressible Navier–Stokes on the GPU (Stam, "Stable Fluids", 1999):
 *   splat → vorticity confinement → divergence → Jacobi pressure solve →
 *   gradient subtract → semi-Lagrangian advection of velocity + dye.
 * Pipeline follows the standard WebGL stable-fluids implementation
 * (cf. P. Dobryakov's MIT-licensed WebGL-Fluid-Simulation).
 *
 * ── INTEGRATION ───────────────────────────────────────────────────────────
 * Save as: assets/js/fluid-cursor.js
 *
 * Applied in index.html and all three projects/*.html pages:
 *   1. <canvas id="fluid" class="fluid-bg" aria-hidden="true"></canvas>
 *      replaces the old <canvas id="flowField" class="flow-field">
 *   2. the old "Ambient flow field" IIFE is deleted
 *   3. the theme toggle now calls window.refreshFluidColors()
 *   4. <script src="assets/js/fluid-cursor.js" defer></script> before </body>
 *      (../assets/js/... on the project pages)
 *   5. style.css defines .fluid-bg, .fluid-tag, --fluid-cold, --fluid-hot
 *
 * Tuning: everything worth changing is in CONFIG below. Start with
 * INTENSITY_DARK / INTENSITY_LIGHT (how visible) and CURL (how smoky).
 * ──────────────────────────────────────────────────────────────────────────
 */
(function () {
  'use strict';

  var canvas = document.getElementById('fluid');
  if (!canvas) return;

  // Respect motion preferences; skip touch devices entirely (no cursor, and
  // it costs battery on phones).
  if (matchMedia('(prefers-reduced-motion: reduce)').matches ||
      matchMedia('(pointer: coarse)').matches) {
    canvas.style.display = 'none';
    return;
  }

  var CONFIG = {
    SIM_RESOLUTION:       128,   // velocity grid (128 is plenty)
    DYE_RESOLUTION:       640,   // visible dye grid
    DENSITY_DISSIPATION:  1.7,   // higher = dye clears faster
    VELOCITY_DISSIPATION: 2.4,
    PRESSURE:             0.8,
    PRESSURE_ITERATIONS:  16,
    CURL:                 24,    // vorticity confinement — the "smoke curl"
    SPLAT_RADIUS:         0.0022,
    SPLAT_FORCE:          5200,
    DYE_AMOUNT:           0.9,
    INTENSITY_DARK:       0.55,  // global alpha of the dye, dark theme
    INTENSITY_LIGHT:      0.30,  // lighter — dark dye on pale paper reads heavy
    IDLE_SLEEP_MS:        6000   // stop the RAF loop after this much stillness
  };

  /* ── WebGL context ───────────────────────────────────────────────────── */

  var params = {
    alpha: true, depth: false, stencil: false, antialias: false,
    premultipliedAlpha: false, preserveDrawingBuffer: false
  };

  var gl = canvas.getContext('webgl2', params);
  var isWebGL2 = !!gl;
  if (!isWebGL2) {
    gl = canvas.getContext('webgl', params) ||
         canvas.getContext('experimental-webgl', params);
  }
  if (!gl) { canvas.style.display = 'none'; return; }

  var halfFloat, supportLinearFiltering;
  if (isWebGL2) {
    gl.getExtension('EXT_color_buffer_float');
    supportLinearFiltering = gl.getExtension('OES_texture_float_linear');
  } else {
    halfFloat = gl.getExtension('OES_texture_half_float');
    supportLinearFiltering = gl.getExtension('OES_texture_half_float_linear');
    if (!halfFloat) { canvas.style.display = 'none'; return; }
  }

  var halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : halfFloat.HALF_FLOAT_OES;
  gl.clearColor(0.0, 0.0, 0.0, 0.0);

  function supportRenderTextureFormat(internalFormat, format, type) {
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);
    var fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
                            gl.TEXTURE_2D, texture, 0);
    var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(texture);
    return status === gl.FRAMEBUFFER_COMPLETE;
  }

  function getSupportedFormat(internalFormat, format, type) {
    if (!supportRenderTextureFormat(internalFormat, format, type)) {
      if (!isWebGL2) return null;
      if (internalFormat === gl.R16F) return getSupportedFormat(gl.RG16F, gl.RG, type);
      if (internalFormat === gl.RG16F) return getSupportedFormat(gl.RGBA16F, gl.RGBA, type);
      return null;
    }
    return { internalFormat: internalFormat, format: format };
  }

  var formatRGBA, formatRG, formatR;
  if (isWebGL2) {
    formatRGBA = getSupportedFormat(gl.RGBA16F, gl.RGBA, halfFloatTexType);
    formatRG   = getSupportedFormat(gl.RG16F,   gl.RG,   halfFloatTexType);
    formatR    = getSupportedFormat(gl.R16F,    gl.RED,  halfFloatTexType);
  } else {
    formatRGBA = getSupportedFormat(gl.RGBA, gl.RGBA, halfFloatTexType);
    formatRG   = formatRGBA;
    formatR    = formatRGBA;
  }
  if (!formatRGBA) { canvas.style.display = 'none'; return; }

  /* ── shader plumbing ─────────────────────────────────────────────────── */

  function compileShader(type, source, keywords) {
    var src = source;
    if (keywords) {
      var prefix = '';
      for (var i = 0; i < keywords.length; i++) prefix += '#define ' + keywords[i] + '\n';
      src = prefix + source;
    }
    var shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.warn('fluid: shader compile failed —', gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  function Program(vertexShader, fragmentShader) {
    this.program = gl.createProgram();
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.bindAttribLocation(this.program, 0, 'aPosition');
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.warn('fluid: program link failed —', gl.getProgramInfoLog(this.program));
    }
    this.uniforms = {};
    var count = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
    for (var i = 0; i < count; i++) {
      var name = gl.getActiveUniform(this.program, i).name;
      this.uniforms[name] = gl.getUniformLocation(this.program, name);
    }
  }
  Program.prototype.bind = function () { gl.useProgram(this.program); };

  var baseVertexShader = compileShader(gl.VERTEX_SHADER, [
    'precision highp float;',
    'attribute vec2 aPosition;',
    'varying vec2 vUv; varying vec2 vL; varying vec2 vR; varying vec2 vT; varying vec2 vB;',
    'uniform vec2 texelSize;',
    'void main () {',
    '  vUv = aPosition * 0.5 + 0.5;',
    '  vL = vUv - vec2(texelSize.x, 0.0);',
    '  vR = vUv + vec2(texelSize.x, 0.0);',
    '  vT = vUv + vec2(0.0, texelSize.y);',
    '  vB = vUv - vec2(0.0, texelSize.y);',
    '  gl_Position = vec4(aPosition, 0.0, 1.0);',
    '}'
  ].join('\n'));

  var copyShader = compileShader(gl.FRAGMENT_SHADER, [
    'precision mediump float; precision mediump sampler2D;',
    'varying highp vec2 vUv; uniform sampler2D uTexture;',
    'void main () { gl_FragColor = texture2D(uTexture, vUv); }'
  ].join('\n'));

  var clearShader = compileShader(gl.FRAGMENT_SHADER, [
    'precision mediump float; precision mediump sampler2D;',
    'varying highp vec2 vUv; uniform sampler2D uTexture; uniform float value;',
    'void main () { gl_FragColor = value * texture2D(uTexture, vUv); }'
  ].join('\n'));

  var splatShader = compileShader(gl.FRAGMENT_SHADER, [
    'precision highp float; precision highp sampler2D;',
    'varying vec2 vUv; uniform sampler2D uTarget; uniform float aspectRatio;',
    'uniform vec3 color; uniform vec2 point; uniform float radius;',
    'void main () {',
    '  vec2 p = vUv - point.xy; p.x *= aspectRatio;',
    '  vec3 splat = exp(-dot(p, p) / radius) * color;',
    '  vec3 base = texture2D(uTarget, vUv).xyz;',
    '  gl_FragColor = vec4(base + splat, 1.0);',
    '}'
  ].join('\n'));

  var advectionShader = compileShader(gl.FRAGMENT_SHADER, [
    'precision highp float; precision highp sampler2D;',
    'varying vec2 vUv;',
    'uniform sampler2D uVelocity; uniform sampler2D uSource;',
    'uniform vec2 texelSize; uniform vec2 dyeTexelSize;',
    'uniform float dt; uniform float dissipation;',
    'vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {',
    '  vec2 st = uv / tsize - 0.5;',
    '  vec2 iuv = floor(st); vec2 fuv = fract(st);',
    '  vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);',
    '  vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);',
    '  vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);',
    '  vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);',
    '  return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);',
    '}',
    'void main () {',
    '#ifdef MANUAL_FILTERING',
    '  vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;',
    '  vec4 result = bilerp(uSource, coord, dyeTexelSize);',
    '#else',
    '  vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;',
    '  vec4 result = texture2D(uSource, coord);',
    '#endif',
    '  float decay = 1.0 + dissipation * dt;',
    '  gl_FragColor = result / decay;',
    '}'
  ].join('\n'), supportLinearFiltering ? null : ['MANUAL_FILTERING']);

  var divergenceShader = compileShader(gl.FRAGMENT_SHADER, [
    'precision mediump float; precision mediump sampler2D;',
    'varying highp vec2 vUv; varying highp vec2 vL; varying highp vec2 vR;',
    'varying highp vec2 vT; varying highp vec2 vB;',
    'uniform sampler2D uVelocity;',
    'void main () {',
    '  float L = texture2D(uVelocity, vL).x;',
    '  float R = texture2D(uVelocity, vR).x;',
    '  float T = texture2D(uVelocity, vT).y;',
    '  float B = texture2D(uVelocity, vB).y;',
    '  vec2 C = texture2D(uVelocity, vUv).xy;',
    '  if (vL.x < 0.0) { L = -C.x; }',
    '  if (vR.x > 1.0) { R = -C.x; }',
    '  if (vT.y > 1.0) { T = -C.y; }',
    '  if (vB.y < 0.0) { B = -C.y; }',
    '  float div = 0.5 * (R - L + T - B);',
    '  gl_FragColor = vec4(div, 0.0, 0.0, 1.0);',
    '}'
  ].join('\n'));

  var curlShader = compileShader(gl.FRAGMENT_SHADER, [
    'precision mediump float; precision mediump sampler2D;',
    'varying highp vec2 vUv; varying highp vec2 vL; varying highp vec2 vR;',
    'varying highp vec2 vT; varying highp vec2 vB;',
    'uniform sampler2D uVelocity;',
    'void main () {',
    '  float L = texture2D(uVelocity, vL).y;',
    '  float R = texture2D(uVelocity, vR).y;',
    '  float T = texture2D(uVelocity, vT).x;',
    '  float B = texture2D(uVelocity, vB).x;',
    '  float vorticity = R - L - T + B;',
    '  gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);',
    '}'
  ].join('\n'));

  var vorticityShader = compileShader(gl.FRAGMENT_SHADER, [
    'precision highp float; precision highp sampler2D;',
    'varying vec2 vUv; varying vec2 vL; varying vec2 vR; varying vec2 vT; varying vec2 vB;',
    'uniform sampler2D uVelocity; uniform sampler2D uCurl;',
    'uniform float curl; uniform float dt;',
    'void main () {',
    '  float L = texture2D(uCurl, vL).x;',
    '  float R = texture2D(uCurl, vR).x;',
    '  float T = texture2D(uCurl, vT).x;',
    '  float B = texture2D(uCurl, vB).x;',
    '  float C = texture2D(uCurl, vUv).x;',
    '  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));',
    '  force /= length(force) + 0.0001;',
    '  force *= curl * C;',
    '  force.y *= -1.0;',
    '  vec2 velocity = texture2D(uVelocity, vUv).xy;',
    '  velocity += force * dt;',
    '  velocity = min(max(velocity, -1000.0), 1000.0);',
    '  gl_FragColor = vec4(velocity, 0.0, 1.0);',
    '}'
  ].join('\n'));

  var pressureShader = compileShader(gl.FRAGMENT_SHADER, [
    'precision mediump float; precision mediump sampler2D;',
    'varying highp vec2 vUv; varying highp vec2 vL; varying highp vec2 vR;',
    'varying highp vec2 vT; varying highp vec2 vB;',
    'uniform sampler2D uPressure; uniform sampler2D uDivergence;',
    'void main () {',
    '  float L = texture2D(uPressure, vL).x;',
    '  float R = texture2D(uPressure, vR).x;',
    '  float T = texture2D(uPressure, vT).x;',
    '  float B = texture2D(uPressure, vB).x;',
    '  float divergence = texture2D(uDivergence, vUv).x;',
    '  float pressure = (L + R + B + T - divergence) * 0.25;',
    '  gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);',
    '}'
  ].join('\n'));

  var gradientSubtractShader = compileShader(gl.FRAGMENT_SHADER, [
    'precision mediump float; precision mediump sampler2D;',
    'varying highp vec2 vUv; varying highp vec2 vL; varying highp vec2 vR;',
    'varying highp vec2 vT; varying highp vec2 vB;',
    'uniform sampler2D uPressure; uniform sampler2D uVelocity;',
    'void main () {',
    '  float L = texture2D(uPressure, vL).x;',
    '  float R = texture2D(uPressure, vR).x;',
    '  float T = texture2D(uPressure, vT).x;',
    '  float B = texture2D(uPressure, vB).x;',
    '  vec2 velocity = texture2D(uVelocity, vUv).xy;',
    '  velocity.xy -= vec2(R - L, T - B);',
    '  gl_FragColor = vec4(velocity, 0.0, 1.0);',
    '}'
  ].join('\n'));

  // Display: dye magnitude → a cool→warm ramp, i.e. the same diverging
  // colormap convention every CFD contour plot uses. Alpha keeps the page
  // background showing through so text contrast never drops.
  var displayShader = compileShader(gl.FRAGMENT_SHADER, [
    'precision highp float; precision highp sampler2D;',
    'varying vec2 vUv; uniform sampler2D uTexture;',
    'uniform vec3 uCold; uniform vec3 uHot; uniform float uIntensity;',
    'void main () {',
    '  float d = clamp(texture2D(uTexture, vUv).r, 0.0, 1.0);',
    '  float a = smoothstep(0.02, 0.85, d);',
    '  vec3 col = mix(uCold, uHot, smoothstep(0.25, 0.95, d));',
    '  gl_FragColor = vec4(col, a * uIntensity);',
    '}'
  ].join('\n'));

  var copyProgram      = new Program(baseVertexShader, copyShader);
  var clearProgram     = new Program(baseVertexShader, clearShader);
  var splatProgram     = new Program(baseVertexShader, splatShader);
  var advectionProgram = new Program(baseVertexShader, advectionShader);
  var divergenceProgram= new Program(baseVertexShader, divergenceShader);
  var curlProgram      = new Program(baseVertexShader, curlShader);
  var vorticityProgram = new Program(baseVertexShader, vorticityShader);
  var pressureProgram  = new Program(baseVertexShader, pressureShader);
  var gradienSubtractProgram = new Program(baseVertexShader, gradientSubtractShader);
  var displayProgram   = new Program(baseVertexShader, displayShader);

  /* ── framebuffers ────────────────────────────────────────────────────── */

  var blit = (function () {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);
    return function (target) {
      if (target == null) {
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      } else {
        gl.viewport(0, 0, target.width, target.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      }
      gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    };
  })();

  function createFBO(w, h, internalFormat, format, type, param) {
    gl.activeTexture(gl.TEXTURE0);
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    var fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    return {
      texture: texture, fbo: fbo, width: w, height: h,
      texelSizeX: 1.0 / w, texelSizeY: 1.0 / h,
      attach: function (id) {
        gl.activeTexture(gl.TEXTURE0 + id);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        return id;
      }
    };
  }

  function createDoubleFBO(w, h, internalFormat, format, type, param) {
    var fbo1 = createFBO(w, h, internalFormat, format, type, param);
    var fbo2 = createFBO(w, h, internalFormat, format, type, param);
    return {
      width: w, height: h, texelSizeX: fbo1.texelSizeX, texelSizeY: fbo1.texelSizeY,
      get read() { return fbo1; },
      set read(v) { fbo1 = v; },
      get write() { return fbo2; },
      set write(v) { fbo2 = v; },
      swap: function () { var t = fbo1; fbo1 = fbo2; fbo2 = t; }
    };
  }

  function resizeFBO(target, w, h, internalFormat, format, type, param) {
    var newFBO = createFBO(w, h, internalFormat, format, type, param);
    copyProgram.bind();
    gl.uniform1i(copyProgram.uniforms.uTexture, target.attach(0));
    blit(newFBO);
    return newFBO;
  }

  function resizeDoubleFBO(target, w, h, internalFormat, format, type, param) {
    if (target.width === w && target.height === h) return target;
    target.read = resizeFBO(target.read, w, h, internalFormat, format, type, param);
    target.write = createFBO(w, h, internalFormat, format, type, param);
    target.width = w; target.height = h;
    target.texelSizeX = 1.0 / w; target.texelSizeY = 1.0 / h;
    return target;
  }

  function getResolution(resolution) {
    var aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspectRatio < 1) aspectRatio = 1.0 / aspectRatio;
    var min = Math.round(resolution);
    var max = Math.round(resolution * aspectRatio);
    if (gl.drawingBufferWidth > gl.drawingBufferHeight) return { width: max, height: min };
    return { width: min, height: max };
  }

  var dye, velocity, divergence, curl, pressure;
  var filtering = supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

  function initFramebuffers() {
    var simRes = getResolution(CONFIG.SIM_RESOLUTION);
    var dyeRes = getResolution(CONFIG.DYE_RESOLUTION);
    var texType = halfFloatTexType;

    gl.disable(gl.BLEND);

    if (!dye) dye = createDoubleFBO(dyeRes.width, dyeRes.height, formatRGBA.internalFormat, formatRGBA.format, texType, filtering);
    else dye = resizeDoubleFBO(dye, dyeRes.width, dyeRes.height, formatRGBA.internalFormat, formatRGBA.format, texType, filtering);

    if (!velocity) velocity = createDoubleFBO(simRes.width, simRes.height, formatRG.internalFormat, formatRG.format, texType, filtering);
    else velocity = resizeDoubleFBO(velocity, simRes.width, simRes.height, formatRG.internalFormat, formatRG.format, texType, filtering);

    divergence = createFBO(simRes.width, simRes.height, formatR.internalFormat, formatR.format, texType, gl.NEAREST);
    curl       = createFBO(simRes.width, simRes.height, formatR.internalFormat, formatR.format, texType, gl.NEAREST);
    pressure   = createDoubleFBO(simRes.width, simRes.height, formatR.internalFormat, formatR.format, texType, gl.NEAREST);
  }

  function resizeCanvas() {
    var dpr = Math.min(window.devicePixelRatio || 1, 1.5); // 1.5 is plenty; 2 doubles cost
    var w = Math.floor(canvas.clientWidth * dpr);
    var h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      return true;
    }
    return false;
  }

  /* ── colours, driven by the site's CSS variables ─────────────────────── */

  var cold = [0.31, 0.42, 0.53], hot = [0.85, 0.44, 0.24], intensity = CONFIG.INTENSITY_DARK;

  function hexToRgb(hex, fallback) {
    var m = String(hex || '').trim().replace('#', '');
    if (m.length === 3) m = m[0] + m[0] + m[1] + m[1] + m[2] + m[2];
    if (!/^[0-9a-f]{6}$/i.test(m)) return fallback;
    var n = parseInt(m, 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }

  function refreshColors() {
    var cs = getComputedStyle(document.documentElement);
    var isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    cold = hexToRgb(cs.getPropertyValue('--fluid-cold'), isDark ? [0.31, 0.66, 0.78] : [0.12, 0.42, 0.53]);
    hot  = hexToRgb(cs.getPropertyValue('--fluid-hot'),  isDark ? [0.85, 0.44, 0.24] : [0.76, 0.35, 0.16]);
    intensity = isDark ? CONFIG.INTENSITY_DARK : CONFIG.INTENSITY_LIGHT;
  }
  window.refreshFluidColors = refreshColors;

  /* ── simulation ──────────────────────────────────────────────────────── */

  var lastTime = Date.now();
  var lastInputTime = 0;
  var running = false;
  var rafId = null;

  function correctRadius(radius) {
    var aspectRatio = canvas.width / canvas.height;
    return aspectRatio > 1 ? radius * aspectRatio : radius;
  }

  function splat(x, y, dx, dy, amount) {
    splatProgram.bind();
    gl.uniform1i(splatProgram.uniforms.uTarget, velocity.read.attach(0));
    gl.uniform1f(splatProgram.uniforms.aspectRatio, canvas.width / canvas.height);
    gl.uniform2f(splatProgram.uniforms.point, x, y);
    gl.uniform3f(splatProgram.uniforms.color, dx, dy, 0.0);
    gl.uniform1f(splatProgram.uniforms.radius, correctRadius(CONFIG.SPLAT_RADIUS));
    blit(velocity.write);
    velocity.swap();

    gl.uniform1i(splatProgram.uniforms.uTarget, dye.read.attach(0));
    gl.uniform3f(splatProgram.uniforms.color, amount, 0.0, 0.0);
    blit(dye.write);
    dye.swap();
  }

  function step(dt) {
    gl.disable(gl.BLEND);

    curlProgram.bind();
    gl.uniform2f(curlProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(curlProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(curl);

    vorticityProgram.bind();
    gl.uniform2f(vorticityProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(vorticityProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(vorticityProgram.uniforms.uCurl, curl.attach(1));
    gl.uniform1f(vorticityProgram.uniforms.curl, CONFIG.CURL);
    gl.uniform1f(vorticityProgram.uniforms.dt, dt);
    blit(velocity.write);
    velocity.swap();

    divergenceProgram.bind();
    gl.uniform2f(divergenceProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(divergenceProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(divergence);

    clearProgram.bind();
    gl.uniform1i(clearProgram.uniforms.uTexture, pressure.read.attach(0));
    gl.uniform1f(clearProgram.uniforms.value, CONFIG.PRESSURE);
    blit(pressure.write);
    pressure.swap();

    pressureProgram.bind();
    gl.uniform2f(pressureProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(pressureProgram.uniforms.uDivergence, divergence.attach(0));
    for (var i = 0; i < CONFIG.PRESSURE_ITERATIONS; i++) {
      gl.uniform1i(pressureProgram.uniforms.uPressure, pressure.read.attach(1));
      blit(pressure.write);
      pressure.swap();
    }

    gradienSubtractProgram.bind();
    gl.uniform2f(gradienSubtractProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(gradienSubtractProgram.uniforms.uPressure, pressure.read.attach(0));
    gl.uniform1i(gradienSubtractProgram.uniforms.uVelocity, velocity.read.attach(1));
    blit(velocity.write);
    velocity.swap();

    advectionProgram.bind();
    gl.uniform2f(advectionProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    if (!supportLinearFiltering) {
      gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, velocity.texelSizeX, velocity.texelSizeY);
    }
    var velocityId = velocity.read.attach(0);
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocityId);
    gl.uniform1i(advectionProgram.uniforms.uSource, velocityId);
    gl.uniform1f(advectionProgram.uniforms.dt, dt);
    gl.uniform1f(advectionProgram.uniforms.dissipation, CONFIG.VELOCITY_DISSIPATION);
    blit(velocity.write);
    velocity.swap();

    if (!supportLinearFiltering) {
      gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, dye.texelSizeX, dye.texelSizeY);
    }
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(advectionProgram.uniforms.uSource, dye.read.attach(1));
    gl.uniform1f(advectionProgram.uniforms.dissipation, CONFIG.DENSITY_DISSIPATION);
    blit(dye.write);
    dye.swap();
  }

  function render() {
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    displayProgram.bind();
    gl.uniform1i(displayProgram.uniforms.uTexture, dye.read.attach(0));
    gl.uniform3f(displayProgram.uniforms.uCold, cold[0], cold[1], cold[2]);
    gl.uniform3f(displayProgram.uniforms.uHot, hot[0], hot[1], hot[2]);
    gl.uniform1f(displayProgram.uniforms.uIntensity, intensity);
    blit(null);
    gl.disable(gl.BLEND);
  }

  function frame() {
    var now = Date.now();
    var dt = Math.min((now - lastTime) / 1000, 0.0166);
    lastTime = now;

    if (resizeCanvas()) initFramebuffers();
    step(dt);
    render();

    // Idle: once the dye has dissipated, stop burning frames.
    if (now - lastInputTime > CONFIG.IDLE_SLEEP_MS) {
      running = false;
      rafId = null;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }
    rafId = requestAnimationFrame(frame);
  }

  function wake() {
    lastInputTime = Date.now();
    if (!running && !document.hidden) {
      running = true;
      lastTime = Date.now();
      rafId = requestAnimationFrame(frame);
    }
  }

  /* ── pointer input ───────────────────────────────────────────────────── */

  var prevX = null, prevY = null;

  window.addEventListener('pointermove', function (e) {
    if (e.pointerType === 'touch') return;
    var x = e.clientX / window.innerWidth;
    var y = 1.0 - e.clientY / window.innerHeight;

    if (prevX === null) { prevX = x; prevY = y; wake(); return; }

    var dx = (x - prevX) * CONFIG.SPLAT_FORCE;
    var dy = (y - prevY) * CONFIG.SPLAT_FORCE;
    prevX = x; prevY = y;

    var speed = Math.min(Math.sqrt(dx * dx + dy * dy) / CONFIG.SPLAT_FORCE, 0.08) / 0.08;
    if (speed < 0.01) return;

    wake();
    splat(x, y, dx, dy, CONFIG.DYE_AMOUNT * (0.35 + 0.65 * speed));
  }, { passive: true });

  // A brief pulse on click gives the interaction a payoff.
  window.addEventListener('pointerdown', function (e) {
    if (e.pointerType === 'touch') return;
    var x = e.clientX / window.innerWidth;
    var y = 1.0 - e.clientY / window.innerHeight;
    wake();
    for (var i = 0; i < 4; i++) {
      var a = (i / 4) * Math.PI * 2;
      splat(x, y, Math.cos(a) * 900, Math.sin(a) * 900, CONFIG.DYE_AMOUNT * 0.8);
    }
  }, { passive: true });

  document.addEventListener('visibilitychange', function () {
    if (document.hidden && rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
      running = false;
    } else if (!document.hidden) {
      prevX = prevY = null;
    }
  });

  var resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      if (resizeCanvas()) initFramebuffers();
    }, 200);
  }, { passive: true });

  /* ── boot ────────────────────────────────────────────────────────────── */

  refreshColors();
  resizeCanvas();
  initFramebuffers();
  render();
})();
