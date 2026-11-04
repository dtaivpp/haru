// Haru face: renders the Live2D avatar and does whatever the backend says.
//
// URL parameters (the Pi's kiosk launcher fills these in):
//   device       terminal device_id whose avatar this is (required)
//   token        terminal token, for the gateway's /display endpoint (required)
//   backend      host:port of the gateway (default: this page's host, port 8765)
//   model        which avatar: "haru" (the default and only one)
//   zoom         1 = whole body; higher frames the upper body (default: per model)
//   mouthDelayMs how long after "speaking" the mouth starts, to match the audio
//                still sitting in the Pi's playback buffer (default 140)
//   renderScale  fraction of screen pixels the avatar is drawn at, then stretched
//                (default: 0.5 on screens wider than 2000 px, i.e. 1080p work on 4K)
//
// Everything here is presentation. No decisions are made on this page.

(() => {
  const params = new URLSearchParams(location.search)
  const device = params.get('device') || ''
  const token = params.get('token') || ''
  const backend = params.get('backend') || `${location.hostname}:8765`
  const mouthDelayMs = Number.parseFloat(params.get('mouthDelayMs') || '140')
  const renderScale = Number.parseFloat(params.get('renderScale') || (window.innerWidth > 2000 ? '0.5' : '1'))
  // Live2D Inc. sample models (Free Material License). Each maps our emotion
  // vocabulary either onto its own authored expressions (nicer, where a model
  // has them) or onto raw parameter targets (EMOTIONS below).
  const MODELS = {
    haru: {
      url: 'models/haru/Haru.model3.json',
      zoom: 1.9,
      top: 0.02,
      expressions: {
        neutral: null,
        happy: 'F02',
        sad: 'F04',
        angry: 'F03',
        surprised: 'F06',
        awkward: 'F07',
        think: 'F08',
        question: 'F01',
        curious: 'F01',
      },
      // Hand-built: Haru has no gesture motions. The A arm parts hang at her sides (ArmRA
      // her right, ArmLA her left). The B parts are posable arms, posed by ParamArm?B (tuned
      // on the Pi screen), but drawn on the *opposite* side to their names: ArmRB is her
      // left arm (~5 a raised open palm, ~2.5 hand-to-chin) and ArmLB her right (~5 a raised
      // fist). So raising RB must hide LA, not RA, or one side has two arms and the other
      // none. A gesture swaps arm parts at mid-fade, swings or holds params, adds offsets,
      // and can override the face.
      // A swing marked `swap` poses a swapped-in drawing, so it only runs while that's shown.
      gestures: {
        // Raised palm on her left, her right arm hanging at her side.
        wave: {
          durationMs: 2600, fadeMs: 300,
          show: ['Part01ArmRB001', 'Part01ArmRA001'], hide: ['Part01ArmLA001', 'Part01ArmLB001'],
          // from 3.6: blending up from lower values passed through her hand-to-chin pose.
          swing: [{ param: 'ParamArmRB', swap: true, from: 3.6, center: 4.5, amp: 0.5, hz: 2.2 }],
          add: { ParamAngleZ: -6, ParamBodyAngleZ: -2 },
        },
        // Both hands up ("yay!"): palm held high, fist pumping, a little bounce.
        celebrate: {
          durationMs: 2400, fadeMs: 300,
          show: ['Part01ArmRB001', 'Part01ArmLB001'], hide: ['Part01ArmRA001', 'Part01ArmLA001'],
          swing: [
            { param: 'ParamArmRB', swap: true, from: 3.6, center: 5, amp: 0 },
            { param: 'ParamArmLB', swap: true, from: 3.6, center: 4.6, amp: 0.4, hz: 2.5 },
            { param: 'ParamBodyAngleY', from: 0, center: 0, amp: 3, hz: 2.5 },
          ],
          add: { ParamAngleY: 4 },
          expression: 'F02',
        },
        // One palm up, head tilted, eyes off to the side, lopsided brows, unsure mouth.
        shrug: {
          durationMs: 2300, fadeMs: 350,
          show: ['Part01ArmRB001', 'Part01ArmRA001'], hide: ['Part01ArmLA001', 'Part01ArmLB001'],
          swing: [{ param: 'ParamArmRB', swap: true, from: 3.6, center: 4.9, amp: 0 }],
          add: { ParamAngleZ: 16, ParamAngleX: -8, ParamBodyAngleZ: 4, ParamEyeBallX: 0.7, ParamEyeBallY: 0.4 },
          set: { ParamBrowLY: 0.9, ParamBrowRY: -0.4, ParamBrowLAngle: 0.6, ParamBrowRAngle: -0.6, ParamMouthForm: -0.8 },
          expression: 'F07',
        },
      },
    },
  }
  const MODEL = MODELS[params.get('model') || 'haru'] || MODELS.haru
  const zoom = Number.parseFloat(params.get('zoom') || String(MODEL.zoom))

  // Parameter targets per emotion, for models without authored expressions.
  // "set" params replace the motion's value; "add" params are offsets on top of
  // idle motion (so she keeps breathing and swaying). Models with expressions
  // still get the "add" head/eye cues. Eye openness is left to auto-blink.
  const EMOTIONS = {
    neutral: {},
    happy: { set: { ParamEyeLSmile: 1, ParamEyeRSmile: 1, ParamMouthForm: 1, ParamCheek: 0.6 } },
    sad: { set: { ParamMouthForm: -1, ParamBrowLForm: -0.8, ParamBrowRForm: -0.8 }, add: { ParamAngleY: -8 } },
    angry: { set: { ParamMouthForm: -0.7, ParamBrowLForm: -1, ParamBrowRForm: -1 }, add: { ParamAngleY: -4 } },
    surprised: { set: { ParamBrowLForm: 1, ParamBrowRForm: 1, ParamMouthForm: 0 }, add: { ParamAngleY: 6 } },
    think: { set: { ParamMouthForm: -0.2 }, add: { ParamEyeBallX: 0.6, ParamEyeBallY: 0.6, ParamAngleZ: 8 } },
    awkward: { set: { ParamMouthForm: -0.3, ParamCheek: 0.8 }, add: { ParamEyeBallX: -0.5, ParamAngleX: -10 } },
    question: { set: { ParamBrowLForm: 0.6 }, add: { ParamAngleZ: -12 } },
    curious: { set: { ParamBrowRForm: 0.5, ParamMouthForm: 0.3 }, add: { ParamAngleZ: 10, ParamEyeBallY: 0.2 } },
  }
  // Small cues for pipeline state, layered under the emotion.
  const STATES = {
    idle: {},
    listening: { add: { ParamAngleZ: 5, ParamAngleX: 4 } },
    thinking: { add: { ParamEyeBallX: 0.5, ParamEyeBallY: 0.7, ParamAngleZ: -4 } },
    speaking: {},
  }

  const face = {
    state: 'idle',
    emotion: 'neutral',
    speakingStream: null,
    mouthPos: 0,            // playhead into the speaking stream's mouth track, in ms
    mouthClock: 0,
    mouthTracks: new Map(), // stream_id -> { fps, values: [] }
    mouth: 0,
    current: {}, // parameter id -> eased value
    model: null,
    shownExpression: undefined,
    pendingGesture: null,   // { name, at }: waits for speech to start
    gesture: null,          // { spec, start }: playing
  }

  // For models with authored expressions: switch when the emotion changes
  // (the expression manager fades between them).
  function showEmotion() {
    if (!MODEL.expressions || !face.model || face.gesture?.spec.expression) return
    const exp = MODEL.expressions[face.emotion] ?? null
    if (exp === face.shownExpression) return
    face.shownExpression = exp
    if (exp) face.model.expression(exp)
    else face.model.internalModel.motionManager.expressionManager?.resetExpression()
  }

  // --- notifications ---------------------------------------------------------------
  // A bell with the unread count pops in when something arrives. Nothing is spoken:
  // the list only opens when the user asks Haru for it (notify.panel).

  const PANEL_OPEN_MS = 45000
  let unread = 0
  let panelTimer = null

  function showBadge(n) {
    const bell = document.getElementById('bell')
    bell.querySelector('.count').textContent = n > 99 ? '99+' : String(n)
    bell.classList.toggle('show', n > 0)
    if (n > unread) {          // something new: a little wiggle
      bell.classList.remove('pop')
      void bell.offsetWidth    // restart the animation
      bell.classList.add('pop')
    }
    unread = n
  }

  function hidePanel() {
    clearTimeout(panelTimer)
    document.getElementById('panel').classList.remove('open')
  }

  function showPanel(items) {
    const panel = document.getElementById('panel')
    const list = panel.querySelector('ol')
    list.replaceChildren()
    if (!items.length) {
      const li = document.createElement('li')
      li.className = 'empty'
      li.textContent = 'Nothing new'
      list.append(li)
    }
    for (const it of items) {
      const li = document.createElement('li')
      if (!it.read) li.className = 'new'
      const title = document.createElement('div')
      title.className = 'title'
      title.textContent = it.title || ''
      li.append(title)
      if (it.body) {
        const body = document.createElement('div')
        body.className = 'body'
        body.textContent = it.body
        li.append(body)
      }
      const meta = document.createElement('div')
      meta.className = 'meta'
      meta.textContent = [it.source, it.when].filter(Boolean).join(' · ')
      li.append(meta)
      list.append(li)
    }
    panel.classList.add('open')
    clearTimeout(panelTimer)
    panelTimer = setTimeout(() => panel.classList.remove('open'), PANEL_OPEN_MS)
  }

  // --- backend link -------------------------------------------------------------

  function connect(attempt = 0) {
    if (!device || !token) {
      document.getElementById('status').textContent = 'missing ?device= / ?token= in the URL'
      document.body.classList.add('offline')
      return
    }
    const ws = new WebSocket(`ws://${backend}/display?device=${encodeURIComponent(device)}&token=${encodeURIComponent(token)}`)
    ws.onopen = () => {
      attempt = 0
      document.body.classList.remove('offline')
    }
    ws.onmessage = (ev) => {
      let msg
      try { msg = JSON.parse(ev.data) }
      catch { return }
      const d = msg.data || {}
      if (msg.type === 'avatar.state') {
        if (d.emotion) {
          face.emotion = d.emotion in EMOTIONS ? d.emotion : 'neutral'
          showEmotion()
        }
        if (d.state) {
          if (d.state === 'speaking' && d.stream_id !== undefined && d.stream_id !== face.speakingStream) {
            face.speakingStream = d.stream_id
            face.mouthPos = -mouthDelayMs
            face.mouthClock = performance.now()
          }
          if (d.state !== 'speaking') {
            if (face.speakingStream !== null) face.mouthTracks.delete(face.speakingStream)
            face.speakingStream = null
          }
          face.state = d.state
        }
      }
      else if (msg.type === 'avatar.gesture') {
        if ((MODEL.gestures || {})[d.name]) face.pendingGesture = { name: d.name, at: performance.now() }
      }
      else if (msg.type === 'notify.badge') showBadge(d.unread || 0)
      else if (msg.type === 'notify.panel') {
        if (d.open === false) hidePanel()
        else showPanel(d.items || [])
      }
      else if (msg.type === 'avatar.mouth') {
        const track = face.mouthTracks.get(d.stream_id) || { fps: d.fps || 50, values: [] }
        track.values.push(...(d.values || []))
        face.mouthTracks.set(d.stream_id, track)
        // Keep only the few most recent streams.
        while (face.mouthTracks.size > 4) face.mouthTracks.delete(face.mouthTracks.keys().next().value)
      }
    }
    ws.onclose = () => {
      document.body.classList.add('offline')
      face.state = 'idle'
      face.speakingStream = null
      const delay = Math.min(10000, 1000 * 2 ** attempt) * (0.5 + Math.random() / 2)
      setTimeout(() => connect(attempt + 1), delay)
    }
  }

  // --- rendering ---------------------------------------------------------------------

  // The playhead advances with time but waits at the end of the data received so far.
  // Speech arrives sentence by sentence; when the LLM is slower than playback (or a tool
  // call runs mid-reply, after a filler line) the speaker goes quiet until more audio
  // comes, and its mouth data arrives with it. A plain clock ran on through those gaps and
  // finished the mouth early, while she was still talking.
  function mouthTarget(now) {
    const dt = now - face.mouthClock
    face.mouthClock = now
    if (face.state !== 'speaking' || face.speakingStream === null) return 0
    const track = face.mouthTracks.get(face.speakingStream)
    if (!track) return 0
    const frameMs = 1000 / track.fps
    face.mouthPos = Math.min(face.mouthPos + dt, track.values.length * frameMs)
    const idx = Math.floor(face.mouthPos / frameMs)
    return idx >= 0 && idx < track.values.length ? track.values[idx] : 0
  }

  function ease(id, target, rate) {
    const cur = face.current[id] ?? 0
    const next = cur + (target - cur) * rate
    face.current[id] = next
    return next
  }

  function applyParameters(coreModel, dtMs) {
    const now = performance.now()
    // Lips move fast (but not jittery); expressions drift slowly.
    const lipRate = Math.min(1, dtMs / 45)
    const exprRate = Math.min(1, dtMs / 350)

    const emo = EMOTIONS[face.emotion] || {}
    const st = STATES[face.state] || {}
    const set = { ...(emo.set || {}) }
    const add = { ...(st.add || {}) }
    for (const [k, v] of Object.entries(emo.add || {})) add[k] = (add[k] || 0) + v

    // Models with expressions own their face; overwriting these would erase them.
    if (!MODEL.expressions) {
      const setIds = new Set([...Object.keys(set), 'ParamEyeLSmile', 'ParamEyeRSmile', 'ParamMouthForm',
        'ParamCheek', 'ParamBrowLForm', 'ParamBrowRForm'])
      for (const id of setIds) coreModel.setParameterValueById(id, ease(`set:${id}`, set[id] ?? 0, exprRate))
    }

    const addIds = new Set([...Object.keys(add), 'ParamAngleX', 'ParamAngleY', 'ParamAngleZ',
      'ParamEyeBallX', 'ParamEyeBallY'])
    for (const id of addIds) coreModel.addParameterValueById(id, ease(`add:${id}`, add[id] ?? 0, exprRate))

    applyGesture(coreModel, now)

    const surprisedOpen = face.emotion === 'surprised' && face.state !== 'speaking' ? 0.3 : 0
    face.mouth = ease('mouth', Math.max(mouthTarget(now), surprisedOpen), lipRate)
    coreModel.setParameterValueById('ParamMouthOpenY', face.mouth)
  }

  // One-off body motions (e.g. a wave on hello/goodbye). Held until she starts
  // speaking, so the wave lands with "Hi!" rather than a second early.
  function applyGesture(coreModel, now) {
    const p = face.pendingGesture
    if (p && !face.gesture && (face.state === 'speaking' || now - p.at > 1500)) {
      startGesture(MODEL.gestures[p.name], now)
      face.pendingGesture = null
    }
    const g = face.gesture
    if (!g) return
    const t = now - g.start
    const s = g.spec
    if (t >= s.durationMs) {
      endGesture()
      return
    }
    const smooth = x => x * x * (3 - 2 * x)
    const env = smooth(Math.min(1, t / s.fadeMs, (s.durationMs - t) / s.fadeMs))
    // Swap arm drawings at the midpoint of the fade, both in and out; a crossfade shows
    // both arms (ghosting). Outside that window the arms are the idle motion's: show the
    // drawing its pose parameter asks for, exactly. Leaving the swapped parts to the pose
    // system made it crossfade them while the arm param snapped back, so the raised arm
    // vanished and reappeared hanging down.
    const swapped = env > 0.5
    const motionWants = id => (coreModel.getParameterValueById(id) > 0.001 ? 1 : 0)
    for (const id of s.show || []) coreModel.setPartOpacityById(id, swapped ? 1 : motionWants(id))
    for (const id of s.hide || []) coreModel.setPartOpacityById(id, swapped ? 0 : motionWants(id))
    for (const w of s.swing || []) {
      // A swapped drawing's pose param is only ours while that drawing is.
      if (w.swap && !swapped) continue
      const v = w.center + w.amp * Math.sin(2 * Math.PI * (w.hz || 0) * (t / 1000))
      coreModel.setParameterValueById(w.param, w.from + (v - w.from) * env)
    }
    for (const [id, v] of Object.entries(s.set || {})) {
      const cur = coreModel.getParameterValueById(id)
      coreModel.setParameterValueById(id, cur + (v - cur) * env)
    }
    for (const [id, v] of Object.entries(s.add || {})) coreModel.addParameterValueById(id, v * env)
  }

  // A gesture's own expression replaces the emotion's for its duration.
  function startGesture(spec, now) {
    face.gesture = { spec, start: now }
    if (spec.expression && face.model) {
      face.model.expression(spec.expression)
      face.shownExpression = spec.expression
    }
  }

  function endGesture() {
    const spec = face.gesture.spec
    face.gesture = null
    if (spec.expression) {
      face.shownExpression = undefined
      showEmotion()
    }
  }

  async function main() {
    const canvas = document.getElementById('stage')
    // Draw at renderScale x the screen's pixels and let CSS stretch the canvas.
    // On the Pi's 4K panel that's a quarter of the fill work. (Don't use
    // Chromium's --force-device-scale-factor for this: on Wayland/cage it
    // inflated the viewport and only the top-left quarter showed.)
    const app = new PIXI.Application({ view: canvas, resizeTo: window, backgroundAlpha: 0, antialias: true,
      resolution: renderScale * (window.devicePixelRatio || 1), autoDensity: true })
    const model = await PIXI.live2d.Live2DModel.from(MODEL.url, { autoInteract: false })
    app.stage.addChild(model)
    face.model = model
    // Handle for debugging and tuning over DevTools (127.0.0.1:9222 on the Pi).
    window.__haru = { model, face, PIXI }
    showEmotion()

    const natural = { w: model.width, h: model.height }
    function fit() {
      const scale = (window.innerHeight / natural.h) * zoom
      model.scale.set(scale)
      model.anchor.set(0.5, 0)
      model.x = window.innerWidth / 2
      model.y = window.innerHeight * MODEL.top
    }
    fit()
    window.addEventListener('resize', fit)

    let last = performance.now()
    model.internalModel.on('beforeModelUpdate', () => {
      const now = performance.now()
      applyParameters(model.internalModel.coreModel, now - last)
      last = now
    })
    connect()
  }

  main().catch((err) => {
    document.getElementById('status').textContent = `face failed to load: ${err}`
    document.body.classList.add('offline')
    console.error(err)
  })
})()
