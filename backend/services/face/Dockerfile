# syntax=docker/dockerfile:1.7
# Haru's face page (Stage 7): a static Live2D display that a terminal's screen
# loads in kiosk mode.
FROM nginx:stable-alpine
# Haru (receptionist/greeter), Live2D Inc. sample model, Free Material License,
# from Live2D's own Cubism Web SDK samples at a pinned commit (the raw URL is
# immutable). The tap-reaction voice clips (sounds/) are left out; the kiosk never taps.
ARG HARU_BASE=https://raw.githubusercontent.com/Live2D/CubismWebSamples/b1de66b0b1f1/Samples/Resources/Haru
RUN set -e; d=/usr/share/nginx/html/models/haru; mkdir -p $d/Haru.2048 $d/expressions $d/motions; cd $d \
 && for f in Haru.model3.json Haru.moc3 Haru.physics3.json Haru.pose3.json Haru.cdi3.json Haru.userdata3.json \
      Haru.2048/texture_00.png Haru.2048/texture_01.png \
      expressions/F01.exp3.json expressions/F02.exp3.json expressions/F03.exp3.json expressions/F04.exp3.json \
      expressions/F05.exp3.json expressions/F06.exp3.json expressions/F07.exp3.json expressions/F08.exp3.json \
      motions/haru_g_idle.motion3.json motions/haru_g_m15.motion3.json motions/haru_g_m26.motion3.json \
      motions/haru_g_m06.motion3.json motions/haru_g_m20.motion3.json motions/haru_g_m09.motion3.json; do \
      wget -q -O "$f" "$HARU_BASE/$f"; done
# Pinned third-party runtime: PixiJS 6.5.10 (MIT), pixi-live2d-display 0.4.0
# (MIT, Cubism 4 build), Live2D Cubism Core (Live2D "Redistributable Code").
COPY vendor /usr/share/nginx/html/vendor
COPY static/ /usr/share/nginx/html/
