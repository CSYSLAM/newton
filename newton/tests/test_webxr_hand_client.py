# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test the WebXR browser client without importing Newton or initializing CUDA.

Run directly with ``uv run --no-sync python newton/tests/test_webxr_hand_client.py``.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


class TestWebXRHandClient(unittest.TestCase):
    def test_explicit_hand_entry_requires_tracking_and_resumes_simulation(self):
        """Require hand tracking and surface failures instead of silent fallback."""
        script = r"""
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const elements = new Map();
const context = {document:{querySelector:(id)=>{
  if (!elements.has(id)) elements.set(id,{addEventListener(){},setAttribute(){},classList:{toggle(){}}});
  return elements.get(id);
}}, console, assert, Date, Math, Float32Array, WeakMap,
  navigator:{xr:{}}, XRWebGLLayer:function(){}, fetch:null};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1],"utf8").replace(/initialize\(\);\s*$/,""),context);
vm.runInContext(`
  latestScene = {handTrackingEnabled:true};
  geometryPromise = Promise.resolve({});
  connectSocket = () => {};
  ensureRenderer = () => {gl={makeXRCompatible:async()=>{}};};
  createHandPanelRenderer = () => ({dispose(){}});
  requestDesktopPreview = () => {};
  let requested, resumed = 0, ended = 0;
  const fakeSession = () => {
    const callbacks = {};
    return {visibilityState:"visible", updateRenderState(){}, requestAnimationFrame(){},
      requestReferenceSpace:async()=>({addEventListener(){}}),
      addEventListener:(name,callback)=>{callbacks[name]=callback;},
      end:async()=>{ended++; if(callbacks.end) callbacks.end();}};
  };
  navigator.xr.requestSession = async (mode,options) => {requested=options;return fakeSession();};
  fetch = async (url, options) => {
    assert.equal(url,"/control/resume"); assert.equal(options.method,"POST"); resumed++;return {ok:true};
  };
  globalThis.finished = (async () => {
    await enterVR({hands:true});
    assert.equal(requested.requiredFeatures.includes("hand-tracking"),true);
    assert.equal(resumed,1);
    assert.equal(inputMode,"hands");
    assert.equal(handIsFollowing("right"),false);
    assert.notEqual(handRecovery.right,null);
    assert.equal(typeof enableHand,"undefined");
    await session.end();
    navigator.xr.requestSession = async () => {throw new Error("hand tracking unavailable");};
    await enterVR({hands:true});
    assert.equal(session,null);
    assert.equal(resumed,1);
    assert.equal(xrStatus.textContent.includes("hand tracking unavailable"),true);
    navigator.xr.requestSession = async () => fakeSession();
    fetch = async () => ({ok:false});
    await enterVR({hands:true});
    assert.equal(session,null);
    assert.equal(ended,2);
    assert.equal(xrStatus.textContent.includes("Newton 未能退出待机"),true);
  })();
`,context);
context.finished.catch(error=>{console.error(error);process.exitCode=1;});
"""
        subprocess.run(["node", "-e", script, str(_ROOT / "newton/examples/assets/webxr_teleop/app.js")], check=True)

    def test_reload_does_not_pass_lock_to_launcher(self):
        """Prevent launcher daemons from keeping the reload lock indefinitely."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            systemctl = root / "systemctl"
            systemctl.write_text("#!/bin/sh\necho inactive\n")
            systemctl.chmod(0o755)
            start = root / "start.sh"
            start.write_text('#!/bin/sh\ntest ! -e "/proc/$$/fd/9"\n')
            start.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{root}:{os.environ['PATH']}",
                "XDG_RUNTIME_DIR": str(root),
                "XDG_STATE_HOME": str(root / "state"),
                "XDG_CACHE_HOME": str(root / "cache"),
                "NEWTON_WEBXR_DEVICE": "cpu",
                "NEWTON_WEBXR_START_SCRIPT": str(start),
                "NEWTON_WEBXR_RESTART_DELAY_SECONDS": "0",
            }
            result = subprocess.run(
                ["bash", str(_ROOT / "scripts/reload_quest_webxr_teleop.sh")],
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_separate_hand_sources_and_preserve_controller_buttons(self):
        """Exercise automatic following, source arbitration, clutching and recovery."""
        script = r"""
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const elements = new Map();
const context = {
  document: {querySelector: (id) => {
    if (!elements.has(id)) elements.set(id, {addEventListener(){}, setAttribute(){}, classList:{toggle(){}}});
    return elements.get(id);
  }}, console, assert, Date, Math, Float32Array, WeakMap, WebSocket: {OPEN: 1},
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8").replace(/initialize\(\);\s*$/, ""), context);
vm.runInContext(`
  latestScene = {handTrackingEnabled: true, camera: {}};
  viewerAnchorMatrix = modelMatrix([0,0,0], [0,0,0,1], [1,1,1]);
  referenceSpace = {};
  const optical = {handedness: "right", hand: new Map(HAND_JOINT_NAMES.map((name) => [name, {}])),
    gripSpace: {}, gamepad: {buttons: [{pressed:true, value:1}], axes: []}};
  const controller = {handedness: "right", gripSpace: {}, gamepad: {
    buttons: [{value:.7}, {}, {}, {}, {pressed:true, value:1}], axes:[0,0,0,0]}};
  const pose = {transform: {matrix: viewerAnchorMatrix, position:{x:0,y:0,z:0}}};
  const frame = {getJointPose: () => pose, getPose: () => pose};
  session = {inputSources: [controller, optical], visibilityState:"visible"};
  setHandInputMode();
  for (const time of [0, 50, 100, 150, 200]) updateOpticalHands(frame, time);
  assert.equal(handIsFollowing("right"), true);
  assert.equal(handIsFollowing("left"), false);
  globalThis.result = [];
  socket = {readyState:1, bufferedAmount:0, send:(value)=>result.push(JSON.parse(value))};
  sendControllerFrame(frame, 1, pose);
  globalThis.saved = result[0];
  session.inputSources = [controller];
  updateOpticalHands(frame, 250);
  globalThis.lost = handIsFollowing("right");
  session.inputSources = [optical, controller];
  updateOpticalHands(frame, 300);
  globalThis.returned = handIsFollowing("right");
  for (const time of [350, 400, 450, 500]) updateOpticalHands(frame, time);
  sendControllerFrame(frame, 2, pose);
  globalThis.rearmed = result[1];

  // Head pointing clutches without selecting any of the panel's options.
  handPanel = {update(){}};
  handPanelMatrix = modelMatrix([0,0,-1], [0,0,0,1], [.55,.64,1]);
  updateHandPanel(pose, 100);
  updateHandPanel(pose, 3000);
  assert.equal(handGazePaused, true);
  assert.equal(handIsFollowing("right"), false);
  assert.equal(handIsFollowing("left"), false);
  assert.equal(handRecordingRequest, 0);
  assert.equal(inputMode, "hands");
  sendControllerFrame(frame, 3, pose);
  assert.equal(result[2].hands.right.enabled, false);
  // Hands can leave the cameras while the operator deliberately adjusts them.
  session.inputSources = [controller];
  updateOpticalHands(frame);
  assert.equal(handIsFollowing("right"), false);
  const away = {transform:{matrix:modelMatrix([0,0,0], [0,Math.SQRT1_2,0,Math.SQRT1_2], [1,1,1])}};
  updateHandPanel(away, 3100);
  updateHandPanel(away, 3250);
  assert.equal(handGazePaused, true);
  // Looking back into the region cancels the exit timer.
  updateHandPanel(pose, 3280);
  updateHandPanel(away, 3300);
  updateHandPanel(away, 3501);
  updateOpticalHands(frame);
  assert.equal(handGazePaused, false);
  assert.equal(handResumePending.right, true);
  assert.equal(handActivation.right, 2);
  session.inputSources = [optical, controller];
  updateOpticalHands(frame);
  sendControllerFrame(frame, 4, pose);
  assert.equal(result[3].hands.right.enabled, true);
  assert.equal(result[3].hands.right.activation, 3);
  assert.equal(handIsFollowing("left"), false);
  updateOpticalHands(frame);
  assert.equal(handActivation.right, 3);
  // Status rows cannot toggle a hand off; a newly seen left hand joins automatically.
  for (const row of [1, 2, 3]) activateHandPanelRow(row);
  assert.equal(handIsFollowing("right"), true);
  const leftOptical = {...optical, handedness:"left"};
  session.inputSources = [optical, leftOptical, controller];
  updateOpticalHands(frame, 4000);
  assert.equal(handIsFollowing("left"), true);
  assert.equal(handActivation.left, 1);
  // Menu actions require an explicit ray selection event.
  const recordRay = {transform:{matrix:modelMatrix([0,-.096,0],[0,0,0,1],[1,1,1])}};
  selectHandPanel({inputSource:controller, frame:{getPose:()=>recordRay}});
  assert.equal(handRecordingRequest, 1);
  // Distinguish ungranted hand tracking from missing sources/occluded joints.
  session.enabledFeatures = ["local-floor"];
  session.inputSources = [controller];
  updateOpticalHands(frame);
  assert.equal(handInputStatus.right, "会话未启用手追踪");
  session.enabledFeatures = ["local-floor", "hand-tracking"];
  updateOpticalHands(frame);
  assert.equal(handInputStatus.right, "无手部输入");
  session.inputSources = [optical];
  updateOpticalHands({getJointPose:()=>null});
  assert.equal(handInputStatus.right, "骨架 0/25");
  assert.equal(trackedHands.right, null);
  updateOpticalHands({getJointPose:()=>{throw new Error("temporarily unavailable");}});
  assert.equal(handInputStatus.right, "读取异常");
  // A valid skeleton remains readable when looking away from the clutch area.
  updateHandPanel(away, 5000);
  updateOpticalHands(frame, 5000);
  assert.equal(handInputStatus.right, "骨架 25/25");
  assert.notEqual(trackedHands.right, null);
  assert.equal(xrSessionOptions(true).requiredFeatures.includes("hand-tracking"), true);
  assert.equal(xrSessionOptions(false).optionalFeatures.includes("hand-tracking"), true);

  // Ordinary loss sends no motion until 200 ms of stable input, without an enable action.
  for (const time of [5700, 5750, 5800, 5850, 5900]) updateOpticalHands(frame, time);
  const beforeRecovery = handActivation.right;
  session.inputSources = [];
  updateOpticalHands(frame, 6000);
  assert.equal(handIsFollowing("right"), false);
  session.inputSources = [optical];
  for (const time of [6010, 6060, 6110, 6160]) {
    updateOpticalHands(frame, time);
    assert.equal(handIsFollowing("right"), false);
    assert.equal(handActivation.right, beforeRecovery);
  }
  assert.equal(handPanelRows()[2].includes("等待稳定后接续"), true);
  sendControllerFrame(frame, 6160, pose);
  assert.equal(result[result.length - 1].hands.right.enabled, false);
  updateOpticalHands(frame, 6210);
  assert.equal(handIsFollowing("right"), true);
  assert.equal(handActivation.right, beforeRecovery + 1);
  sendControllerFrame(frame, 6210, pose);
  assert.equal(result[result.length - 1].hands.right.enabled, true);
  assert.equal(result[result.length - 1].hands.right.activation, beforeRecovery + 1);
  updateOpticalHands(frame, 6260);
  assert.equal(handActivation.right, beforeRecovery + 1);
  assert.equal(handIsFollowing("left"), false);

  // A new loss or a long frame gap restarts the stability interval.
  updateOpticalHands({getJointPose:()=>null}, 6300);
  updateOpticalHands(frame, 6350);
  updateOpticalHands(frame, 6400);
  updateOpticalHands({getJointPose:()=>null}, 6450);
  updateOpticalHands(frame, 6500);
  updateOpticalHands(frame, 6550);
  updateOpticalHands(frame, 6700);
  for (const time of [6750, 6800, 6850]) {
    updateOpticalHands(frame, time);
    assert.equal(handIsFollowing("right"), false);
  }
  updateOpticalHands(frame, 6900);
  assert.equal(handIsFollowing("right"), true);
  assert.equal(handActivation.right, beforeRecovery + 2);

  // Acknowledged backend loss also requires a new token; old acknowledgements cannot re-pause it.
  latestScene.handTrackingActivation = {right:handActivation.right};
  latestScene.handTrackingState = {right:"tracking-lost"};
  for (const time of [7000, 7050, 7100, 7150]) {
    updateOpticalHands(frame, time);
    assert.equal(handIsFollowing("right"), false);
  }
  updateOpticalHands(frame, 7200);
  updateOpticalHands(frame, 7250);
  assert.equal(handIsFollowing("right"), true);
  assert.equal(handActivation.right, beforeRecovery + 3);

  // Backend rejection holds motion, then automatically retries with a fresh baseline.
  latestScene.handTrackingActivation = {right:handActivation.right};
  latestScene.handTrackingState = {right:"invalid-tracking"};
  updateOpticalHands(frame, 7300);
  assert.equal(handIsFollowing("right"), false);
  assert.equal(handPanelRows()[2].includes("等待稳定后接续"), true);
  for (const time of [7350, 7400, 7450, 7500]) updateOpticalHands(frame, time);
  assert.equal(handIsFollowing("right"), true);
  assert.equal(handActivation.right, beforeRecovery + 4);
  // The backend may already have changed a missed loss/rejection status to paused.
  latestScene.handTrackingActivation = {right:handActivation.right};
  latestScene.handTrackingState = {right:"paused"};
  updateOpticalHands(frame, 7600);
  assert.equal(handIsFollowing("right"), false);
  for (const time of [7650, 7700, 7750, 7800]) updateOpticalHands(frame, time);
  assert.equal(handIsFollowing("right"), true);
  assert.equal(handActivation.right, beforeRecovery + 5);

  // Session interruptions and mode changes resume automatically once input is stable.
  session.visibilityState = "hidden";
  updateOpticalHands(frame, 7900);
  assert.equal(handIsFollowing("right"), false);
  session.visibilityState = "visible";
  for (const time of [8000, 8050, 8100, 8150, 8200]) updateOpticalHands(frame, time);
  assert.equal(handIsFollowing("right"), true);
  setHandInputMode();
  updateOpticalHands(frame, 8300);
  assert.equal(handIsFollowing("right"), false);
  setHandInputMode();
  for (const time of [8400, 8450, 8500, 8550, 8600]) updateOpticalHands(frame, time);
  assert.equal(handIsFollowing("right"), true);
  assert.equal(typeof enableHand, "undefined");
  assert.equal(handPanelRows().some(row => /启用|手动暂停/.test(row)), false);
  // Plug and soft-bag scenes advertise right-hand control only; T-shirt keeps both.
  latestScene.handTrackingSides = ["right"];
  session.inputSources = [optical, leftOptical, controller];
  updateOpticalHands(frame, 8700);
  assert.equal(handIsFollowing("right"), true);
  assert.equal(handIsFollowing("left"), false);
  assert.equal(trackedHands.left, null);
  assert.equal(handPanelRows()[1].includes("此场景未使用"), true);
  sendControllerFrame(frame, 8700, pose);
  assert.equal(result[result.length - 1].hands.left, undefined);
  assert.equal(result[result.length - 1].controllers.right.buttons[4].pressed, true);
  delete latestScene.handTrackingSides;
  for (const time of [8800, 8850, 8900, 8950, 9000]) updateOpticalHands(frame, time);
  assert.equal(handIsFollowing("left"), true);
`, context);
assert.equal(context.saved.controllers.right.triggerValue, .7);
assert.equal(context.saved.controllers.right.buttons[4].pressed, true);
assert.equal(context.saved.hands.right.joints.length, 25);
assert.equal(context.saved.hands.right.enabled, true);
assert.equal(context.saved.hands.right.activation, 1);
assert.equal(context.lost, false);
assert.equal(context.returned, false);
assert.equal(context.rearmed.hands.right.activation, 2);
"""
        subprocess.run(["node", "-e", script, str(_ROOT / "newton/examples/assets/webxr_teleop/app.js")], check=True)


if __name__ == "__main__":
    unittest.main()
